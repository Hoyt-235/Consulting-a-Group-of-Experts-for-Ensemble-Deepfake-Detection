"""
eval pretained model.
"""
import os
import numpy as np
from os.path import join
import cv2
import random
import datetime
import time
import yaml
import pickle
from tqdm import tqdm
from copy import deepcopy
from PIL import Image as pil_image
from .metrics.utils import get_test_metrics
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
import src.models.jury.uia_vit_detector
import src.models.jury.spsl_detector
import src.models.jury.ucf_detector
import src.models.jury.stil_detector
from .dataset.abstract_dataset import DeepfakeAbstractBaseDataset
#from dataset.ff_blend import FFBlendDataset
#rom dataset.fwa_blend import FWABlendDataset
#from dataset.pair_dataset import pairDataset
import pandas as pd
from trainer import Trainer
from training.metrics.registry import DETECTOR
print("registered detectors:", DETECTOR.data)
from .metrics.base_metrics_class import Recorder
from collections import defaultdict

import argparse
from .logger import create_logger

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str, 
                    required=True,
                    help='path to detector YAML file')
parser.add_argument("--test_datasets", nargs="+")
parser.add_argument('--weights_path', type=str, 
                    required=True)

parser.add_argument('--preds_dir', type=str,
                    default='/workspace/predictions')
#parser.add_argument("--lmdb", action='store_true', default=False)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])


def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_datasets'] = test_name  # specify the current test dataset
        test_set = DeepfakeAbstractBaseDataset(
                config=config,
                mode='test', 
            )
        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set, 
                batch_size=config['test_batchSize'],
                shuffle=False, 
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
                drop_last=False
            )
        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_datasets']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def test_one_dataset(model, data_loader):
    prediction_lists = []
    feature_lists = []
    label_lists = []
    for i, data_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
        # get data
        data, label, mask, landmark = \
        data_dict['image'], data_dict['label'], data_dict['mask'], data_dict['landmark']
        label = torch.where(data_dict['label'] != 0, 1, 0)
        # move data to GPU
        data_dict['image'], data_dict['label'] = data.to(device), label.to(device)
        if mask is not None:
            data_dict['mask'] = mask.to(device)
        if landmark is not None:
            data_dict['landmark'] = landmark.to(device)

        # model forward without considering gradient computation
        predictions = inference(model, data_dict)
        logits      = predictions['cls']           # shape [B,2]
        probs       = torch.softmax(logits, dim=1) # shape [B,2]
        fake_probs  = probs[:, 1].cpu().numpy()
        label_lists += list(data_dict['label'].cpu().detach().numpy())
        prediction_lists += list(fake_probs)
        #feature_lists += list(predictions['feat'].cpu().detach().numpy())
    
    return np.array(prediction_lists), np.array(label_lists) #,np.array(feature_lists)
    

def test_epoch(model, test_data_loaders, csv_out_dir=None):
    """
    Runs through each DataLoader in `test_data_loaders`, collects predictions & labels,
    computes metrics, and (optionally) writes a CSV per dataset containing two columns:
      - prediction
      - label

    Args:
        model:               a PyTorch nn.Module in eval mode
        test_data_loaders:   dict mapping dataset name -> DataLoader
        csv_out_dir (str):   if not None, will save CSVs under this folder. One file per dataset:
                             <csv_out_dir>/<dataset_name>_preds.csv

    Returns:
        metrics_all_datasets: dict mapping dataset name -> metrics dict
    """
    model.eval()
    metrics_all_datasets = {}

    # Ensure output directory exists (if specified)
    if csv_out_dir is not None:
        os.makedirs(csv_out_dir, exist_ok=True)

    for dataset_name, loader in test_data_loaders.items():
        # 1) Run test_one_dataset to get NumPy arrays of predictions & labels
        #    (Modify test_one_dataset so it returns exactly (predictions_np, labels_np))
        predictions_np, labels_np = test_one_dataset(model, loader)

        # 2) If requested, write a CSV file with two columns: prediction, label
        if csv_out_dir is not None:
            csv_path = os.path.join(csv_out_dir, f"{dataset_name}.csv")
            df = pd.DataFrame({
                "prediction": predictions_np,
                "label":      labels_np
            })
            df.to_csv(csv_path, index=False,sep=';')
            tqdm.write(f"  -> Saved predictions+labels to: {csv_path}")

        # 3) Compute your metrics on these arrays
        data_dict = loader.dataset.data_dict  # assuming data_dict has 'image' names, etc.
        metric_one_dataset = get_test_metrics(
            y_pred    = predictions_np,
            y_true    = labels_np,
            img_names = data_dict.get('image', None)  # if get_test_metrics expects image names
        )
        metrics_all_datasets[dataset_name] = metric_one_dataset

        # 4) Print/log the metrics
        tqdm.write(f"Dataset: {dataset_name}")
        for metric_name, metric_val in metric_one_dataset.items():
            tqdm.write(f"    {metric_name}: {metric_val}")

    return metrics_all_datasets


@torch.no_grad()
def inference(model, data_dict):
    predictions = model(data_dict, inference=True)
    return predictions


def main():
    # parse options and load config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('/workspace/src/training/config/test_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)
    if 'label_dict' in config:
        config2['label_dict']=config['label_dict']
    weights_path = None
    # If arguments are provided, they will overwrite the yaml settings
    if args.test_datasets:
        config['test_datasets'] = args.test_datasets
    if args.weights_path:
        config['weights_path'] = args.weights_path
        weights_path = args.weights_path
    
    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True

    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config)
    
    # prepare the model (detector)
    print('===> Building model...')
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)
    epoch = 0
    if weights_path:
        try:
            epoch = int(weights_path.split('/')[-1].split('.')[0].split('_')[2])
        except:
            epoch = 0
        ckpt = torch.load(weights_path, map_location=device)
        model.load_state_dict(ckpt, strict=True)
        print('===> Load checkpoint done!')
    else:
        print('Fail to load the pre-trained weights')
    
    # start testing
    best_metric = test_epoch(model, test_data_loaders, args.preds_dir)
    print('===> Test Done!')

if __name__ == '__main__':
    main()
