import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

from metrics.registry import LOSSFUNC

from .cross_entropy_loss import CrossEntropyLoss
from .consistency_loss import ConsistencyCos
from .bce_loss import BCELoss
from .am_softmax import AMSoftmaxLoss
from .am_softmax import AMSoftmax_OHEM
from .contrastive_regularization import ContrastiveLoss
from .l1_loss import L1Loss
from .patch_consistency_loss import PatchConsistencyLoss

