import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

from training.metrics.registry import DETECTOR
from utils import slowfast
from .uia_vit_detector import UIAViTDetector
from .spsl_detector import SpslDetector
from .ucf_detector import UCFDetector
from .stil_detector import STILDetector
