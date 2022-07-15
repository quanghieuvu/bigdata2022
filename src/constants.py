import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import torch.nn.functional as F
from PIL import Image
from shutil import rmtree
import math
import queue
from shutil import copyfile
import sys
import platform
from random import shuffle
from tqdm import trange, tqdm
import colorsys
from pathlib import Path
from collections import defaultdict, OrderedDict
import json
import glob
from utils import util_os

CWF = Path(__file__)
PROJECT_PATH = str(CWF.parent.parent) + '/'
# PROJECT_PATH = "/home/tnguyenhu2/alan_project/bigdata2022/"


DATA_PATH = PROJECT_PATH + 'input/'
TRAIN_VAL_PATH = util_os.gen_dir(PROJECT_PATH + 'train_val/')
CKPT_PATH = util_os.gen_dir(PROJECT_PATH + 'ckpt/')
RESULT_PATH = util_os.gen_dir(PROJECT_PATH + 'result/')
LOG_PATH = util_os.gen_dir(PROJECT_PATH + 'log/')

INPUT_SIZE = 512 
OUTPUT_SIZE = 512

IMAGENET_MEAN=[0.485, 0.456, 0.406]
IMAGENET_STD=[0.229, 0.224, 0.225]




#CUDA ENVIRONMENT
os.environ["CUDA_VISIBLE_DEVICES"] = '1' 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
