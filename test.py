import os
import random
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.val_data_functions import ValData
from models import seg_network
from models.USCFormer import USCFormer
from utils import validation

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-exp_name', help='directory for saving the networks of the experiment', type=str, default='checkpoints')
parser.add_argument('-seed', help='set random seed', default=19, type=int)
parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                    choices=['deeplabv3_resnet50', 'deeplabv3plus_resnet50',
                             'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                             'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet'], help='model name')
parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
parser.add_argument("--num_classes", type=int, default=19, help="num classer (default:19)")
parser.add_argument("--ckpt", type=str, default="./snapshots/best_deeplabv3plus_mobilenet_cityscapes_os16.pth",
                    help="restore from checkpoint")

args = parser.parse_args()

val_batch_size = args.val_batch_size
exp_name = args.exp_name
model = args.model
separable_conv = args.separable_conv
output_stride = args.output_stride
num_classes = args.num_classes
ckpt = args.ckpt

#set seed
seed = args.seed
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed) 
    print('Seed:\t{}'.format(seed))

# --- Set category-specific hyper-parameters  --- #
val_data_dir = './data/cityscapes/val'

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# --- Validation data loader --- #

val_data_loader = DataLoader(ValData(val_data_dir), batch_size=val_batch_size, shuffle=False, num_workers=8)


# --- Define the network --- #

net = USCFormer().cuda()

# --- Define the seg network --- #
model_map = {
    'deeplabv3_resnet50': seg_network.deeplabv3_resnet50,
    'deeplabv3plus_resnet50': seg_network.deeplabv3plus_resnet50,
    'deeplabv3_resnet101': seg_network.deeplabv3_resnet101,
    'deeplabv3plus_resnet101': seg_network.deeplabv3plus_resnet101,
    'deeplabv3_mobilenet': seg_network.deeplabv3_mobilenet,
    'deeplabv3plus_mobilenet': seg_network.deeplabv3plus_mobilenet
}
num_classes = 19
model = model_map[model](num_classes=num_classes, output_stride=output_stride)
if separable_conv and 'plus' in model:
    model.convert_to_separable_conv(model.classifier)

net = nn.DataParallel(net, device_ids=device_ids)

# --- Load the seg network weight --- #
checkpoint = torch.load(ckpt)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# --- Multi-GPU --- #
model = model.to(device)
model = nn.DataParallel(model, device_ids=device_ids)


# --- Multi-GPU --- #
net = net.to(device)
net = nn.DataParallel(net, device_ids=device_ids)

# --- Load the network weight --- #
net.load_state_dict(torch.load('./{}/USCFormer.pth'.format(exp_name)))

# --- Use the evaluation model in testing --- #
net.eval()
model.eval()

if os.path.exists('./results/{}/'.format(exp_name))==False:
    os.makedirs('./results/{}/'.format(exp_name))


print('--- Testing starts! ---')
start_time = time.time()
val_psnr, val_ssim, val_ciede = validation(net, model, val_data_loader, device, exp_name, save_tag=True)
end_time = time.time() - start_time
print('val_psnr: {0:.2f}, val_ssim: {1:.4f}, val_ciede: {2:.4f}'.format(val_psnr, val_ssim, val_ciede))
print('validation time is {0:.4f}'.format(end_time))
