import os
import random
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datasets.train_data_functions import TrainData
from datasets.val_data_functions import ValData
from models import seg_network
from models.USCFormer import USCFormer
from loss import MSSSIM, CR
from utils import to_psnr, print_log, validation, adjust_learning_rate, findLastCheckpoint

plt.switch_backend('agg')

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('-learning_rate', help='Set the learning rate', default=2e-4, type=float)
parser.add_argument('-crop_size', help='Set the crop_size', default=[256, 256], nargs='+', type=int)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=16, type=int)
parser.add_argument('-epoch_start', help='Starting epoch number of the training', default=0, type=int)
parser.add_argument('-alpha_loss', help='Set the lambda in loss function', default=1, type=float)
parser.add_argument('-beta_loss', help='Set the lambda in loss function', default=0.02, type=float)
parser.add_argument('-gamma_loss', help='Set the lambda in loss function', default=0, type=float)
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-exp_name', help='directory for saving the networks of the experiment', type=str,default='checkpoints')
parser.add_argument('-seed', help='set random seed', default=19, type=int)
parser.add_argument('-num_epochs', help='number of epochs', default=200, type=int)
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

learning_rate = args.learning_rate
crop_size = args.crop_size
train_batch_size = args.train_batch_size
epoch_start = args.epoch_start
alpha_loss = args.alpha_loss
beta_loss = args.beta_loss
gamma_loss = args.gamma_loss
val_batch_size = args.val_batch_size
exp_name = args.exp_name
num_epochs = args.num_epochs
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

print('--- Hyper-parameters for training ---')
print('learning_rate: {}\ncrop_size: {}\ntrain_batch_size: {}\nval_batch_size: {}\nalpha_loss: {}\nbeta_loss: {}\ngamma_loss: {}'.format(learning_rate, crop_size, train_batch_size, val_batch_size, alpha_loss, beta_loss, gamma_loss))


train_data_dir = './data/cityscapes/train'
val_data_dir = './data/cityscapes/val'

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Define the network --- #
net = USCFormer()

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

# --- Build optimizer --- #
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# --- Multi-GPU --- #
net = net.to(device)
net = nn.DataParallel(net, device_ids=device_ids)

# --- Load the seg network weight --- #
checkpoint = torch.load(ckpt)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# --- Multi-GPU --- #
model = model.to(device)
model = nn.DataParallel(model, device_ids=device_ids)


# --- Load the network weight --- #
if os.path.exists('./{}/'.format(exp_name))==False:     
    os.mkdir('./{}/'.format(exp_name))  
try:
    net.load_state_dict(torch.load('./{}/best'.format(exp_name)))
    print('--- weight loaded ---')
except:
    print('--- no weight loaded ---')


# --- Define the L1 loss function --- #
l1_loss = nn.L1Loss()

# --- Define the ms_ssim loss function --- #
msssim_loss = MSSSIM()

# --- Load training data and validation/test data --- #
lbl_train_data_loader = DataLoader(TrainData(crop_size, train_data_dir), batch_size=train_batch_size, shuffle=True, num_workers=8)
val_data_loader = DataLoader(ValData(val_data_dir), batch_size=val_batch_size, shuffle=False, num_workers=8)

# --- Previous PSNR and SSIM in testing --- #
net.eval()

# load the lastest model
initial_epoch = findLastCheckpoint(save_dir='./{}'.format(exp_name))
if initial_epoch > 0:
    print('resuming by loading epoch %d' % initial_epoch)
    net.load_state_dict(torch.load(os.path.join('./{}'.format(exp_name), 'net_epoch%d.pth' % initial_epoch)))

old_val_psnr, old_val_ssim, old_val_ciede = validation(net, model, val_data_loader, device, exp_name)

print('foggycityscapes old_val_psnr: {0:.2f}, old_val_ssim: {1:.4f}, old_val_ciede: {2:.4f}'.format(old_val_psnr, old_val_ssim, old_val_ciede))

net.train()

total = sum(torch.numel(parameter) for parameter in net.parameters())
print("Number of parameter:%.2fM" % (total/1e6))


for epoch in range(initial_epoch, num_epochs):
#for epoch in range(epoch_start, num_epochs):
    psnr_list = []
    start_time = time.time()
    adjust_learning_rate(optimizer, epoch)
#-------------------------------------------------------------------------------------------------------------
    for batch_id, train_data in enumerate(lbl_train_data_loader):

        input_image, gt = train_data
        input_image = input_image.to(device)
        gt = gt.to(device)

        # --- Zero the parameter gradients --- #
        optimizer.zero_grad()

        # --- Forward + Backward + Optimize --- #
        net.train()
        model.eval()

        seg = model(input_image).to(device)
        pred_image = net(input_image, seg)

        L1_loss = l1_loss(pred_image, gt)
        ms_ssim = -msssim_loss(pred_image, gt)

        cr_loss = CR.ContrastLoss()
        contrast_loss = cr_loss(pred_image, gt, input_image)

        loss = L1_loss  + alpha_loss * ms_ssim + beta_loss * contrast_loss

        loss.backward()
        optimizer.step()

        # --- To calculate average PSNR --- #
        psnr_list.extend(to_psnr(pred_image, gt))

        if not (batch_id % 100):
            print('Epoch: {0}, Iteration: {1}'.format(epoch+1, batch_id))

    # --- Calculate the average training PSNR in one epoch --- #
    train_psnr = sum(psnr_list) / len(psnr_list)

    # --- Save the network parameters --- #
    torch.save(net.state_dict(), './{}/latest'.format(exp_name))
    torch.save(net.state_dict(), './{}/net_epoch{}.pth'.format(exp_name, str(epoch + 1)))

    # --- Use the evaluation model in testing --- #
    net.eval()

    val_psnr, val_ssim, val_ciede = validation(net, model, val_data_loader, device, exp_name)

    one_epoch_time = time.time() - start_time

    print("foggycityscapes")
    print_log(epoch+1, num_epochs, one_epoch_time, train_psnr, val_psnr, val_ssim, val_ciede, exp_name)

    # --- update the network weight --- #
    if val_psnr >= old_val_psnr:
        torch.save(net.state_dict(), './{}/best'.format(exp_name))
        print('model saved')
        old_val_psnr = val_psnr
