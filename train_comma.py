import argparse
import os

from numpy import exp
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from model import ARCNN, FastARCNN
from dataset import CommaDataset
from utils import AverageMeter
from torch.utils.tensorboard import SummaryWriter

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian_window(window_size, 1.5)
        _2D_window = _1D_window.unsqueeze(1).mm(_1D_window.unsqueeze(0))
        _2D_window = _2D_window.float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def gaussian_window(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        window = self.create_window(self.window_size, channel)

        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)

        return 1 - self.ssim(img1, img2, window, self.window_size, channel, self.size_average)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='ARCNN', help='ARCNN or FastARCNN')
    parser.add_argument('--train_images_dir', type=str, required=True)
    parser.add_argument('--val_images_dir', type=str, required=True)
    parser.add_argument('--outputs_dir', type=str, required=True)
    parser.add_argument('--patch_size', type=int, default=24)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    opt = parser.parse_args()

    if not os.path.exists(opt.outputs_dir):
        os.makedirs(opt.outputs_dir)

    torch.manual_seed(opt.seed)

    if opt.arch == 'ARCNN':
        model = ARCNN()
    elif opt.arch == 'FastARCNN':
        model = FastARCNN()

    model = model.to(device)
    # criterion = SSIMLoss()
    criterion = nn.MSELoss()

    optimizer = optim.Adam([
        {'params': model.base.parameters()},
        {'params': model.last.parameters(), 'lr': opt.lr * 0.1},
    ], lr=opt.lr)

    dataset = CommaDataset(opt.train_images_dir, opt.patch_size)
    val_dataset = CommaDataset(opt.val_images_dir, opt.patch_size)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=opt.batch_size,
                            shuffle=True,
                            num_workers=opt.threads,
                            pin_memory=True,
                            drop_last=True)
    
    val_dataloader = DataLoader(dataset=val_dataset,
                            batch_size=32,
                            shuffle=False,
                            num_workers=opt.threads,
                            pin_memory=True,
                            drop_last=True)
    
    best_val_loss = float('inf')

    # Initialize Tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join('./', 'logs'))

    # Track validation and training losses using Tensorboard
    for epoch in range(opt.num_epochs):
        epoch_losses = AverageMeter()

        with tqdm(total=(len(dataset) - len(dataset) % opt.batch_size)) as _tqdm:
            _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, opt.num_epochs))
            for data in dataloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)

                loss = criterion(preds, labels)
                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _tqdm.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                _tqdm.update(len(inputs))

        # Log training loss to Tensorboard
        writer.add_scalar('Loss/train', epoch_losses.avg, epoch)

        # Validation phase
        model.eval()
        val_losses = AverageMeter()
        
        with torch.no_grad():
            for data in val_dataloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                preds = model(inputs)
                loss = criterion(preds, labels)
                
                val_losses.update(loss.item(), len(inputs))

        print(f'Validation loss: {val_losses.avg:.6f}')

        # Log validation loss to Tensorboard
        writer.add_scalar('Loss/val', val_losses.avg, epoch)

        # Save model if validation loss improves
        if val_losses.avg < best_val_loss:
            best_val_loss = val_losses.avg
            torch.save(model.state_dict(), 
                      os.path.join(opt.outputs_dir, f'{opt.arch}_best_comma_mse.pth'))

        # torch.save(model.state_dict(), os.path.join(opt.outputs_dir, '{}_epoch_{}_clean.pth'.format(opt.arch, epoch)))

    # Close the Tensorboard writer
    writer.close()

        
