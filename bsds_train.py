import argparse
import os
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from model import ARCNN, FastARCNN
from dataset import BSDSDataset
from utils import AverageMeter
from torch.utils.tensorboard import SummaryWriter

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='ARCNN', help='ARCNN or FastARCNN')
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--val_images_dir', type=str, required=True)
    parser.add_argument('--outputs_dir', type=str, required=True)
    parser.add_argument('--jpeg_quality', type=int, default=10)
    parser.add_argument('--patch_size', type=int, default=24)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--use_augmentation', action='store_true')
    opt = parser.parse_args()

    if not os.path.exists(opt.outputs_dir):
        os.makedirs(opt.outputs_dir)

    torch.manual_seed(opt.seed)

    if opt.arch == 'ARCNN':
        model = ARCNN()
    elif opt.arch == 'FastARCNN':
        model = FastARCNN()

    model = model.to(device)
    criterion = nn.MSELoss()

    optimizer = optim.Adam([
        {'params': model.base.parameters()},
        {'params': model.last.parameters(), 'lr': opt.lr * 0.1},
    ], lr=opt.lr)

    dataset = BSDSDataset(opt.images_dir, opt.patch_size, opt.jpeg_quality, opt.use_augmentation)
    val_dataset = BSDSDataset(opt.val_images_dir, opt.patch_size, opt.jpeg_quality, train=False)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=opt.batch_size,
                            shuffle=True,
                            num_workers=opt.threads,
                            pin_memory=True,
                            drop_last=True)
    val_dataloader = DataLoader(dataset=val_dataset,
                            batch_size=opt.batch_size,
                            shuffle=False,
                            num_workers=opt.threads,
                            pin_memory=True,
                            drop_last=True)

    # best_val_loss = float('inf')

    # Initialize Tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join('./', 'logs'))    
    
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

        # # Validation phase
        # model.eval()
        # val_losses = AverageMeter()
        
        # with torch.no_grad():
        #     for data in val_dataloader:
        #         inputs, labels = data
        #         inputs = inputs.to(device)
        #         labels = labels.to(device)
                
        #         preds = model(inputs)
        #         loss = criterion(preds, labels)
                
        #         val_losses.update(loss.item(), len(inputs))

        # print(f'Validation loss: {val_losses.avg:.6f}')

        # # Log validation loss to Tensorboard
        # writer.add_scalar('Loss/val', val_losses.avg, epoch)

        # # Save model if validation loss improves
        # if val_losses.avg < best_val_loss:
        #     best_val_loss = val_losses.avg
        #     torch.save(model.state_dict(), 
        #               os.path.join(opt.outputs_dir, f'{opt.arch}_best_bsds_jpeg{opt.jpeg_quality}.pth'))

        torch.save(model.state_dict(), os.path.join(opt.outputs_dir, '{}_epoch_{}_bsds_jpeg{}.pth'.format(opt.arch, epoch, opt.jpeg_quality)))