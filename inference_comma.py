import argparse
import os
import io
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
import PIL.Image as pil_image
from model import ARCNN, FastARCNN
import glob
import cv2
import tqdm

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='ARCNN', help='ARCNN or FastARCNN')
    parser.add_argument('--weights_path', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--outputs_dir', type=str, required=True)
    opt = parser.parse_args()

    if not os.path.exists(opt.outputs_dir):
        os.makedirs(opt.outputs_dir)

    if opt.arch == 'ARCNN':
        model = ARCNN()
    elif opt.arch == 'FastARCNN':
        model = FastARCNN()

    state_dict = model.state_dict()
    for n, p in torch.load(opt.weights_path, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model = model.to(device)
    model.eval()

    compressed_folders = sorted(glob.glob(os.path.join(opt.image_path, "2016*")))
    print(compressed_folders)

    for folder in compressed_folders:
        path = folder + '/h264/*.png'
        folder_name = os.path.basename(folder)
        imgs = sorted(glob.glob(path))
        print("Working on folder: ", folder_name)

        with torch.no_grad():
            for img in tqdm.tqdm(imgs):
                filename = os.path.basename(img).split('.')[0]
                input = pil_image.open(img).convert('RGB')
                input = transforms.ToTensor()(input).unsqueeze(0).to(device)

                pred = model(input)

                pred = pred.mul_(255.0).clamp_(0.0, 255.0).squeeze(0).permute(1, 2, 0).byte().cpu().numpy()
                # pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)

                output_subdir = opt.outputs_dir + folder_name
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)

                output_path = output_subdir + '/' + filename + '.png'
                cv2.imwrite(output_path, pred, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    
    print("Done Processing!")
