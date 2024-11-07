# AR-CNN, Fast AR-CNN

Credits: https://github.com/yjn870/ARCNN-pytorch

This repository is implementation of the "Deep Convolution Networks for Compression Artifacts Reduction". <br />
In contrast with original paper, It use RGB channels instead of luminance channel in YCbCr space and smaller(16) batch size.


## Requirements
- PyTorch
- tqdm
- Numpy
- Pillow

## Usages

### Train

Data augmentation option **--use_augmentation** performs random rescale and rotation. <br />

```bash
python main.py --arch "ARCNN" \     # ARCNN, FastARCNN
               --raw_images_dir "" \
               --comp_images_dir "" \
               --val_raw_images_dir "" \
               --val_comp_images_dir "" \
               --outputs_dir "" \
               --patch_size 24 \
               --batch_size 16 \
               --num_epochs 20 \
               --lr 5e-4 \
               --threads 8 \
               --seed 123      
```

### Test

Output results consist of image compressed with JPEG and image with artifacts reduced.

```bash
python inference.py --arch "ARCNN" \     # ARCNN, FastARCNN
               --weights_path "" \
               --image_path "" \
               --outputs_dir "" \      
```
