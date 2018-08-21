<img src='imgs/teaser.gif' align="right" width=360>

<br><br><br><br>

# vid2vid
### [[Project]](https://tcwang0509.github.io/vid2vid/) [[YouTube]](https://youtu.be/S1OwOd-war8) [[Paper]](https://tcwang0509.github.io/vid2vid/paper_vid2vid.pdf) [[ArXiv]](https://arxiv.org/)  <br>
Pytorch implementation of our method for high-resolution (e.g. 2048x1024) photorealistic video-to-video translation. It can be used for turning semantic label maps into photo-realistic videos, synthesizing people talking from edge maps, or generating human bodies from poses. <br><br>
[Video-to-Video Synthesis](https://tcwang0509.github.io/vid2vid/)  
 [Ting-Chun Wang](https://tcwang0509.github.io/)<sup>1</sup>, [Ming-Yu Liu](http://mingyuliu.net/)<sup>1</sup>, [Jun-Yan Zhu](http://people.csail.mit.edu/junyanz/)<sup>1,2</sup>, [Guilin Liu](https://liuguilin1225.github.io/)<sup>1</sup>, Andrew Tao<sup>1</sup>, [Jan Kautz](http://jankautz.com/)<sup>1</sup>, [Bryan Catanzaro](http://catanzaro.name/)<sup>1</sup>  
 <sup>1</sup>NVIDIA Corporation, <sup>2</sup>MIT  
 In arXiv, 2018.  

## Video-to-Video Translation
- Label-to-Streetview Results
<p align='center'>  
  <img src='imgs/city_change_styles.gif' width='440'/>  
  <img src='imgs/city_change_labels.gif' width='440'/>
</p>

- Edge-to-Face Results
<p align='center'>
  <img src='imgs/face.gif' width='440'/>
  <img src='imgs/face_multiple.gif' width='440'/>
</p>

- Pose-to-Body Results
<p align='center'>
  <img src='imgs/pose.gif' width='440'/>
</p>

## Prerequisites
- Linux or macOS
- Python 3
- NVIDIA GPU + CUDA cuDNN

## Getting Started
### Installation
- Install PyTorch and dependencies from http://pytorch.org
- Install python libraries [dominate](https://github.com/Knio/dominate) and requests.
```bash
pip install dominate
```
```bash
pip install requests
```
- Clone this repo:
```bash
git clone https://github.com/NVIDIA/vid2vid
cd vid2vid
```


### Testing
- An example Cityscapes video is included in the `datasets` folder.
- To generate the first frame for the model, there are 3 different ways. 
  - The first is using the real image by specifying `--use_real_img`. 
  - The second is to use another model which was trained on single images (e.g. pix2pixHD) by specifying `--use_single_G`. 
  - The third is forcing the model to also synthesize the first frame by specifying `--no_first_img`. This must be trained separately before inference.
  - Throughout the rest of the repo, we assume the second option is adopted. 
- First, download and compile a snapshot of the FlowNet2 repo from https://github.com/NVIDIA/flownet2-pytorch by running `python scripts/download_flownet2.py`.
- Please download the pre-trained Cityscapes model by:
  ```bash
  python scripts/download_models.py
  ```
- To test the model (`bash ./scripts/test_2048.sh`):
  ```bash
  #!./scripts/test_2048.sh
  python test.py --name label2city_2048 --loadSize 2048 --n_scales_spatial 3 --use_instance --fg --use_single_G
  ```
  The test results will be saved to a html file here: `./results/label2city_2048/test_latest/index.html`.

- A more compact model trained with 1 GPU is also provided. 
It has slightly worse performance, and only works on 1024 x 512 resolution.
  - Please download the model by
  ```bash
  python scripts/download_models_g1.py
  ```
  - To test the model (`bash ./scripts/test_1024_g1.sh`):
  ```bash
  #!./scripts/test_1024_g1.sh
  python test.py --name label2city_1024_g1 --loadSize 1024 --n_scales_spatial 3 --use_instance --fg --n_downsample_G 2 --use_single_G
  ```

- More example scripts can be found in the `scripts` directory.


### Dataset
- We use the Cityscapes dataset as an example. To train a model on the full dataset, please download it from the [official website](https://www.cityscapes-dataset.com/) (registration required).
- We apply a pre-trained segmentation algorithm to get the corresponding semantic maps (train_A) and instance maps (train_inst).
- Please put the obtained images under the `datasets` folder in the same way the example images are provided.


### Training
- First, download the FlowNet2 checkpoint file by running `python scripts/download_models_flownet2.py`.
- Training with 8 GPUs:
  - We adopt a coarse-to-fine approach, sequentially increasing the resolution from 512 x 256, 1024 x 512, to 2048 x 1024.
  - Train a model at 512 x 256 resolution (`bash ./scripts/train_512.sh`)
  ```bash
  #!./scripts/train_512.sh
  python train.py --name label2city_512 --gpu_ids 0,1,2,3,4,5,6,7 --n_gpus_gen 6 --n_frames_total 6 --use_instance --fg
  ```
  - Train a model at 1024 x 512 resolution (must train 512 x 256 first) (`bash ./scripts/train_1024.sh`):
  ```bash
  #!./scripts/train_1024.sh
  python train.py --name label2city_1024 --loadSize 1024 --n_scales_spatial 2 --num_D 3 --gpu_ids 0,1,2,3,4,5,6,7 --n_gpus_gen 4 --use_instance --fg --niter_step 2 --niter_fix_global 10 --load_pretrain checkpoints/label2city_512
  ```
- To view training results, please checkout intermediate results in `./checkpoints/label2city_1024/web/index.html`.
If you have tensorflow installed, you can see tensorboard logs in `./checkpoints/label2city_1024/logs` by adding `--tf_log` to the training scripts.

- Training with a single GPU: 
  - We trained our models using multiple GPUs. For convenience, we provide some sample training scripts (XXX_g1.sh) for single GPU users, up to 1024 x 512 resolution. Again a coarse-to-fine approach is adopted (256 x 128, 512 x 256, 1024 x 512). Performance is not guaranteed using these scripts.
  - For example, to train a 256 x 128 video with a single GPU (`bash ./scripts/train_256_g1.sh`)
  ```bash
  #!./scripts/train_256_g1.sh
  python train.py --name label2city_256_g1 --loadSize 256 --use_instance --fg --n_downsample_G 2 --num_D 1 --max_frames_per_gpu 6 --n_frames_total 6
  ```

### Training at full (2k x 1k) resolution
- To train the images at full resolution (2048 x 1024) requires 8 GPUs with at least 24G memory (`bash ./scripts/train_2048.sh`).
If only GPUs with 12G/16G memory are available, please use the script `./scripts/train_2048_crop.sh`, which will crop the images during training. Performance is not guaranteed with this script.

### Training with your own dataset
- If your input is a label map, please generate label maps which are one-channel whose pixel values correspond to the object labels (i.e. 0,1,...,N-1, where N is the number of labels). This is because we need to generate one-hot vectors from the label maps. Please also specity `--label_nc N` during both training and testing.
- If your input is not a label map, please just specify `--label_nc 0` and `--input_nc N` where N is the number of input channels (Default is 3 for RGB images).
- The default setting for preprocessing is `scaleWidth`, which will scale the width of all training images to `opt.loadSize` (1024) while keeping the aspect ratio. If you want a different setting, please change it by using the `--resize_or_crop` option. For example, `scaleWidth_and_crop` first resizes the image to have width `opt.loadSize` and then does random cropping of size `(opt.fineSize, opt.fineSize)`. `crop` skips the resizing step and only performs random cropping. `scaledCrop` crops the image while retraining the original aspect ratio. If you don't want any preprocessing, please specify `none`, which will do nothing other than making sure the image is divisible by 32.

## More Training/Test Details
- The way we train the model is as follows: suppose we have 8 GPUs, 4 for generators and 4 for discriminators, and we want to train 28 frames. Also assume each GPU can generate only one frame. The first GPU generates the first frame, and pass it to the next GPU, and so on. After the 4 frames are generated, they are passed to the 4 discriminator GPUs to compute the losses. Then the last generated frame becomes input to the next batch, and the next 4 frames in the training sequence are loaded into GPUs. This is repeated 7 times (4 x 7 = 28), to train all the 28 frames.
- Some important flags:
  - `n_gpus_gen`: the number of GPUs to use for generators (while the others are used for discriminators). We separate generators and discriminators into different GPUs since when dealing with high resolutions, even one frame cannot fit in a GPU. If the number is set to `-1`, there is no separation and all GPUs are used for both generators and discriminators (only works for low-res images).
  - `n_frames_G`: number of input frames to feed into the generator network; i.e., `n_frames_G - 1` is the number of frames we look into the past. Default is 3 (conditioned on two previous frames).
  - `n_frames_D`: number of frames to feed into the temporal discriminator. Default is 3.
  - `n_scales_spatial`: number of scales in the spatial domain. We train from the coarsest scale, and all the way to the finest scale. Default is 3.
  - `n_scales_temporal`: number of scales for the temporal discriminator. The finest scale takes in the sequence in the original frame rate. The coarser scales subsample the frames by a factor of `n_frames_D` before feeding the frames into the discriminator. For example, if `n_frames_D = 3` and `n_scales_temporal = 3`, the discriminator effectively sees 27 frames. Default is 3.
  - `max_frames_per_gpu`: number of frames in one GPU during training. If your GPU memory can fit more frames, try to make this number bigger. Default is 1.
  - `max_frames_backpropagate`: the number of frames that loss backpropagates to previous frames. For example, if this number is 4, the loss on frame n will backpropagate to frame n-3. Increasing this number will slightly improve the performance, but also cause training to be less stable. Default is 1.
  - `n_frames_total`: the total number of frames in a sequence we want to train with. We gradually increase this number during training.
  - `niter_step`: for how many epochs do we double `n_frames_total`. Default is 5.  
  - `niter_fix_global`: if this number if not 0, only train the finest spatial scale for this number of epochs before starting to finetune all scales.
  - `batchSize`: the number of sequences to train at a time. We normally set batchSize to 1 since usually one sequence is enough to occupy all GPUs. If you want to do batchSize > 1, currently only `batchSize == n_gpus_gen` is supported. 
  - `no_first_img`: if not specified, the model will assume the first frame is given and synthesize the successive frames. If specified, the model will also try to synthesize the first frame instead.
  - `fg`: if specified, use the foreground-background seperation model.
- For other flags, please see `options/train_options.py` and `options/base_options.py` for all the training flags; see `options/test_options.py` and `options/base_options.py` for all the test flags.


## Citation

If you find this useful for your research, please use the following.

```
@article{wang2018vid2vid,
  title={Video-to-Video Synthesis},
  author={Ting-Chun Wang and Ming-Yu Liu and Jun-Yan Zhu and Guilin Liu and Andrew Tao and Jan Kautz and Bryan Catanzaro},  
  journal={arXiv},
  year={2018}
}
```

## Acknowledgments
This code borrows heavily from [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
                                                                                                                                                                         