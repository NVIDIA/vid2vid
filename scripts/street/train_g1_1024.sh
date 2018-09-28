python train.py --name label2city_1024_g1 \
--label_nc 35 --loadSize 896 --n_scales_spatial 3 --n_frames_D 2 \
--use_instance --fg --n_downsample_G 2 --num_D 3 \
--max_frames_per_gpu 1 --n_frames_total 4 \
--niter_step 2 --niter_fix_global 8 --niter_decay 5 \
--load_pretrain checkpoints/label2city_512_g1 --lr 0.0001
