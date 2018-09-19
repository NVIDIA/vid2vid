python train.py --name pose2body_256p_g1 \
--dataroot datasets/pose --dataset_mode pose \
--input_nc 6 --ngf 64 --num_D 2 \
--resize_or_crop randomScaleHeight_and_scaledCrop --loadSize 384 --fineSize 256 \
--niter 5 --niter_decay 5 \
--no_first_img --n_frames_total 12 --max_frames_per_gpu 4 --max_t_step 4
