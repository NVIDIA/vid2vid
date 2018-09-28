python train.py --name pose2body_1024p_g1 \
--dataroot datasets/pose --dataset_mode pose \
--input_nc 6 --n_scales_spatial 3 --num_D 4 --ngf 64 --ndf 32 \
--resize_or_crop randomScaleHeight_and_scaledCrop --loadSize 1536 --fineSize 1024 \
--no_first_img --n_frames_total 12 --max_t_step 4 --add_face_disc \
--niter_fix_global 3 --niter 5 --niter_decay 5 \
--lr 0.00005 --load_pretrain checkpoints/pose2body_512p_g1
