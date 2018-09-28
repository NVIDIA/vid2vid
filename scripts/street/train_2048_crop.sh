python train.py --name label2city_2048_crop \
--label_nc 35 --loadSize 2048 --fineSize 1024 --resize_or_crop crop \
--n_scales_spatial 3 --num_D 4 --use_instance --fg \
--gpu_ids 0,1,2,3,4,5,6,7 --n_gpus_gen 4 \
--n_frames_total 4 --niter_step 1 \
--niter 5 --niter_decay 5 \
--niter_fix_global 5 --load_pretrain checkpoints/label2city_1024 --lr 0.00005
