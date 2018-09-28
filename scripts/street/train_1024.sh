python train.py --name label2city_1024 \
--label_nc 35 --loadSize 1024 --n_scales_spatial 2 --num_D 3 --use_instance --fg \
--gpu_ids 0,1,2,3,4,5,6,7 --n_gpus_gen 4 \
--n_frames_total 4 --niter_step 2 \
--niter_fix_global 10 --load_pretrain checkpoints/label2city_512 --lr 0.0001
