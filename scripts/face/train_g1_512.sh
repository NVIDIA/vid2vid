python train.py --name edge2face_512_g1 \
--dataroot datasets/face/ --dataset_mode face \
--n_scales_spatial 2 --num_D 3 \
--input_nc 15 --loadSize 512 --ngf 64 \
--n_frames_total 6 --niter_step 2 --niter_fix_global 5 \
--load_pretrain checkpoints/edge2face_256_g1
