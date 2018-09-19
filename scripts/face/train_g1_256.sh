python train.py --name edge2face_256_g1 \
--dataroot datasets/face/ --dataset_mode face \
--input_nc 15 --loadSize 256 --ngf 64 \
--max_frames_per_gpu 6 --n_frames_total 12 \
--niter 20 --niter_decay 20
