python train.py --name edge2face_512 \
--dataroot datasets/face/ --dataset_mode face \
--input_nc 15 --loadSize 512 --num_D 3 \
--gpu_ids 0,1,2,3,4,5,6,7 --n_gpus_gen 8 --batchSize 7 \
--niter 20 --niter_decay 20 --n_frames_total 12