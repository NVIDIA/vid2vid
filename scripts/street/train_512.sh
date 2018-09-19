python train.py --name label2city_512 \
--label_nc 35 --loadSize 512 --use_instance --fg \
--gpu_ids 0,1,2,3,4,5,6,7 --n_gpus_gen 6 \
--n_frames_total 6 --max_frames_per_gpu 2
