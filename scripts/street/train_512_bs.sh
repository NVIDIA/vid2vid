python train.py --name label2city_512_bs \
--label_nc 35 --loadSize 512 --use_instance --fg \
--gpu_ids 0,1,2,3,4,5,6,7 --n_gpus_gen -1 \
--n_frames_total 6 --batchSize 15
