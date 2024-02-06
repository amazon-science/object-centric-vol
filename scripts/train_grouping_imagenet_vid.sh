torchrun --nnodes=1 --node_rank 0 --master_addr 127.0.0.1 --master_port 8899 --nproc_per_node=8 ./train_grouping_imagenet_vid.py
