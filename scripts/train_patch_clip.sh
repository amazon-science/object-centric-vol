
python train.py --dist-url 'tcp://IP_OF_NODE0:FREEPORT' \
  --dist-backend 'nccl' \
  --multiprocessing-distributed \
  --world-size 4 \
  --rank 0 \
  --data /home/ubuntu/ImageNet/imagenet \
  --epochs 200 \
  --lr 1.0 \
  --batch-size 4096

python train.py --dist-url 'tcp://IP_OF_NODE0:FREEPORT' \
  --dist-backend 'nccl' \
  --multiprocessing-distributed \
  --world-size 4 \
  --rank 1 \
  --data /home/ubuntu/ImageNet/imagenet \
  --epochs 200 \
  --lr 1.0 \
  --batch-size 4096

python train.py --dist-url 'tcp://IP_OF_NODE0:FREEPORT' \
  --dist-backend 'nccl' \
  --multiprocessing-distributed \
  --world-size 4 \
  --rank 2 \
  --data /home/ubuntu/ImageNet/imagenet \
  --epochs 200 \
  --lr 1.0 \
  --batch-size 4096

python train.py --dist-url 'tcp://IP_OF_NODE0:FREEPORT' \
  --dist-backend 'nccl' \
  --multiprocessing-distributed \
  --world-size 4 \
  --rank 3 \
  --data /home/ubuntu/ImageNet/imagenet \
  --epochs 200 \
  --lr 1.0 \
  --batch-size 4096

