torchrun --nnodes=1 --nproc_per_node=8 test_imagenet_vid.py \
  --st_grouping_ckpt_path data_ckpt_logs/ckpt/checkpoint_Grouping_ImageNetVID_VideoMAE_15slots/mae_grouping_299.pth \
  --num_slots 15 --n_stmae_seeds 1 \
  --seed 287 \
  --output_folder evaluation_results/VideoMAE_STGrouping_15slots_8frames_299epoch