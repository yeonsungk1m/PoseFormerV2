#!/bin/bash
#SBATCH --job-name=PoseformerV2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --partition=P2
#SBATCH --output=./logs/%A_out.log
#SBATCH --error=./logs/%A_error.log

# python run_energy_poseformer.py -g 0 -k cpn_ft_h36m_dbb \
#   -frame 27 -frame-kept 3 -coeff-kept 3 -c /home/s3/kimyeonsung/PoseFormerV2/checkpoint \
#   --energy-weight 1e-5 --lr-loss 1e-4 --em-loss-type margin --em-margin-type mpjpe \
#   > out.log 2> error.log

python -u run_3dhp.py --gpu 0 \
    -f 27 -frame-kept 3 -coeff-kept 3 \
    --train 1 --lr 0.0007 -lrd 0.97 \
    -c checkpoint_3dhp_seal\
    > out_3dhp_seal.log 2> error_3dhp_seal.log
    CUDA_VISIBLE_DEVICES=0  stdbuf -oL -eL python -u run_poseformer.py -g 0 -k gt   -frame 81 -frame-kept 9 -coeff-kept 9   -c /workspace/PoseFormerV2/checkpoint_baseline   > logs/out_base_h36m.log 2> logs/err_base_h36m.log &
    CUDA_VISIBLE_DEVICES=1  stdbuf -oL -eL python -u run_energy_poseformer.py -g 0 -k gt   -frame 81 -frame-kept 9 -coeff-kept 9   -c /workspace/PoseFormerV2/checkpoint_seal_graph   --energy-weight 1e-5 --lr-loss 1e-4 --em-loss-type margin --em-margin-type mpjpe   > logs/out_seal_graph_h36m.log 2> logs/err_seal_graph_h36m.log &
    CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python -u run_3dhp.py --gpu 0   -f 81 -frame-kept 9 -coeff-kept 9   --train 1 --lr 0.0007 -lrd 0.97   -c /workspace/PoseFormerV2/checkpoint_3dhp_base   > logs/out_base_3dhp.log 2> logs/err_base_3dhp.log &
    CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python -u run_3dhp_seal.py --gpu 0   -f 81 -frame-kept 9 -coeff-kept 9   --train 1 --lr 0.0007 -lrd 0.97   --lr-loss 1e-6 --energy-weight 1e-3   --em-loss-type margin --em-margin-type mpjpe   -c /workspace/PoseFormerV2/checkpoint_3dhp_seal_margin   > logs/out_3dhp_seal_margin.log 2> logs/err_3dhp_seal_margin.log &


    CUDA_VISIBLE_DEVICES=0  stdbuf -oL -eL python -u run_poseformer.py -g 0 -k gt   -frame 27 -frame-kept 3 -coeff-kept 3   -c /workspace/PoseFormerV2/checkpoint_baseline_3   > logs/out_base_h36m_3.log 2> logs/err_base_h36m_3.log &
    CUDA_VISIBLE_DEVICES=1  stdbuf -oL -eL python -u run_energy_poseformer.py -g 0 -k gt   -frame 27 -frame-kept 3 -coeff-kept 3   -c /workspace/PoseFormerV2/checkpoint_seal_graph_3   --energy-weight 1e-5 --lr-loss 1e-4 --em-loss-type margin --em-margin-type mpjpe   > logs/out_seal_graph_h36m_3.log 2> logs/err_seal_graph_h36m_3.log &
    CUDA_VISIBLE_DEVICES=2 stdbuf -oL -eL python -u run_3dhp.py --gpu 0   -f 27 -frame-kept 3 -coeff-kept 3   --train 1 --lr 0.0007 -lrd 0.97   -c /workspace/PoseFormerV2/checkpoint_3dhp_base_3   > logs/out_base_3dhp_3.log 2> logs/err_base_3dhp_3.log &
    CUDA_VISIBLE_DEVICES=3 stdbuf -oL -eL python -u run_3dhp_seal.py --gpu 0   -f 27 -frame-kept 3 -coeff-kept 3   --train 1 --lr 0.0007 -lrd 0.97   --lr-loss 1e-6 --energy-weight 1e-3   --em-loss-type margin --em-margin-type mpjpe   -c /workspace/PoseFormerV2/checkpoint_3dhp_seal_margin_3   > logs/out_3dhp_seal_margin_3.log 2> logs/err_3dhp_seal_margin_3.log &