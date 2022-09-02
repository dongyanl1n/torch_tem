#!/bin/bash
#SBATCH --job-name=rnn_simple
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lindongy@mila.quebec
#SBATCH --output=experiments/slurm-%j.out

module load anaconda/3 python/3.7/cuda/11.0/cudnn/8.0/pytorch/1.7.0
source $HOME/testenv/bin/activate

python python rl_run_simple.py --size 5 --agent_type 'rnn' --save_dir 'experiments/2022-09-01/0' --num_episodes 20000 --save_model_freq 10000 --window_size 2000 --lr 0.0001 --num_neurons 512