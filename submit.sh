#!/bin/bash
#SBATCH --job-name=og_mlp_1
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

#python rl_run.py --size 5 --agent_type 'mlp' --save_dir 'experiments/2022-09-03/og/1' --num_episodes 100000 --save_model_freq 20000 --window_size 10000 --lr 0.0005 --num_neurons 256 --load_existing_agent "experiments/2022-09-03/og/1/baseline_mlp_Epi49999.pt" --record_activity True

python rl_run.py --size 5 --num_episodes 50000 --lr 0.0005 --window_size 5000 --agent_type 'conv_rnn' --save_dir "experiments/2022-09-10" --save_model_freq 25000 --record_activity False --morris_water_maze False