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

#python rl_run_simple.py --size 5 --agent_type 'mlp' --save_dir 'experiments/2022-09-03/1' --load_existing_agent 'experiments/2022-09-01/1/baseline_mlp_Epi49999.pt' --num_episodes 100000 --save_model_freq 50000 --window_size 10000 --lr 0.00008 --num_neurons 512

python rl_run_simple.py --size 5 --agent_type 'og_mlp' --save_dir 'experiments/2022-09-03/og/1' --num_episodes 100000 --save_model_freq 20000 --window_size 10000 --lr 0.0005 --num_neurons 256 --load_existing_agent "experiments/2022-09-03/og/1/baseline_mlp_Epi49999.pt" --record_activity True
