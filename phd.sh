#!/bin/bash
#

#SBATCH --job-name=phjob
#SBATCH --output=temp.txt
#
#SBATCH --cpus-per-task=20
#SBATCH --nodelist=treachery
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

echo `hostname`

# Activate your virtual environment
source /users/ikhomchenko/qutip-env/bin/activate

# Run your Python simulation
python ph_avg_d.py
