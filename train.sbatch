#!/bin/bash
### SLURM specific parameters, for e.g:

##SBATCH -o ".train.out"
##SBATCH -e ".train.err"
#SBATCH -J train
#SBATCH -p gpu_p
#SBATCH -q gpu_normal
#SBATCH -c 5
#SBATCH --gres=gpu:1
#SBATCH --mem=8G

#SBATCH -t 24:00:00
## Un-comment i.e remove 2+(## symbol) from below lines, if you need email notifications for your jobs status.
##SBATCH --mail-user=your_email_address
##SBATCH --mail-type=ALL

### User's commands, apps, parameters, etc. for e.g:
export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4NDgyY2M3ZC0yNTU1LTRiNDgtYjM1Zi00YzA3MTM2NjJmMTIifQ=="
perun monitor src/model_training/train.py -c config/maskRCNN_train.yaml

### Exit
exit 0
