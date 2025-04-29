#!/bin/bash
### SLURM specific parameters, for e.g:

##SBATCH -o ".train.out"
##SBATCH -e ".train.err"
#SBATCH -J train
#SBATCH -p gpu_p
#SBATCH -q gpu_long
#SBATCH -c 5
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

#SBATCH -t 2-00
## Un-comment i.e remove 2+(## symbol) from below lines, if you need email notifications for your jobs status.
##SBATCH --mail-user=your_email_address
##SBATCH --mail-type=ALL

### User's commands, apps, parameters, etc. for e.g:
export CUDA_VISIBLE_DEVICES=0
export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4NDgyY2M3ZC0yNTU1LTRiNDgtYjM1Zi00YzA3MTM2NjJmMTIifQ=="
# perun monitor src/model_training/train.py -c config/maskRCNN_train.yaml
perun monitor src/model_training/train.py -c config/maskRCNN_train_no_augm.yaml
# python src/model_training/train.py -c config/example.yaml

### Exit
exit 0
