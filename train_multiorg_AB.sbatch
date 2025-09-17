#!/bin/bash
### SLURM specific parameters, for e.g:

##SBATCH -o ".train_multi_1.out"
##SBATCH -e ".train_multi_1.err"
#SBATCH -J train_multiorg_AB
#SBATCH -p booster
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH -t 1-00
#SBATCH  -A hai_hmgu
## Un-comment i.e remove 2+(## symbol) from below lines, if you need email notifications for your jobs status.
##SBATCH --mail-user=your_email_address
##SBATCH --mail-type=ALL

### User's commands, apps, parameters, etc. for e.g:
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4NDgyY2M3ZC0yNTU1LTRiNDgtYjM1Zi00YzA3MTM2NjJmMTIifQ=="
# perun monitor src/model_training/train.py -c config/maskRCNN_train.yaml
python src/scripts/train.py -c config/fasterRCNN_train_multiorg_AB.yaml

### Exit
exit 0
