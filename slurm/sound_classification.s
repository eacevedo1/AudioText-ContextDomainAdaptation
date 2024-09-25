#!/bin/bash

#SBATCH --job-name=sound_classification
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --mem=18GB
#SBATCH --output=/scratch/ea3418/sound_classification_%j.out

EMBEDDINGS_PATH="urbansound8k_1030.pt \
                 urbansound8k-20240923151200_1038.pt \
                 urbansound8k-20240923161033_1054.pt \
                 urbansound8k-20240923171455_1109.pt \
                 urbansound8k-20240923180514_1046.pt \
                 urbansound8k-20240923185657_1102.pt \
                 urbansound8k-20240923195207_1119.pt" 

TRAIN_MODES="zs tgap sv"

TEMPERATURES="0.2 0.5 0.8"

BACKGROUND_EMBEDDING_PATH="tau2019uas_1143.pt"

cd /scratch/ea3418/me-uyr-trans-exp/singularity-atm-da-51257214

singularity \
    exec \
    --nv \
    --overlay overlay-25GB-500K.ext3:rw \
    cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
    /bin/bash -c "
source /ext3/env.sh
conda activate atm-domain-adapt
cd /scratch/ea3418/me-uyr-trans-exp/AudioText-ContextDomainAdaptation
for embeddings_path in $EMBEDDINGS_PATH; do
    for train_mode in $TRAIN_MODES; do
        for temperature in $TEMPERATURES; do
            echo '-- Sound classification -- \$embeddings_path -- \$train_mode -- \$temperature --'
            python3 scripts/sound_classification.py --embeddings_path \$embeddings_path \
                                                    --dataset urbansound8k \
                                                    --mode \$train_mode \
                                                    --temperature \$temperature
                                                    --bg_embeddings_path $BACKGROUND_EMBEDDING_PATH \
                                                    --save_results True
        done
    done
done
echo 'Completed!'
"