# AUDIO-TEXT DOMAIN ADAPATATION FOR SOUND CLASSIFICATION

Official release of the ICASSP 2025 paper: Domain Adaptation and Modality Gap in Audio-Text Models for Sound Classification 

![Method Proposed](figures/figura_v1.001.png "Proposed Method")

## Setup

### Clone repository

```
git clone git@github.com:eacevedo1/AudioText-ContextDomainAdaptation.git
```

### Create environment

```
conda create --name atm-domain-adapt python=3.9
conda activate atm-domain-adapt
pip install -r requirements.txt
```

### Download the pretrained models

```
# Create the directory if it doesn't exist
mkdir -p models/LAION-CLAP

# Download the model into models/LAION-CLAP
wget -P models/LAION-CLAP https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-fusion-best.pt
```

## Download Datasets

```
python3 download_dataset.py --dataset_name urbansound8k 
```

## Create Soundscapes

```
python3 soundscape_augmentations.py --folds 1,2,3 --parameters snr_dist=(const,6) n_soundscapes=2 bg=park
```

## Exctract Audio Embeddings 

```
python extract_embddings.py --dataset urbansound8k --path urbansound8k-20240705184401
```
