# HW1: Autoencoder

## Download the dataset
```bash
kaggle datasets download -d julinmaloof/the-oxfordiiit-pet-dataset
unzip the-oxfordiiit-pet-dataset.zip
rm the-oxfordiiit-pet-dataset.zip
rm -rf annotations
```
## How to train the model
### Train/Test the autoencoder
```bash
python train_ae.py
python test_ae.py
```
### Train/Test the variational autoencoder
```bash
python train_vae.py
python test_vae.py
```
## Tesnorboard
```bash
tensorboard --logdir=runs
```
## Draw comparison results
```bash
python draw_comparison.py
```

