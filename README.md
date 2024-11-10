# TS-JEPA: Joint Embedding goes Temporal.

This is the official implementation of our "TS-JEPA" which was proposed on "Joint Embedding go Temporal".

## Requirements

Requirements can be installed automatically by running
```
pip install -r requirements.txt
```
The implementation is based on PyTorch.
We recommend to install PyTorch independently, according to the specifics of your system, as suggested on [PyTorch website](https://pytorch.org/get-started/locally/).
These packages are necessary to run the code:
- numpy
- pandas
- pytorch


## Setup
The code's folder should be divided into the following subfolders:
- data: contains the datasets
- config: contains the config files
- main: contains the main scripts and utils scripts to be used
- src: contains the implementation of the different models (Encoder, Predictor and Decoder)
- logs: contains the saved checkpoint of the models.

## Usage
To pretrain the TS-JEPA, the user should specify:
- The dataset
- Learning Rate
- Batch Size
- EMA momentum parameter (m)
- Ratio_patches (number of patches)
- Mask Ratio in the JEPA
- Parameters of the Encoder and Predictor

For our paper, we used the following command:

```python
python pretrain.py --data weather --batch_size 32 --lr 1e-07 --ema_momentum 0.998 --ratio_patches 10 --mask_ratio 0.7 --encoder_embed_dim 128 --encoder_nhead 2 --encoder_num_layers 1 --predictor_embed 128 --predictor_nhead 2 --predictor_num_layers 1
```

This command will create a checkpoint model in the Logs folder. To Evaluate the pre-trained model on predicting the last value downstream task, the user can do the following command:

```python
python eval_forecast_last_pred.py --data weather --batch_size 32 --lr 1e-04 --lr_pretrain 1e-07 --mask_ratio 0.7 --ema_pretrain 0.998 --ratio_patches 10 --checkpoint_to_use 5000 --pretrain_encoder_embed_dim 128 --pretrain_encoder_nhead 2 --pretrain_encoder_num_layers 1 --pretrain_encoder_kernel_size 3 --pretrain_decoder_embed_dim 128 --pretrain_decoder_nhead 2 --pretrain_decoder_num_layers 1
```

## Details

Additional details about the architecture and other elements are provided in the paper.

## Citing
If you find our proposed analysis useful for your research, please consider citing our paper.

For any additional questions/suggestions you might have about the code and/or the proposed analysis, please contact: ennadir@kth.se.
