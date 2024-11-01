config = {
    # Data Path
    "path_train" : "./data/FordB/train_data.ts",
    "path_test" : "./data/FordB/test_data.ts",
    "path_save" : "./logs/output_model/",

    "regression": False,

    "wandb_project_name": "",
    "log_wandb" : False,

    # Printing and Logging settings
    "checkpoint_print": 20,

    # Loader
    "ratio_patches" : 20,
    "ratio_supervision": 1.0,

    # Optim
    "num_epochs": 101,
    "batch_size" : 512,
    "lr": 1e-05,

    # CNN Model
    "cnn_out_channels": [32, 64, 128],
    "cnn_kernel_size" : 3,
    "cnn_dense_dim" : 32,

    # Attention Only Model
    "attention_only_dense_dim" : 512,
    "attention_only_embed_dim": 512,

    # Transformer Model
    "embed_dim" : 128,
    "nhead" : 2,
    "num_layers": 1,
    "kernel_size" : 3,
    "embed_bias": True,
    "transformer_dense_dim": 64,
    "pooling": "Mean",

    # Pretrained Transformer -- Config
    "pretrain_encoder_embed_dim" : 256,
    "pretrain_encoder_nhead" : 4,
    "pretrain_encoder_num_layers": 2,
    "pretrain_encoder_kernel_size" : 3,
    "pretrain_encoder_embed_bias" : True,
    "pretrain_transformer_dense_dim" : 128,

    "pretrain_decoder_embed_dim" : 128,
    "pretrain_decoder_nhead" : 2,
    "pretrain_decoder_num_layers": 1,

    "checkpoint_to_use": 3000,
    "lr_pretrain": 1e-05,
    "batch_size_pretrain": 256,
    "mask_ratio" : 0.9,
    "ema_pretrain" : 0.999
}
