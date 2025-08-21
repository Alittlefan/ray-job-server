from typing import Any, Sequence

import torch
from model.templates import MLP, ConvNet, ImpalaNet, ResNetBlock
from torch import nn

model_mapping = {
    "mlp": MLP,
    "conv_net": ConvNet,
    "resnet_block": ResNetBlock,
    "impala_net": ImpalaNet,
    "rnn": nn.RNN,
    "lstm": nn.LSTM,
    "gru": nn.GRU,
    "transformer": nn.Transformer,
    # 'auto_encoder': AutoEncoder,
}
default_configs = {
    "mlp": {
        "in_features": None,
        "out_features": 2,
        "num_cells": (32, 32),
        "activation_class": nn.Tanh,
        "activation_kwargs": None,
        "activate_last_layer": False,
    },
    "conv_net": {
        "in_channels": 3,
        "num_channels": (16, 32, 64),
        "kernel_sizes": (3, 3, 3),
        "strides": (1, 1, 1),
        "paddings": (0, 0, 0),
        "activation_class": nn.Tanh,
        "activation_kwargs": None,
        "norm_class": None,
        "norm_kwargs": None,
        "squash_last_layer": True,
    },
    "resnet_block": {
        "num_ch": 3,
        "activation_class": nn.Tanh,
        "activation_kwargs": None,
    },
    "impala_net": {
        "channels": (16, 32, 32),
        "activation_class": nn.Tanh,
        "activation_kwargs": None,
    },
    "rnn": {
        "input_size": 32,
        "hidden_size": 256,
        "num_layers": 1,
        "nonlinearity": "tanh",
        "bias": True,
        "batch_first": True,
    },
    "lstm": {
        "input_size": 32,
        "hidden_size": 256,
        "num_layers": 1,
        "nonlinearity": "tanh",
        "bias": True,
        "batch_first": True,
    },
    "gru": {
        "input_size": 32,
        "hidden_size": 256,
        "num_layers": 1,
        "nonlinearity": "tanh",
        "bias": True,
        "batch_first": True,
    },
    "transformer": {
        "d_model": 512,
        "nhead": 8,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "dim_feedforward": 2048,
        "dropout": 0.1,
        "activation": "relu",
    },
    # 'auto_encoder': {},
}


def get_model(model_configs: Sequence[dict[str, Any]]) -> nn.Module:
    layers = []
    for model_config in model_configs:
        layers.append(
            model_mapping[model_config["model_name"]](**model_config["model_config"])
        )
    model = nn.Sequential(*layers)
    return model


if __name__ == "__main__":
    test_model_configs = (
        {
            "model_name": "conv_net",
            "model_config": {
                "in_channels": 3,
                "num_channels": (16, 32, 64),
                "kernel_sizes": (3, 3, 3),
                "strides": (1, 1, 1),
                "paddings": (0, 0, 0),
                "activation_class": nn.Tanh,
                "activation_kwargs": None,
                "norm_class": None,
                "norm_kwargs": None,
                "squash_last_layer": True,
            },
        },
        {
            "model_name": "mlp",
            "model_config": {
                "in_features": None,
                "out_features": 1,
                "num_cells": (512, 256),
                "activation_class": nn.Tanh,
                "activation_kwargs": None,
                "activate_last_layer": False,
            },
        },
    )
    test_model = get_model(test_model_configs)
    dummy_data = torch.zeros([32, 3, 16, 16])
    out = test_model(dummy_data)
    print(test_model)
    print(out.shape)
