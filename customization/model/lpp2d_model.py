import torch
import torch.nn as nn
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override


class SquashDims(nn.Module):
    def __init__(self, ndims_in: int = 3):
        super().__init__()
        self.ndims_in = ndims_in

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value.flatten(-self.ndims_in, -1)
        return value


class _ResNetBlock(nn.Module):
    def __init__(self, num_in, num_ch):
        super(_ResNetBlock, self).__init__()
        resnet_block = []
        resnet_block.append(nn.ReLU(inplace=False))
        resnet_block.append(
            nn.Conv2d(
                in_channels=num_in,
                out_channels=num_ch,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )
        resnet_block.append(nn.ReLU(inplace=False))
        resnet_block.append(
            nn.Conv2d(
                in_channels=num_ch,
                out_channels=num_ch,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )
        self.seq = nn.Sequential(*resnet_block)

    def forward(self, x):
        x += self.seq(x)
        return x


class _ConvNetBlock(nn.Module):
    def __init__(
        self, num_in, num_ch, kernel_size=3, stride=1, max_stride=2, padding=1
    ):
        super().__init__()

        conv = nn.Conv2d(
            in_channels=num_in,
            out_channels=num_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        mp = nn.MaxPool2d(kernel_size=kernel_size, stride=max_stride, padding=padding)
        self.feats_conv = nn.Sequential(conv, mp)
        self.resnet1 = _ResNetBlock(num_in=num_ch, num_ch=num_ch)
        self.resnet2 = _ResNetBlock(num_in=num_ch, num_ch=num_ch)

    def forward(self, x):
        x = self.feats_conv(x)
        x = self.resnet1(x)
        x = self.resnet1(x)
        return x


class UavCustomModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.original_space = getattr(obs_space, "original_space", obs_space)

        # CNN parameters
        raster_shape = (16, 16, 16)
        cnn_channels = (16, 32, 64)
        kernel_sizes = (3, 3, 3)
        strides = (1, 1, 1)
        vec_dim = 8  # dimension of vector observation
        vec_out = num_outputs

        # Build CNN encoder
        in_ch = raster_shape[0]
        layers = []
        for i in range(len(cnn_channels)):
            layers += [
                _ConvNetBlock(
                    in_ch,
                    cnn_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                )
            ]
            in_ch = cnn_channels[i]
        layers += [torch.nn.ReLU(inplace=True), SquashDims()]
        self.cnn_encoder = torch.nn.Sequential(*layers)

        # Calculate CNN output size using a dummy input
        dummy_raster = torch.ones(raster_shape)
        cnn_output = self.cnn_encoder(dummy_raster)
        cnn_output_size = cnn_output.size(0)

        # Post encoder (combines CNN and vector features)
        self.post_encoder = nn.Sequential(
            nn.Linear(vec_dim + cnn_output_size, vec_out),
            nn.ReLU(),
            nn.Linear(vec_out, vec_out),
            nn.ReLU(),
        )

        # Value function branch
        self.value_branch = nn.Sequential(
            nn.Linear(vec_out, 256), nn.ReLU(), nn.Linear(256, 1)
        )

        self._value = None

    def forward(self, input_dict, state, seq_lens):
        obs_dict = restore_original_dimensions(
            input_dict["obs"], self.original_space, "torch"
        )

        # Process vector and raster observations
        vector_obs = obs_dict["observation"]
        raster_obs = obs_dict["raster"]

        # Get CNN embedding
        cnn_embed = self.cnn_encoder(raster_obs)

        # Concatenate vector observation and CNN embedding
        combined = torch.concatenate(
            [
                vector_obs,
                cnn_embed,
            ],
            dim=-1,
        )

        # Get final embedding
        embedding = self.post_encoder(combined)

        # Calculate value
        self._value = self.value_branch(embedding).squeeze(1)

        return embedding, state

    def value_function(self):
        assert self._value is not None, "must call forward() first"
        return self._value


ModelCatalog.register_custom_model("lpp2d_model", UavCustomModel)
