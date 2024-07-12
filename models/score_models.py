# 2023 (c) LINE Corporation
# MIT License
import copy

import torch
import torchaudio
from hydra.utils import instantiate


class ScoreModelNCSNpp(torch.nn.Module):
    def __init__(
        self,
        num_sources,
        stft_args,
        backbone_args,
        transform="exponent",
        spec_abs_exponent=0.5,
        spec_factor=3.0,
        spec_trans_learnable=False,
    ):
        super().__init__()

        # infer input output channels of backbone from number of sources
        backbone_args.update(
            num_channels_in=2 * num_sources + 2, num_channels_out=2 * num_sources
        )
        
        print(backbone_args)
        self.backbone = instantiate(backbone_args)
        self.stft_args = stft_args
        self.stft = torchaudio.transforms.Spectrogram(power=None, **stft_args)
        self.stft_inv = torchaudio.transforms.InverseSpectrogram(**stft_args)

        self.transform = transform
        self.spec_abs_exponent = spec_abs_exponent
        self.spec_factor = spec_factor
        if spec_trans_learnable:
            self.spec_abs_exponent = torch.nn.Parameter(
                torch.tensor(self.spec_abs_exponent)
            )
            self.spec_factor = torch.nn.Parameter(torch.tensor(spec_factor))

    def transform_forward(self, spec):
        if self.transform == "exponent":
            if self.spec_abs_exponent != 1:
                # only do this calculation if spec_exponent != 1, otherwise it's quite a bit of wasted computation
                # and introduced numerical error
                e = abs(self.spec_abs_exponent)
                spec = spec.abs() ** abs(e) * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform == "log":
            spec = torch.log1p(spec.abs()) * torch.exp(1j * spec.angle())
            spec = spec * abs(self.spec_factor)
        elif self.transform == "none":
            spec = spec
        else:
            raise ValueError("transform must be one of 'exponent'|'log'|'none'")

        return spec

    def transform_backward(self, spec):
        if self.transform == "exponent":
            spec = spec / abs(self.spec_factor)
            if self.spec_abs_exponent != 1:
                e = abs(self.spec_abs_exponent)
                spec = spec.abs() ** (1 / e) * torch.exp(1j * spec.angle())
        elif self.transform == "log":
            spec = spec / abs(self.spec_factor)
            spec = (torch.exp(spec.abs()) - 1) * torch.exp(1j * spec.angle())
        elif self.transform == "none":
            spec = spec
        return spec

    def complex_to_real(self, x):
        # x: (batch, chan, freq, frames)
        x = torch.stack((x.real, x.imag), dim=1)  # (batch, 2, chan, freq, frames)
        x = x.flatten(start_dim=1, end_dim=2)  # (batch, 2 * chan, freq, frames)
        return x

    def real_to_complex(self, x):
        x = x.reshape((x.shape[0], 2, -1) + x.shape[2:])
        x = torch.view_as_complex(x.moveaxis(1, -1).contiguous())
        return x

    def pad(self, x):
        n_frames = x.shape[-1]
        rem = n_frames % 64
        if rem == 0:
            return x, 0
        else:
            pad = 64 - rem
            x = torch.nn.functional.pad(x, (0, pad))
            return x, pad

    def unpad(self, x, pad):
        if pad == 0:
            return x
        else:
            return x[..., :-pad]

    def adjust_length(self, x, n_samples):
        if x.shape[-1] < n_samples:
            return torch.nn.functional.pad(x, (0, n_samples - x.shape[-1]))
        elif x.shape[-1] > n_samples:
            return x[..., :n_samples]
        else:
            return x

    def pre_process(self, x):
        n_samples = x.shape[-1]
        x = torch.nn.functional.pad(
            x, (0, self.stft_args["n_fft"] - self.stft_args["hop_length"])
        )
        x = self.stft(x)
        x = self.transform_forward(x)
        x = self.complex_to_real(x)
        x, n_pad = self.pad(x)
        return x, n_samples, n_pad

    def post_process(self, x, n_samples, n_pad):
        x = self.unpad(x, n_pad)
        x = self.real_to_complex(x)
        x = self.transform_backward(x)
        x = self.stft_inv(x)
        x = self.adjust_length(x, n_samples)
        return x

    def forward(self, xt, time_cond, mix):
        """
        Args:
            x: (batch, channels, time)
            time_cond: (batch,)
        Returns:
            x: (batch, channels, time) same size as input
        """
        x = torch.cat((xt, mix), dim=1)
        x, n_samples, n_pad = self.pre_process(x)
        x = self.backbone(x, time_cond)
        x = self.post_process(x, n_samples, n_pad)
        return x

class TemporalConvNet(torch.nn.Module):
    def __init__(
        self, N, B, H, P, X, R, C, norm_type="gLN", causal=False, mask_nonlinear="relu"
    ):
        """Basic Module of tasnet.
        Args:
            N: Number of filters in autoencoder
            B: Number of channels in bottleneck 1 * 1-conv block
            H: Number of channels in convolutional blocks
            P: Kernel size in convolutional blocks
            X: Number of convolutional blocks in each repeat
            R: Number of repeats
            C: Number of speakers
            norm_type: BN, gLN, cLN
            causal: causal or non-causal
            mask_nonlinear: use which non-linear function to generate mask
        """
        super().__init__()
        # Hyper-parameter
        self.C = C
        self.mask_nonlinear = mask_nonlinear
        # Components
        # [M, N, K] -> [M, N, K]
        layer_norm = ChannelwiseLayerNorm(N)
        # [M, N, K] -> [M, B, K]
        bottleneck_conv1x1 = nn.Conv1d(N, B, 1, bias=False)
        # [M, B, K] -> [M, B, K]
        repeats = []
        for r in range(R):
            blocks = []
            for x in range(X):
                dilation = 2 ** x
                padding = (P - 1) * dilation if causal else (P - 1) * dilation // 2
                blocks += [
                    TemporalBlock(
                        B,
                        H,
                        P,
                        stride=1,
                        padding=padding,
                        dilation=dilation,
                        norm_type=norm_type,
                        causal=causal,
                    )
                ]
            repeats += [nn.Sequential(*blocks)]
        temporal_conv_net = nn.Sequential(*repeats)
        # [M, B, K] -> [M, C*N, K]
        mask_conv1x1 = nn.Conv1d(B, C * N, 1, bias=False)
        # Put together
        self.network = nn.Sequential(
            layer_norm, bottleneck_conv1x1, temporal_conv_net, mask_conv1x1
        )

    def forward(self, mixture_w):
        """Keep this API same with TasNet.
        Args:
            mixture_w: [M, N, K], M is batch size
        Returns:
            est_mask: [M, C, N, K]
        """
        M, N, K = mixture_w.size()
        score = self.network(mixture_w)  # [M, N, K] -> [M, C*N, K]
        score = score.view(M, self.C, N, K)  # [M, C*N, K] -> [M, C, N, K]
        if self.mask_nonlinear == "softmax":
            est_mask = F.softmax(score, dim=1)
        elif self.mask_nonlinear == "relu":
            est_mask = F.relu(score)
        elif self.mask_nonlinear == "sigmoid":
            est_mask = F.sigmoid(score)
        elif self.mask_nonlinear == "tanh":
            est_mask = F.tanh(score)
        else:
            raise ValueError("Unsupported mask non-linear function")
        return est_mask