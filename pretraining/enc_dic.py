import torch
from torch import nn


class Mlp(nn.Module):
    def __init__(self, input_shape):
        super(mlp, self).__init__()
        self.units = [1024, 1024, 512]
        self.activation = nn.ReLU
        self.network = nn.Sequential(
            nn.Linear(input_shape, self.units[0]),
            self.activation,
            nn.Linear(self.units[0], self.units[1]),
            self.activation,
            nn.Linear(self.units[1], self.units[2]),
            self.activation,
        )
        self.init_params()

    def __call__(self, x):
        return self.network(x)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.he_uniform_(m.weight)
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)
        return


class Encoder(nn.Module):
    def __init__(self, ase_latent_shape):
        super(encoder, self).__init__()
        self.mlp_out_size = 512  # Define this according to your needs
        self._ase_latent_shape = ase_latent_shape  # Define this according to your needs
        ENC_LOGIT_INIT_SCALE = 0.1  # Define this constant according to your needs
        self._enc = nn.Sequential(
            nn.Linear(self.mlp_out_size, self._ase_latent_shape[-1]),
        )
        self.init_params()

    def __call__(self, x):
        return self._enc(x)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.uniform_(
                    m.weight, -ENC_LOGIT_INIT_SCALE, ENC_LOGIT_INIT_SCALE)
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)
        return


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.mlp_out_size = 512  # Define this according to your needs
        DISC_LOGIT_INIT_SCALE = 1.0  # Define this constant according to your needs
        self.discriminator = nn.Sequential(
            nn.Linear(self.mlp_out_size, 1),
        )
        self.init_params()

    def __call__(self, x):
        return self.discriminator(x)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.uniform_(
                    m.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)
        return
