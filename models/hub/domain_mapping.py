import torch.nn as nn
from .components.vit_oro import (
    spherical_harmonic_basis,
    GeoINR
)

class dH_to_dT_conv(nn.Module):
    """ 
    learn cat(dT/dx, dT/dy) from cat(dH/dx, dH/dy) using simple conv net
    """
    def __init__(self, in_channels=2, out_channels=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),  # local receptive field
            nn.ReLU(),
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1)  # outputs 2 channels
        )

    def forward(self, x):
        return self.net(x)
    
class H_to_dT_conv_PE(nn.Module):
    """ 
    Learn cat(dT/dx, dT/dy) from H using GeoINR positional encoding + simple conv net
    INFO: Set in_channels=2 to
          learn cat(dT/dx, dT/dy) from cat(dH/dx, dH/dy) using simple conv net with Positional Encoding (GeoINR)
          !!! the Terrain vector in INR will compute second order derivs.. )
    """
    def __init__(self, 
                img_size= (534,534), 
                in_channels=1, 
                out_channels=2,
                n_sh_coeff = 16, ## square of int
                conv_start_size=16, ## <--- embed arch param
                conv_start_size_enc=8, ## <--- embed arch param
                siren_hidden=16, ## <--- embed arch param
                oro_path="dataset/CERRA-534/orography.npz",
        ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_sh_coeff, conv_start_size, kernel_size=3, padding=1),  # local receptive field -> now gets n_sh_coeff channels form the encoding. 
            nn.Tanh(),
            nn.Conv2d(conv_start_size, out_channels, kernel_size=3, padding=1),  # hidden layer
            #nn.Tanh(),
            #nn.Conv2d(2*conv_start_size, out_channels, kernel_size=3, padding=1)  # outputs 2 channels
        )


        H, W = img_size
        self.basis = spherical_harmonic_basis(H, W, int(n_sh_coeff**0.5)) #.view(n_sh_coeff, H, W)
        self.encoder = GeoINR(
            n_sh_coeff,
            self.basis,
            oro_path=oro_path,
            in_channels=in_channels,
            siren_hidden=siren_hidden,
            conv_start_size=conv_start_size_enc,
            slim=True,
            far=False
        )

    def forward(self, x):
        #print(x.shape) # [1, 2, H, W]
        x_enc = self.encoder(x)
        #print( x_enc.shape) # [1, 36, H, W]
        return self.net(x_enc)
    

# class dH_to_dT_conv_PositionalEncodingPretrained(nn.Module):


#     def __init__(self, in_channels=2, out_channels=2):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),  # local receptive field
#             nn.ReLU(),
#             nn.Conv2d(16, out_channels, kernel_size=3, padding=1)  # outputs 2 channels
#         )

#     def forward(self, x):
#         return self.net(x)
    


    


