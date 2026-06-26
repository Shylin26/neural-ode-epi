import torch
import torch.nn as nn
class EpiDecoder(nn.Module):
    def __init__(
            self,
            latent_dim:int,
            output_dim:int, 
            hidden_dim:int=64,
    ):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(latent_dim,hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim,output_dim),
            nn.Softplus(),
        )
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,std=0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self,z:torch.Tensor)->torch.Tensor:
        return self.net(z)
    

if __name__=="__main__":
    B,T=4,90
    latent_dim=32
    output_dim=1
    decoder=EpiDecoder(latent_dim,output_dim)
    z     = torch.randn(B, T, latent_dim)
    x_hat = decoder(z)

    print(f"Input  z:     {z.shape}")
    print(f"Output x_hat: {x_hat.shape}")
    print(f"All positive (Softplus): {(x_hat > 0).all()}")
