import torch 
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
class GRUCell(nn.Module):
    def __init__(self,input_dim:int,hidden_dim:int):
        super().__init__()
        self.gru=nn.GRUCell(input_dim,hidden_dim)
    def forward(self,x:torch.Tensor,h:torch.Tensor)->torch.Tensor:
        return self.gru(x,h)

class HiddenODEFunc(nn.Module):
    def __init__(self,hidden_dim:int):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim,hidden_dim),
        )
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,std=0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self,t:torch.Tensor,h:torch.Tensor)->torch.Tensor:
        return self.net(h)

class ODERNNEncoder(nn.Module):
    def __init__(
            self,
            input_dim:int,
            hidden_dim:int,
            latent_dim:int,
    ):
        super().__init__()
        self.hidden_dim=hidden_dim
        self.latent_dim=latent_dim
        self.odefunc=HiddenODEFunc(hidden_dim)
        self.gru_cell=GRUCell(input_dim,hidden_dim)
        self.to_mu=nn.Linear(hidden_dim,latent_dim)
        self.to_logsigma=nn.Linear(hidden_dim,latent_dim)
    
    def forward(
            self,
            times:torch.Tensor,
            values: torch.Tensor,
            mask:torch.Tensor,
    )->tuple[torch.Tensor,torch.Tensor]:
        B=values.shape[0]
        T=values.shape[1]
        h=torch.zeros(B,self.hidden_dim,device=values.device)
        for i in range(T-1,0,-1):
            t_i=times[i]
            t_prev=times[i-1]
            dt=(t_i-t_prev).abs()
            if dt >1e-6:
                t_span=torch.stack([t_i,t_prev])
                h=odeint(
                    self.odefunc,h,t_span,
                    method='euler',
                    rtol=1e-3,atol=1e-4,
                )[-1]
            obs_mask=mask[:,i,:].mean(dim=-1,keepdim=True)
            x_i=values[:,i,:]
            h_updated=self.gru_cell(x_i,h)
            h=obs_mask*h_updated+(1-obs_mask)*h
            h = torch.nan_to_num(h, nan=0.0)
        
        mu =self.to_mu(h)
        logsigma=self.to_logsigma(h)
        logsigma=torch.clamp(logsigma,min=-4,max=2)
        return mu,logsigma

if __name__ == "__main__":
    B, T, D    = 4, 90, 1
    hidden_dim = 64
    latent_dim = 32

    encoder = ODERNNEncoder(
        input_dim  = D,
        hidden_dim = hidden_dim,
        latent_dim = latent_dim,
    )

    times  = torch.linspace(0, 1, T)
    values = torch.randn(B, T, D)
    mask   = (torch.rand(B, T, D) > 0.1).float()  

    mu, logsigma = encoder(times, values, mask)

    print(f"Input:    values {values.shape}, mask {mask.shape}")
    print(f"Output:   mu {mu.shape}, logsigma {logsigma.shape}")
    print(f"mu range:       [{mu.min():.3f}, {mu.max():.3f}]")
    print(f"logsigma range: [{logsigma.min():.3f}, {logsigma.max():.3f}]")
    print(f"logsigma clamped to [-4, 2]: "
          f"{(logsigma >= -4).all() and (logsigma <= 2).all()}")