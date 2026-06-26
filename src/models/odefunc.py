import torch 
import torch.nn as nn
class FiLM(nn.Module):
    def __init__(self,condition_dim:int,hidden_dim:int):
        super().__init__()
        self.gamma=nn.Linear(condition_dim,hidden_dim)
        self.beta=nn.Linear(condition_dim,hidden_dim)
    
    def forward(self,h:torch.Tensor,condition:torch.Tensor)->torch.Tensor:
        return h*self.gamma(condition)+self.beta(condition)
    
class EpiODEFunc(nn.Module):
    def __init__(
            self,
            latent_dim:int,
            hidden_dim:int,
            country_dim:int,
            n_layers:int=3,
            dropout:float=0.1,
    ):
        super().__init__()
        self.latent_dim=latent_dim
        self.hidden_dim=hidden_dim
        self.nfe=0
        self.input_proj=nn.Linear(latent_dim,hidden_dim)
        self.layers=nn.ModuleList([
            nn.Linear(hidden_dim,hidden_dim)
            for _ in range(n_layers-1)
        ])
        self.dropout=nn.Dropout(dropout)
        self.film=FiLM(country_dim,hidden_dim)
        self.output_proj=nn.Linear(hidden_dim,latent_dim)
        self._initialise_weights()
    
    def _initialise_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,mean=0.0,std=0.01)
                nn.init.zeros_(m.bias)
    
    def forward(
            self,
            t: torch.Tensor,
            z:torch.Tensor,
            country_emb:torch.Tensor,
    )->torch.Tensor:
        self.nfe+=1
        h=torch.tanh(self.input_proj(z))
        h=self.film(h,country_emb)
        h=torch.tanh(h)
        for layer in self.layers:
            h=torch.tanh(layer(h))
            h=self.dropout(h)

        dz_dt=self.output_proj(h)
        return dz_dt

class EpiOdeFuncWrapper(nn.Module):
    def __init__(self,odefunc:EpiODEFunc):
        super().__init__()
        self.odefunc=odefunc
        self.country_emb=None
    
    def set_country(self,country_emb:torch.Tensor):
        self.country_emb=country_emb
    
    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        assert self.country_emb is not None, \
            "Call set_country() before odeint"
        return self.odefunc(t, z, self.country_emb)
if __name__=="__main__":
    batch=4
    latent_dim=32
    hidden_dim=64
    country_dim=16
    odefunc=EpiODEFunc(latent_dim,hidden_dim,country_dim)
    wrapper=EpiOdeFuncWrapper(odefunc)
    z=torch.randn(batch,latent_dim)
    country_emb=torch.randn(batch,country_dim)
    t=torch.tensor(0.0)
    wrapper.set_country(country_emb)
    dz_dt=wrapper(t,z)
    print(f"Input  z:     {z.shape}")
    print(f"Output dz/dt: {dz_dt.shape}")
    print(f"NFE:          {odefunc.nfe}")
    print(f"Max |dz/dt|:  {dz_dt.abs().max().item():.4f}")
    print(f"\nSmall initial derivatives: {dz_dt.abs().max().item() < 0.1}")



                        