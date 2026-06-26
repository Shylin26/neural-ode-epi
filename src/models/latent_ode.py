import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
from src.models.encoder import ODERNNEncoder
from src.models.odefunc import EpiODEFunc, EpiOdeFuncWrapper
from src.models.decoder import EpiDecoder

class LatentODE(nn.Module):
    def __init__(
            self,
        input_dim:   int,
        latent_dim:  int,
        hidden_dim:  int,
        country_dim: int,
        n_countries: int,
        n_layers:    int   = 3,
        dropout:     float = 0.1,
        solver:      str   = "dopri5",
        rtol:        float = 1e-4,
        atol:        float = 1e-5,

    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.solver     = solver
        self.rtol       = rtol
        self.atol       = atol
        self.country_emb = nn.Embedding(n_countries, country_dim)
        self.encoder = ODERNNEncoder(input_dim, hidden_dim, latent_dim)
        self.odefunc  = EpiODEFunc(latent_dim, hidden_dim, country_dim,
                                   n_layers, dropout)
        self.wrapper  = EpiOdeFuncWrapper(self.odefunc)
        self.decoder  = EpiDecoder(latent_dim, input_dim, hidden_dim)
    def reparametrize(
            self,
            mu: torch.Tensor,
            logsigma:torch.Tensor,
    )->torch.Tensor:
        eps=torch.randn_like(mu)
        return mu+eps*torch.exp(logsigma)
    def forward(
            self,
            times: torch.Tensor,
            values: torch.Tensor,
            mask: torch.Tensor,
            country_id: torch.Tensor,
    )->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        c_emb=self.country_emb(country_id)
        mu,logsigma=self.encoder(times,values,mask)
        z0=self.reparametrize(mu,logsigma)
        self.wrapper.set_country(c_emb)
        self.odefunc.nfe=0
        zt=odeint(
            self.wrapper,z0,times,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol,
        )
        zt=zt.permute(1,0,2)
        x_hat=self.decoder(zt)
        return x_hat,mu,logsigma
    @torch.no_grad()
    def predict(
        self,
        times:torch.Tensor,
        values:torch.Tensor,
        mask:torch.Tensor,
        country_id: torch.Tensor,
        n_samples:int=100,

    )->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        self.train()
        preds=[]
        for _ in range(n_samples):
            x_hat, _,_=self.forward(times,values,mask,country_id)
            preds.append(x_hat)
        preds=torch.stack(preds)
        mean=preds.mean(dim=0)
        lower=preds.quantile(0.05,dim=0)
        upper=preds.quantile(0.95,dim=0)
        return mean,lower,upper

def kl_divergence(mu:torch.Tensor,logsigma:torch.Tensor)->torch.Tensor:
    return -0.5 * (1 + 2*logsigma - mu**2 - torch.exp(2*logsigma)).sum(dim=-1).mean()

def elbo_loss(
        x: torch.Tensor,
        x_hat: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        logsigma: torch.Tensor,
        beta:float=1.0,
)->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    sq_err=(x-x_hat)**2
    masked=sq_err*mask
    recon=masked.sum()/(mask.sum()+1e-8)
    kl=kl_divergence(mu,logsigma)
    total=recon+beta*kl
    return total,recon,kl

if __name__=="__main__":
    B,T,D=4,90,1
    latent_dim=32
    hidden_dim=64
    country_dim=16
    n_countries=10

    model=LatentODE(
        input_dim=D,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        country_dim = country_dim,
        n_countries = n_countries,
        solver      = "euler",   
        rtol        = 1e-3,
        atol        = 1e-4,


    )
    times      = torch.linspace(0, 1, T)
    values     = torch.rand(B, T, D)
    mask       = (torch.rand(B, T, D) > 0.1).float()
    country_id = torch.randint(0, n_countries, (B,))

    # Forward pass
    x_hat, mu, logsigma = model(times, values, mask, country_id)
    print(f"x_hat:    {x_hat.shape}")
    print(f"mu:       {mu.shape}")
    print(f"logsigma: {logsigma.shape}")

    total, recon, kl = elbo_loss(values, x_hat, mask, mu, logsigma, beta=0.5)
    print(f"\nTotal loss: {total.item():.4f}")
    print(f"Recon loss: {recon.item():.4f}")
    print(f"KL loss:    {kl.item():.4f}")

    total.backward()
    print(f"\nBackward pass: OK")
    print(f"NFE: {model.odefunc.nfe}")

    mean, lower, upper = model.predict(times, values, mask, country_id,
                                       n_samples=10)
    print(f"\nUncertainty prediction:")
    print(f"mean:  {mean.shape}")
    print(f"lower: {lower.shape}")
    print(f"upper: {upper.shape}")
    print(f"Band width (mean): {(upper - lower).mean().item():.4f}")


    






