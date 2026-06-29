import logging 
import torch 
import pandas as pd
from src.data.loader import load_owid
from src.data.cleaner import clean_owid
from src.data.normalizer import Normalizer
from src.data.dataset import make_dataloaders
from src.models.latent_ode import LatentODE
from src.training.trainer import Trainer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
df          = load_owid()
df_clean, _ = clean_owid(df)
norm        = Normalizer()
df_norm     = norm.fit_transform(df_clean, "new_cases_smoothed_per_million")

train_dl, val_dl = make_dataloaders(
    df_norm,
    feature_cols = ["new_cases_smoothed_per_million"],
    train_end    = "2022-01-01",
    val_weeks    = 8,
    batch_size   = 4,
    window_days  = 90,
    stride_days  = 14,
)
device = "cpu"
log.info(f"Device: {device}")

model = LatentODE(
    input_dim   = 1,
    latent_dim  = 32,
    hidden_dim  = 64,
    country_dim = 16,
    n_countries = 10,
    n_layers    = 3,
    dropout     = 0.1,
    solver      = "euler",
    rtol        = 1e-3,
    atol        = 1e-4,
)


total_params = sum(p.numel() for p in model.parameters())
log.info(f"Model parameters: {total_params:,}")
trainer = Trainer(
    model         = model,
    train_dl      = train_dl,
    val_dl        = val_dl,
    lr            = 1e-3,
    weight_decay  = 1e-4,
    grad_clip     = 1.0,
    warmup_epochs = 50,
    n_epochs      = 200,
    device        = device,
    use_wandb     = False,
)

trainer.fit()
