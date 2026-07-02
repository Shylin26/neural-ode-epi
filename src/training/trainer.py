import logging
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.latent_ode import LatentODE, elbo_loss

log = logging.getLogger(__name__)

CKPT_DIR = Path(__file__).resolve().parents[2] / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

def kl_weight(epoch:int ,warmup_epochs:int )->float:
    return min(1.0, epoch / max(warmup_epochs, 1))

def solver_tolerance(epoch: int, warmup_epochs: int) -> tuple[float, float]:
    
    progress = min(1.0, epoch / max(warmup_epochs, 1))
    rtol = 1e-2 * (1 - progress) + 1e-4 * progress   # 1e-2 → 1e-4
    atol = 1e-3 * (1 - progress) + 1e-5 * progress   # 1e-3 → 1e-5
    return rtol, atol
class Trainer:
    def __init__(
        self,
        model:          LatentODE,
        train_dl:       DataLoader,
        val_dl:         DataLoader,
        lr:             float = 1e-3,
        weight_decay:   float = 1e-4,
        grad_clip:      float = 1.0,
        warmup_epochs:  int   = 50,
        n_epochs:       int   = 500,
        device:         str   = "cpu",
        use_wandb:      bool  = False,
    ):
        self.model         = model.to(device)
        self.train_dl      = train_dl
        self.val_dl        = val_dl
        self.grad_clip     = grad_clip
        self.warmup_epochs = warmup_epochs
        self.n_epochs      = n_epochs
        self.device        = device
        self.use_wandb     = use_wandb
        self.best_val_loss = float("inf")

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_epochs,
        )

        if use_wandb:
            import wandb
            wandb.watch(model, log="gradients", log_freq=50)

    def _batch_to_device(self, batch: dict) -> dict:
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        beta       = kl_weight(epoch, self.warmup_epochs)
        rtol, atol = solver_tolerance(epoch, self.warmup_epochs)

        self.model.rtol = rtol
        self.model.atol = atol

        total_loss = recon_loss = kl_loss = nfe_sum = 0.0
        n_batches  = 0
        nan_count=0
        grad_norm  = torch.tensor(0.0)

        for batch in self.train_dl:
            batch = self._batch_to_device(batch)

            self.optimizer.zero_grad()
            self.model.odefunc.nfe = 0

            x_hat, mu, logsigma = self.model(
                batch["times"][0],
                batch["values"],
                batch["mask"],
                batch["country_id"],
            )

            loss, recon, kl = elbo_loss(
                batch["values"], x_hat, batch["mask"],
                mu, logsigma, beta=beta,
            )

            if torch.isnan(loss):
                nan_count += 1
                continue

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip
            )
            self.optimizer.step()

            total_loss += loss.item()
            recon_loss += recon.item()
            kl_loss    += kl.item()
            nfe_sum    += self.model.odefunc.nfe
            n_batches  += 1

        self.scheduler.step()

        if n_batches == 0:
            return {"loss": float("nan"), "recon": float("nan"),
                    "kl": float("nan"), "nfe": 0, "beta": beta,
                    "rtol": rtol, "lr": self.scheduler.get_last_lr()[0],
                    "grad_norm": 0.0}

        return {
            "loss":      total_loss / n_batches,
            "recon":     recon_loss / n_batches,
            "kl":        kl_loss    / n_batches,
            "nfe":       nfe_sum    / n_batches,
            "beta":      beta,
            "rtol":      rtol,
            "lr":        self.scheduler.get_last_lr()[0],
            "grad_norm": grad_norm.item(),
        }

    @torch.no_grad()
    def val_epoch(self) -> dict:
        self.model.eval()
        total_loss = recon_loss = kl_loss = 0.0
        n_batches  = 0

        for batch in self.val_dl:
            batch = self._batch_to_device(batch)

            x_hat, mu, logsigma = self.model(
                batch["times"][0],
                batch["values"],
                batch["mask"],
                batch["country_id"],
            )

            loss, recon, kl = elbo_loss(
                batch["values"], x_hat, batch["mask"],
                mu, logsigma, beta=1.0,   # full KL at val
            )

            total_loss += loss.item()
            recon_loss += recon.item()
            kl_loss    += kl.item()
            n_batches  += 1

        if n_batches == 0:
            return {"val_loss": 0.0, "val_recon": 0.0, "val_kl": 0.0}

        return {
            "val_loss":  total_loss / n_batches,
            "val_recon": recon_loss / n_batches,
            "val_kl":    kl_loss    / n_batches,
        }

    def save_checkpoint(self, epoch: int, val_loss: float):
        path = CKPT_DIR / f"model_epoch{epoch:04d}_val{val_loss:.4f}.pt"
        torch.save({
            "epoch":      epoch,
            "model":      self.model.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "val_loss":   val_loss,
        }, path)
        log.info(f"Checkpoint saved: {path.name}")

    def fit(self):
        log.info(f"Training on {self.device} | "
                 f"{self.n_epochs} epochs | "
                 f"warmup {self.warmup_epochs} epochs")

        for epoch in range(self.n_epochs):
            t0       = time.time()
            train_m  = self.train_epoch(epoch)
            val_m    = self.val_epoch()
            elapsed  = time.time() - t0

            if epoch % 10 == 0:
                log.info(
                    f"Ep {epoch:4d} | "
                    f"loss {train_m['loss']:.4f} | "
                    f"recon {train_m['recon']:.4f} | "
                    f"kl {train_m['kl']:.4f} | "
                    f"nfe {train_m['nfe']:.0f} | "
                    f"beta {train_m['beta']:.2f} | "
                    f"val {val_m['val_loss']:.4f} | "
                    f"{elapsed:.1f}s"
                )

            if self.use_wandb:
                import wandb
                wandb.log({**train_m, **val_m, "epoch": epoch})

            if val_m["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_m["val_loss"]
                self.save_checkpoint(epoch, val_m["val_loss"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    import numpy as np
    import pandas as pd
    from src.data.dataset import make_dataloaders

    dates     = pd.date_range("2020-01-01", periods=400, freq="D")
    countries = ["IND", "USA", "GBR", "DEU", "BRA"]
    rows = []
    for iso in countries:
        for d in dates:
            rows.append({
                "iso_code": iso,
                "date":     d,
                "new_cases_smoothed_per_million": max(0, np.random.randn()),
                "observed": np.random.rand() > 0.1,
            })
    df = pd.DataFrame(rows)

    feature_cols = ["new_cases_smoothed_per_million"]
    train_dl, val_dl = make_dataloaders(
        df, feature_cols,
        train_end="2021-01-01",
        val_weeks=8,
        batch_size=4,
        window_days=90,
        stride_days=30,
    )

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    log.info(f"Device: {device}")

    model = LatentODE(
        input_dim   = 1,
        latent_dim  = 16,
        hidden_dim  = 32,
        country_dim = 8,
        n_countries = len(countries),
        solver      = "euler",
        rtol        = 1e-3,
        atol        = 1e-4,
    )

    trainer = Trainer(
        model         = model,
        train_dl      = train_dl,
        val_dl        = val_dl,
        lr            = 1e-3,
        warmup_epochs = 10,
        n_epochs      = 30,
        device        = device,
        use_wandb     = False,
    )

    trainer.fit()
    log.info("Smoke test complete.")
