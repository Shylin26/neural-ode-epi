import logging
from typing import Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader


log=logging.getLogger(__name__)

class EpiDataset(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            feature_cols:list[str],
            country_col: str="iso_code",
            date_col:str="date",
            window_days:int=180,
            stride_days:int=14,
    ):
        self.feature_cols=feature_cols
        self.window_days=window_days
        self.stride_days=stride_days
        countries=sorted(df[country_col].unique())
        self.country_to_id={c: i for i ,c in enumerate(countries)}
        self.id_to_country ={i:c for c, i in self.country_to_id.items()}
        self.n_countries=len(countries)
        log.info(f"Countries:{countries}")
        log.info(f"N countries :{self.n_countries}")
        self.samples=[]
        self._build_windows(df,country_col,date_col)
        log.info(f"Total windows: {len(self.samples)}")
    
    def _build_windows(
            self,
            df: pd.DataFrame,
            country_col:str,
            date_col:str,
    )->None:
        for iso,group in df.groupby(country_col):
            group=group.sort_values(date_col).reset_index(drop=True)
            country_id=self.country_to_id[iso]
            n=len(group)
            vals=group[self.feature_cols].values.astype(np.float32)
            obs=group["observed"].values.astype(np.float32)
            start=0
            while start + self.window_days<=n:
                end=start+self.window_days
                window_vals=vals[start:end]
                window_obs=obs[start:end]
                window_mask=np.stack(
                    [window_obs]*len(self.feature_cols),axis=1
                ).astype(np.float32)
                times=np.linspace(0.0,1.0,self.window_days,dtype=np.float32)
                self.samples.append({
                    "times":times,
                    "values":window_vals,
                    "mask":window_mask,
                    "country_id":country_id,
                    "iso_code":iso,
                    "start_idx":start,
                })
                start+=self.stride_days
                if n>self.window_days and start<n:
                    end=n
                    start=end-self.window_days
                    window_vals=vals[start:end]
                    window_obs=obs[start:end]
                    winodw_mask=np.stack([window_obs]*len(self.feature_cols),axis=1).astype(np.float32)
                    times=np.linspace(0.0,1.0,self.window_days,dtype=np.float32)
                    self.samples.append({
                        "times":times,
                        "values":window_vals,
                        "mask":window_mask,
                        "country_id":country_id,
                        "iso_code":iso,
                        "start_idx":start,
                    })
    
    def __len__(self)->int:
        return len(self.samples)
    def __getitem__(self,idx:int)->dict:
        s=self.samples[idx]
        return{
            "times":torch.tensor(s["times"]),
            "values":torch.tensor(s["values"]),
            "mask":torch.tensor(s["mask"]),
            "country_id": torch.tensor(self.country_to_id[s["iso_code"]],
                                       dtype=torch.long),
            "iso_code":   s["iso_code"],

        }

def make_dataloaders(
    df: pd.DataFrame,
    feature_cols:list[str],
    train_end:str,
    val_weeks:int=8,
    batch_size:int =4,
    window_days:int=180,
    stride_days:int=14,

)->tuple[DataLoader,DataLoader]:
    train_end_dt=pd.Timestamp(train_end)
    val_end_dt=train_end_dt+pd.Timedelta(weeks=val_weeks)
    df_train=df[df["date"]<=train_end_dt].copy()
    df_val = df[
    (df["date"] > train_end_dt) &
    (df["date"] <= val_end_dt)
    ].copy()
    log.info(f"Train: {df_train['date'].min().date()} → "
            f"{df_train['date'].max().date()} "
            f"({len(df_train):,} rows)")
    log.info(f"Val:   {df_val['date'].min().date()} → "
            f"{df_val['date'].max().date()} "
            f"({len(df_val):,} rows)")

    train_ds = EpiDataset(df_train, feature_cols,
                        window_days=window_days,
                        stride_days=stride_days)

    val_ds   = EpiDataset(df_val,   feature_cols,
                        window_days=min(window_days, val_weeks*7),
                        stride_days=val_weeks*7)  # no overlap in val

    train_dl = DataLoader(train_ds, batch_size=batch_size,
                        shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size,
                        shuffle=False, num_workers=0)

    return train_dl, val_dl


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Smoke test with synthetic data
    dates   = pd.date_range("2020-01-01", periods=400, freq="D")
    n       = len(dates)
    countries = ["IND", "USA", "GBR"]

    rows = []
    for iso in countries:
        for d in dates:
            rows.append({
                "iso_code": iso,
                "date":     d,
                "new_cases_smoothed_per_million": max(0, np.random.randn()),
                "observed": np.random.rand() > 0.1,  # 10% missing
            })

    df = pd.DataFrame(rows)

    feature_cols = ["new_cases_smoothed_per_million"]
    train_dl, val_dl = make_dataloaders(
        df, feature_cols,
        train_end="2021-01-01",
        val_weeks=8,
        batch_size=2,
        window_days=90,
    )

    print(f"\nTrain batches: {len(train_dl)}")
    print(f"Val batches:   {len(val_dl)}")

    batch = next(iter(train_dl))
    print(f"\nBatch keys:    {list(batch.keys())}")
    print(f"times shape:   {batch['times'].shape}")
    print(f"values shape:  {batch['values'].shape}")
    print(f"mask shape:    {batch['mask'].shape}")
    print(f"country_id:    {batch['country_id']}")
    print(f"iso_codes:     {batch['iso_code']}")    

        