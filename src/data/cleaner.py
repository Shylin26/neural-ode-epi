import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MIN_DAYS = 100  

def flag_outliers(series: pd.Series, k: float = 3.0) -> pd.Series:
    """
    Flag outliers using IQR method.
    Returns boolean mask — True where value is an outlier.
    
    We FLAG, never silently remove. The mask goes into training
    so the model knows which observations to trust less.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return (series < lower) | (series > upper)


def clean_owid(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean OWID dataframe.
    
    Returns:
        df_clean:  cleaned dataframe
        df_report: per-country quality report
    """
    df = df.copy()
    report_rows = []

    
    target = "new_cases_smoothed_per_million"

    cleaned_countries = []

    for iso, group in df.groupby("iso_code"):
        group = group.copy().sort_values("date").reset_index(drop=True)
        n_total = len(group)

        
        neg_count = (group[target] < 0).sum()
        group[target] = group[target].clip(lower=0)

       
        group["observed"] = group[target].notna()
        n_missing = (~group["observed"]).sum()

        
        valid = group.loc[group["observed"], target]
        outlier_mask = flag_outliers(valid)
        group["outlier"] = False
        group.loc[group["observed"].values, "outlier"] = outlier_mask.values
        n_outliers = outlier_mask.sum()

        
        group[target] = (
            group[target]
            .fillna(method="ffill", limit=3)  
            .rolling(3, center=True, min_periods=1)
            .median()
        )

        n_valid = group["observed"].sum()
        if n_valid < MIN_DAYS:
            log.warning(f"{iso}: only {n_valid} valid days, dropping.")
            continue

        cleaned_countries.append(group)

        report_rows.append({
            "iso_code":   iso,
            "n_total":    n_total,
            "n_missing":  n_missing,
            "n_outliers": n_outliers,
            "n_negatives":neg_count,
            "n_valid":    n_valid,
            "pct_missing":round(100 * n_missing / n_total, 1),
        })

    df_clean  = pd.concat(cleaned_countries, ignore_index=True)
    df_report = pd.DataFrame(report_rows).set_index("iso_code")

    log.info(f"Cleaning complete. {len(cleaned_countries)} countries retained.")
    log.info(f"\n{df_report}")

    return df_clean, df_report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.data.loader import load_owid
    df = load_owid()
    df_clean, report = clean_owid(df)
    print("\nClean dataframe shape:", df_clean.shape)
    print("\nQuality report:")
    print(report)