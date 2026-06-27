import hashlib 
import logging
from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm
log=logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR  = ROOT_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

OWID_URL  = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
OWID_PATH = RAW_DIR / "owid_covid_raw.csv"

TARGET_COUNTRIES = [
    "IND", "USA", "GBR", "DEU", "BRA",
    "ZAF", "JPN", "FRA", "ITA", "KOR",
]

KEEP_COLS = [
    "iso_code", "location", "date",
    "new_cases_smoothed_per_million",
    "new_deaths_smoothed_per_million",
    "reproduction_rate",
    "people_fully_vaccinated_per_hundred",
]


def _sha256(path: Path) -> str:
    """Compute SHA256 hash of a file for integrity verification."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_owid(force: bool = False) -> Path:
    """
    Download OWID COVID data if not already cached.
    
    Args:
        force: Re-download even if file exists.
    
    Returns:
        Path to raw CSV file.
    """
    if OWID_PATH.exists() and not force:
        log.info(f"Using cached file: {OWID_PATH}")
        return OWID_PATH

    log.info(f"Downloading OWID data from {OWID_URL}")
    response = requests.get(OWID_URL, stream=True)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))

    with open(OWID_PATH, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc="owid_covid"
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    checksum = _sha256(OWID_PATH)
    log.info(f"Downloaded. SHA256: {checksum}")

    return OWID_PATH


def load_owid(force_download: bool = False) -> pd.DataFrame:
    """
    Load, filter, and do basic cleaning of OWID data.
    
    Returns:
        DataFrame with columns: iso_code, location, date, 
        new_cases_smoothed_per_million, new_deaths_smoothed_per_million,
        reproduction_rate, people_fully_vaccinated_per_hundred
    """
    path = download_owid(force=force_download)

    df = pd.read_csv(path, parse_dates=["date"], low_memory=False)

    df = df[df["iso_code"].isin(TARGET_COUNTRIES)].copy()

    available = [c for c in KEEP_COLS if c in df.columns]
    df = df[available]

    df = df.sort_values(["iso_code", "date"]).reset_index(drop=True)

    log.info(f"Loaded {len(df):,} rows | "
             f"{df['iso_code'].nunique()} countries | "
             f"{df['date'].min().date()} → {df['date'].max().date()}")

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = load_owid()
    print(df.head(10))
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nCountries loaded:")
    print(df.groupby("iso_code")["date"].count())