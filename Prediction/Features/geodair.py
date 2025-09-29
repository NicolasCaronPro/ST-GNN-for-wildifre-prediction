import io
import re
import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError

# === Endpoints ===
HTTP_OBJECT_HOST = "https://object.data.gouv.fr"          # HTTP path-style (CSV direct)
S3_ENDPOINT      = "https://object.infra.data.gouv.fr"    # S3 path-style (API)

BUCKET = "ineris-prod"
PREFIX = "lcsqa/concentrations-de-polluants-atmospheriques-reglementes/temps-reel"

def key_for(d: dt.date) -> str:
    return f"{PREFIX}/{d:%Y}/FR_E2_{d:%Y-%m-%d}.csv"

# ---------- A) HTTP path-style (recommandé en 1er) ----------
def get_bytes_http_path(d: dt.date) -> bytes:
    url = f"{HTTP_OBJECT_HOST}/{BUCKET}/{key_for(d)}"
    # Accept explicite pour éviter que le proxy ne renvoie l'UI
    r = requests.get(
        url,
        timeout=60,
        headers={"Accept": "text/csv,application/octet-stream;q=0.9,*/*;q=0.1"},
        allow_redirects=True,
    )
    r.raise_for_status()
    # garde-fou anti-HTML
    head = r.content[:256].lower()
    if b"<html" in head or b"<!doctype html" in head:
        raise RuntimeError(f"HTML renvoyé par {url}")
    return r.content

# ---------- B) S3 path-style (sans listing, UN-SIGNED) ----------
def get_bytes_s3_path(d: dt.date) -> bytes:
    s3 = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        region_name="us-east-1",
        config=Config(signature_version=UNSIGNED, s3={"addressing_style": "path"}),
    )
    k = key_for(d)
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=k)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in {"NoSuchKey", "NoSuchBucket"}:
            raise FileNotFoundError(f"s3://{BUCKET}/{k} introuvable") from e
        raise
    b = obj["Body"].read()
    if b[:64].lstrip().lower().startswith(b"<!doctype html") or b"<html" in b[:256].lower():
        raise RuntimeError("Réponse HTML via S3 path-style (proxy UI)")
    return b

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    # détecter/renommer la colonne date
    date_col = None
    for c in df.columns:
        cl = c.lower().replace("-", "_")
        if re.fullmatch(r"(date_?heure|date_?debut|datetime|timestamp)", cl):
            date_col = c
            break
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
        if date_col != "date_heure":
            df = df.rename(columns={date_col: "date_heure"})
        df = df.sort_values("date_heure")

    # harmoniser "polluant"
    pol = next((c for c in df.columns if c.lower() in {"polluant","code_polluant","nom_polluant"}), None)
    if pol and pol != "polluant":
        df = df.rename(columns={pol: "polluant"})
    return df

if __name__ == "__main__":
    tz = ZoneInfo("Europe/Paris")
    today = dt.datetime.now(tz).date()

    # Essaye aujourd'hui puis J-1 (selon l'heure de publication)
    for d in (today, today - dt.timedelta(days=1)):
        raw = None
        err_http = err_s3 = None
        # 1) HTTP path-style (object.data.gouv.fr)
        try:
            raw = get_bytes_http_path(d)
        except Exception as e:
            err_http = e
        # 2) Fallback S3 path-style (object.infra.data.gouv.fr)
        if raw is None:
            try:
                raw = get_bytes_s3_path(d)
            except Exception as e:
                err_s3 = e

        if raw is None:
            print(f"[{d}] HTTP error: {err_http}\n[{d}] S3 error: {err_s3}")
            continue

        df = pd.read_csv(io.BytesIO(raw), sep=";", encoding="utf-8", low_memory=False)
        df = normalize(df)
        print(f"OK {d} -> {df.shape}")
        if "polluant" in df.columns:
            df = df[df["polluant"].isin(["NO2", "PM2.5", "PM10", "O3"])]
        print(df.head(8))
        break