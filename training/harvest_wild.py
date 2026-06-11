"""Stage-3 wild-table harvest -> data/wild/<name>.csv + manifest.json.

Sources: verified by the stage-3 wild hunter (2026-06-10). Each handler fetches a
BENCHMARK SAMPLE (the wild bench loads <=800 rows; we keep <=6k rows / ~10MB per
table). Headerless files get their published schemas. Licenses recorded in the
manifest. Giants (NHTSA FLAT_CMPL, full Open Food Facts) are sampled by streaming.

    uv run python -m training.harvest_wild
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WILD = ROOT / "data" / "wild"
UA = {"User-Agent": "scrubdata-research/1.0"}
MAX_ROWS = 6000

# FEC indiv header (fec.gov data dictionary, indiv_header_file.csv)
FEC_HEADER = ("CMTE_ID,AMNDT_IND,RPT_TP,TRANSACTION_PGI,IMAGE_NUM,TRANSACTION_TP,"
              "ENTITY_TP,NAME,CITY,STATE,ZIP_CODE,EMPLOYER,OCCUPATION,"
              "TRANSACTION_DT,TRANSACTION_AMT,OTHER_ID,TRAN_ID,FILE_NUM,MEMO_CD,"
              "MEMO_TEXT,SUB_ID").split(",")
# HM Land Registry price-paid column headings (published schema)
UKPP_HEADER = ("transaction_id,price,date_of_transfer,postcode,property_type,"
               "old_new,duration,paon,saon,street,locality,town_city,district,"
               "county,ppd_category,record_status").split(",")


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def _stream_lines(url: str, n: int, decode="utf-8", gz=False):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=180) as r:
        stream = gzip.GzipFile(fileobj=r) if gz else r
        out = []
        for raw in stream:
            out.append(raw.decode(decode, errors="replace"))
            if len(out) >= n:
                break
        return out


def _save(name: str, text: str) -> None:
    (WILD / f"{name}.csv").write_text(text, encoding="utf-8")
    print(f"  {name}: {len(text.splitlines())} lines saved", flush=True)


def _csv_text(rows: list[list[str]], header: list[str]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()


SOURCES = []


def source(name, domain, license_, url, regimes):
    def deco(fn):
        SOURCES.append({"name": name, "domain": domain, "license": license_,
                        "url": url, "regimes": regimes, "fetch": fn})
        return fn
    return deco


@source("bx_books", "books", "research (BX dump mirror)",
        "https://raw.githubusercontent.com/ashwanidv100/Recommendation-System---Book-Crossing-Dataset/master/BX-CSV-Dump/BX-Books.csv",
        "MOJI,PUNCT,NULL,CANON")
def bx(url):
    lines = _stream_lines(url, MAX_ROWS, decode="latin-1")
    rows = list(csv.reader(io.StringIO("".join(lines)), delimiter=";"))
    return _csv_text(rows[1:MAX_ROWS], rows[0])


@source("salary_survey", "survey", "public (TidyTuesday)",
        "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-05-18/survey.csv",
        "CANON,NULL,FMT")
def survey(url):
    return "\n".join(_stream_lines(url, MAX_ROWS))


@source("fec_indiv80", "political-finance", "US public domain",
        "https://www.fec.gov/files/bulk-downloads/1980/indiv80.zip",
        "PII,FMT,NULL,CANON,PUNCT")
def fec(url):
    zf = zipfile.ZipFile(io.BytesIO(_get(url)))
    fn = next(n for n in zf.namelist() if n.endswith(".txt"))
    rows = []
    for raw in io.TextIOWrapper(zf.open(fn), encoding="latin-1"):
        rows.append(raw.rstrip("\n").split("|"))
        if len(rows) >= MAX_ROWS:
            break
    return _csv_text(rows, FEC_HEADER)


@source("cms_doctors", "healthcare-providers", "US public domain (CMS)",
        "https://data.cms.gov/provider-data/api/1/datastore/query/mj5m-pzi6/0?limit=6000&format=csv",
        "PII,CANON,NULL,FMT")
def cms(url):
    return _get(url).decode("utf-8", errors="replace")


@source("acnc_charities", "nonprofits-au", "CC BY 3.0 AU",
        "https://data.gov.au/data/dataset/b050b242-4487-4306-abf5-07ca073e5594/resource/8fb32972-24e9-4c95-885e-7140be51be8a/download/datadotgov_main.csv",
        "FMT,NULL,CANON")
def acnc(url):
    return "\n".join(_stream_lines(url, MAX_ROWS))


@source("uk_price_paid", "real-estate-uk", "OGL UK v3 / HM Land Registry",
        "http://prod1.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-monthly-update-new-version.csv",
        "FMT,CANON,NULL")
def ukpp(url):
    lines = _stream_lines(url, MAX_ROWS)
    rows = list(csv.reader(io.StringIO("".join(lines))))
    return _csv_text(rows[:MAX_ROWS], UKPP_HEADER)


@source("irs_eo1", "nonprofits-us", "US public domain",
        "https://www.irs.gov/pub/irs-soi/eo1.csv", "CANON,NULL,FMT")
def irs(url):
    return "\n".join(_stream_lines(url, MAX_ROWS))


@source("glassdoor_jobs", "job-listings", "none (research only)",
        "https://raw.githubusercontent.com/PlayingNumbers/ds_salary_proj/master/glassdoor_jobs.csv",
        "NULL,PUNCT,FMT,CANON")
def glassdoor(url):
    return _get(url).decode("utf-8", errors="replace")


@source("paris_trees", "urban-fr", "ODbL",
        "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/les-arbres/exports/csv?limit=6000",
        "PUNCT,FMT,NULL,CANON")
def paris(url):
    text = _get(url).decode("utf-8-sig", errors="replace")
    rows = list(csv.reader(io.StringIO(text), delimiter=";"))
    return _csv_text(rows[1:MAX_ROWS], rows[0])


@source("online_retail", "ecommerce-uk", "CC BY 4.0 (UCI)",
        "https://raw.githubusercontent.com/databricks/Spark-The-Definitive-Guide/master/data/retail-data/all/online-retail-dataset.csv",
        "FMT,PUNCT,NULL,CANON")
def retail(url):
    return "\n".join(_stream_lines(url, MAX_ROWS))


@source("bl_flickr_books", "library", "public domain (BL)",
        "https://raw.githubusercontent.com/realpython/python-data-cleaning/master/Datasets/BL-Flickr-Images-Book.csv",
        "PUNCT,FMT,NULL,CANON")
def bl(url):
    return _get(url).decode("utf-8", errors="replace")


@source("open_food_facts", "food-products", "ODbL",
        "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz",
        "CANON,NULL,FMT,PUNCT")
def off(url):
    lines = _stream_lines(url, 3000, gz=True)          # 211 cols — keep it lean
    rows = [l.rstrip("\n").split("\t") for l in lines]
    return _csv_text(rows[1:], rows[0])


@source("ct_real_estate", "real-estate-us", "public (CT Open Data)",
        "https://data.ct.gov/resource/5mzw-sjtu.csv?$limit=6000", "NULL,FMT,CANON")
def ct(url):
    return _get(url).decode("utf-8", errors="replace")


def main() -> None:
    WILD.mkdir(parents=True, exist_ok=True)
    manifest = []
    for s in SOURCES:
        try:
            text = s["fetch"](s["url"])
            _save(s["name"], text)
            manifest.append({k: s[k] for k in ("name", "domain", "license", "url",
                                               "regimes")})
        except Exception as e:  # noqa: BLE001
            print(f"  {s['name']}: FAILED {type(e).__name__}: {str(e)[:90]}")
    json.dump(manifest, open(WILD / "manifest.json", "w"), indent=1)
    print(f"manifest: {len(manifest)} wild sources")


if __name__ == "__main__":
    main()
