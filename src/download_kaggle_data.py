"""Download the competition data and description pages from Kaggle.

Writes images and CSVs to `data/raw/` (gitignored) plus `kaggle_meta/pages/*.md`
and `kaggle_meta/files_metadata.json` (committed).

Usage:
    conda run -n museumscat-challenge python src/download_kaggle_data.py
"""

import csv, json, os, re, zipfile
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    # Imported lazily in main(): importing the kaggle package authenticates on import.
    from kaggle.api.kaggle_api_extended import KaggleApi

COMPETITION = "museumscat-specimen-collection-annotation-task"

RAW_DIR = os.path.join("data", "raw")
META_DIR = "kaggle_meta"
PAGES_DIR = os.path.join(META_DIR, "pages")
FILES_METADATA_PATH = os.path.join(META_DIR, "files_metadata.json")

class KaggleFile(TypedDict):
    name: str
    size: int

class FileMetadata(KaggleFile, total=False):
    columns: list[str]

def slugify_page_name(name: str) -> str:
    slug = name.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")

def read_csv_columns(csv_path: str) -> list[str]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return next(csv.reader(f))

def build_files_metadata(kaggle_files: list[KaggleFile], raw_dir: str) -> list[FileMetadata]:
    metadata: list[FileMetadata] = []
    for file_info in kaggle_files:
        name = file_info["name"]
        entry: FileMetadata = {"name": name, "size": file_info["size"]}
        local_path = os.path.join(raw_dir, name)
        if name.endswith(".csv") and os.path.exists(local_path):
            entry["columns"] = read_csv_columns(local_path)
        metadata.append(entry)
    return metadata

def download_data(api: "KaggleApi") -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    api.competition_download_files(COMPETITION, path=RAW_DIR, quiet=False)
    zip_path = os.path.join(RAW_DIR, f"{COMPETITION}.zip")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW_DIR)
    os.remove(zip_path)

def write_files_metadata(api: "KaggleApi") -> None:
    kaggle_files: list[KaggleFile] = [{"name": f.name, "size": f.total_bytes}
                     for f in api.competition_list_files(COMPETITION).files]
    metadata = build_files_metadata(kaggle_files, RAW_DIR)
    os.makedirs(META_DIR, exist_ok=True)
    with open(FILES_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

def write_pages(api: "KaggleApi") -> None:
    os.makedirs(PAGES_DIR, exist_ok=True)
    for page in api.competition_list_pages(COMPETITION):
        if not page.content:
            continue
        filename = f"{slugify_page_name(page.name)}.md"
        with open(os.path.join(PAGES_DIR, filename), "w", encoding="utf-8") as f:
            f.write(page.content)

def main() -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    download_data(api)
    write_files_metadata(api)
    write_pages(api)

if __name__ == "__main__":
    main()
