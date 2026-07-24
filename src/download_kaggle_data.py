import csv, json, os, re, zipfile

COMPETITION = "museumscat-specimen-collection-annotation-task"

RAW_DIR = os.path.join("data", "raw")
META_DIR = "kaggle_meta"
PAGES_DIR = os.path.join(META_DIR, "pages")
FILES_METADATA_PATH = os.path.join(META_DIR, "files_metadata.json")

def slugify_page_name(name):
    slug = name.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")

def read_csv_columns(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        return next(csv.reader(f))

def build_files_metadata(kaggle_files, raw_dir):
    metadata = []
    for file_info in kaggle_files:
        name = file_info["name"]
        entry = {"name": name, "size": file_info["size"]}
        local_path = os.path.join(raw_dir, name)
        if name.endswith(".csv") and os.path.exists(local_path):
            entry["columns"] = read_csv_columns(local_path)
        metadata.append(entry)
    return metadata

def download_data(api):
    os.makedirs(RAW_DIR, exist_ok=True)
    api.competition_download_files(COMPETITION, path=RAW_DIR, quiet=False)
    zip_path = os.path.join(RAW_DIR, f"{COMPETITION}.zip")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW_DIR)
    os.remove(zip_path)

def write_files_metadata(api):
    kaggle_files = [{"name": f.name, "size": f.total_bytes}
                     for f in api.competition_list_files(COMPETITION).files]
    metadata = build_files_metadata(kaggle_files, RAW_DIR)
    os.makedirs(META_DIR, exist_ok=True)
    with open(FILES_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

def write_pages(api):
    os.makedirs(PAGES_DIR, exist_ok=True)
    for page in api.competition_list_pages(COMPETITION):
        if not page.content:
            continue
        filename = f"{slugify_page_name(page.name)}.md"
        with open(os.path.join(PAGES_DIR, filename), "w", encoding="utf-8") as f:
            f.write(page.content)

def main():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    download_data(api)
    write_files_metadata(api)
    write_pages(api)

if __name__ == "__main__":
    main()
