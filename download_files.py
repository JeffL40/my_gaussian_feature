import os
from pathlib import Path
from urllib.parse import urlparse, urljoin
import requests
from bs4 import BeautifulSoup
import argparse
from typing import List
import functools
import pdb


def get_links(content):
    soup = BeautifulSoup(content)
    for a in soup.findAll("a"):
        yield a.get("href")


def declare_dir(dir_):
    os.makedirs(dir_)
    return os.path.abspath(dir_)


def download(base_url: str, additional: List[str], output_path: str):
    url = base_url
    for addl in additional:
        url += addl
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        raise Exception("status code is {} for {}".format(r.status_code, url))
    path = os.path.join(output_path, *additional)
    if url.endswith("/"):
        content = r.text
        for link in get_links(content):
            if not link.startswith("."):  # skip hidden files such as .DS_Store
                download(base_url, additional + [link], output_path)
    else:
        print(f"saving {url} at path {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=10 * 1024):
                f.write(chunk)


if __name__ == "__main__":
    # the trailing / indicates a folder
    parser = argparse.ArgumentParser()
    parser.add_argument("--download_url", type=str)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()
    download(
        args.download_url + ("" if args.download_url.endswith("/") else "/"),
        [],
        args.output_path,
    )
