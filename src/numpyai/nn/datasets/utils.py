"""Utility functions for loading and caching datasets."""

import os
from typing import Optional
from urllib.request import urlretrieve
import hashlib
from _hashlib import HASH
from numpyai.backend import CACHE_DIR

DATASET_DIR = CACHE_DIR / 'datasets'
"""Location to cache downloaded datasets."""

def hash_file(fpath: str, hasher: HASH, chunk_size: int = 65535) -> str:
    """Calculates a file's sha256 or md5 hash.

    Parameters
    ----------
    fpath : str
        Path to the file being validated.
    algorithm : str, optional
        Hash algorithm to use, by default 'sha256'
    chunk_size : int, optional
        Bytes to read at a time, by default 65535

    Returns
    -------
    str
        The file hash.
    """
    with open(fpath, "rb") as fpath_file:
        for chunk in iter(lambda: fpath_file.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def resolve_hasher(algorithm, file_hash=None):
    """Returns hash algorithm as hashlib function."""
    if algorithm == "sha256":
        return hashlib.sha256()
    if algorithm == "auto" and file_hash is not None and len(file_hash) == 64:
        return hashlib.sha256()
    return hashlib.md5()

def validate_file(
    fpath: str, 
    file_hash: str, 
    algorithm: str = 'auto', 
    chunk_size: int = 65535
) -> bool:
    """Validates a file against a sha256 or md5 hash.

    Parameters
    ----------
    fpath : str
        Path to the file being validated.
    file_hash : str
        Expected hash string of the file.
    algorithm : str, optional
        Hash algorithm to use, by default 'auto'
    chunk_size : int, optional
        Bytes to read at a time, by default 65535

    Returns
    -------
    bool
        Whether the file is valid.
    """
    hasher = resolve_hasher(algorithm, file_hash)
    if str(hash_file(fpath, hasher, chunk_size)) == str(file_hash):
        return True
    else:
        return False

def get_file(
    fname: str,
    origin: str,
    file_hash: Optional[str] = None,
    hash_algorithm: str = 'auto',
    force_download: bool = False
) -> str:
    """Downloads a file from a URL if it's not already in the cache.

    By default the file at the url `origin` is downloaded to the
    directory `./.pyai/datasets` and given the filename `fname`.

    Passing a hash will verify the file after download.

    Parameters
    ----------
    fname : str
        Desired local name for the file.
    origin : str
        Original URL of the file.
    file_hash : Optional[str], optional
        Expected hash string of the file after download, by default None
    hash_algorithm : str, optional
        Hash algorithm to verify the file. Options are
        'md5', 'sha256', or 'auto', by default 'auto'
    force_download : bool, optional
        Whether to force the file to be redownloaded regardless
        of cache state, by default False

    Returns
    -------
    str
        Path to the downloaded file.
    """
    # Makes cache directory and checks if download is required
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    download_target = os.path.join(DATASET_DIR, fname)
    download = force_download or not os.path.exists(download_target)

    # Attempts to retrieve a file from the origin URL and save it
    if download:
        try:
            urlretrieve(origin, download_target)
        except (Exception, KeyboardInterrupt):
            if os.path.exists(download_target):
                os.remove(download_target)
            raise

    # If the download was successful and a file hash was provided, verify the download
    if os.path.exists(download_target) and file_hash is not None:
        if not validate_file(
            download_target, file_hash, algorithm=hash_algorithm
        ):
            raise ValueError(
                "Incomplete or corrupted file detected. "
                f"The {hash_algorithm} "
                "file hash does not match the provided value "
                f"of {file_hash}."
            )

    return download_target