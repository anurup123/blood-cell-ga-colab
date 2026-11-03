"""Verified downloads for the self-contained blood-cell notebook."""
import hashlib
import json
from pathlib import Path
import shutil
import stat
import time
import urllib.request
import zipfile


def sha256_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def download(url, destination, expected_sha256, expected_size=None):
    """Reuse correct files; resume partial downloads; reject incorrect bytes."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and sha256_file(destination) == expected_sha256:
        print("Ready:", destination.name)
        return destination
    partial = destination.with_name(destination.name + ".download")
    for attempt in range(5):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if expected_size is not None and offset >= expected_size:
                if offset == expected_size and sha256_file(partial) == expected_sha256:
                    partial.replace(destination)
                    return destination
                partial.unlink()
                offset = 0
            headers = {"User-Agent": "Blood-Cell-GA/1.0", "Accept-Encoding": "identity"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(url, headers=headers)
            print(f"Downloading {destination.name}" + (f" (resuming at {offset / 1e6:.1f} MB)" if offset else ""))
            with urllib.request.urlopen(request, timeout=120) as response:
                append = offset > 0 and response.status == 206
                if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise IOError("Download server returned the wrong byte range")
                with partial.open("ab" if append else "wb") as stream:
                    shutil.copyfileobj(response, stream, length=8 * 1024 * 1024)
            if expected_size is not None and partial.stat().st_size != expected_size:
                raise IOError("Download incomplete; retrying")
            if sha256_file(partial) != expected_sha256:
                partial.unlink()
                raise IOError("Downloaded checksum mismatch; retrying")
            partial.replace(destination)
            print("Verified:", destination.name)
            return destination
        except Exception:
            if attempt == 4:
                raise
            time.sleep(min(2 ** attempt, 8))


def safe_extract(archive, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            target = (destination / item.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("Unsafe archive path: " + item.filename)
            if stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError("Unexpected archive symlink: " + item.filename)
        z.extractall(destination)


def release_url(manifest, name):
    return f"https://github.com/{manifest['repository']}/releases/download/{manifest['release']}/{name}"


def fetch_bundle(manifest, work_dir):
    work_dir = Path(work_dir)
    info = manifest["model_bundle"]
    archive = download(release_url(manifest, info["name"]), work_dir / "downloads" / info["name"], info["sha256"], info["size"])
    directory = work_dir / "model_bundle"
    marker = directory / ".archive_sha256"
    if not marker.exists() or marker.read_text().strip() != info["sha256"]:
        safe_extract(archive, directory)
        marker.write_text(info["sha256"] + "\n")
    checksums = json.loads((directory / "checksums.json").read_text())
    for relative, expected in checksums.items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file() or sha256_file(path) != expected:
            raise ValueError("Model bundle file is missing or changed: " + relative)
    print("Model, class labels, feature mask, and example images are ready.")
    return directory
