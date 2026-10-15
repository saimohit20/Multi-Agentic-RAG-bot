"""
dc_available/common/sharepoint_connector.py — shared SharePoint / Microsoft Graph client.

One place for everything the SharePoint-facing code needs:
  - config       load pipeline/.env  ->  cfg(name)
  - Graph        app-only token, site/drive resolution, list, download, upload
  - shortcuts    .url  -> Confluence/Docupedia HTML ;  .lnk -> real file (matched by name)
  - classify     topic / level / training from a file name

Imported by the pipeline stages (pipeline/ingest.py, pipeline/sync.py) and by
dc_available/trust_score/collect_metadata.py. Lives in common/ (the shared layer)
so pipeline depends on core — not the other way around.
"""

import base64
import io
import os
import re
import time
from pathlib import Path
from urllib.parse import unquote

import requests

GRAPH = "https://graph.microsoft.com/v1.0"


# ===========================================================================
# config  (pipeline/.env)
# ===========================================================================
def load_env(path: Path = None) -> None:
    # This module lives in dc_available/common/, but the SharePoint/Graph creds
    # still live in pipeline/.env (repo_root/pipeline/.env). parents[2] = repo root.
    path = path or (Path(__file__).resolve().parents[2] / "pipeline" / ".env")
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


load_env()   # load pipeline/.env at import time


def cfg(name, default=""):
    return os.environ.get(name, default).strip()


# ===========================================================================
# Microsoft Graph  (app-only login)
# ===========================================================================
_token = {"value": "", "exp": 0.0}


def token():
    if _token["value"] and time.time() < _token["exp"] - 60:
        return _token["value"]
    resp = requests.post(
        f"https://login.microsoftonline.com/{cfg('GRAPH_TENANT_ID')}/oauth2/v2.0/token",
        data={
            "grant_type": "client_credentials",
            "client_id": cfg("GRAPH_CLIENT_ID"),
            "client_secret": cfg("GRAPH_CLIENT_SECRET"),
            "scope": "https://graph.microsoft.com/.default",
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    _token["value"] = data["access_token"]
    _token["exp"] = time.time() + int(data.get("expires_in", 3600))
    return _token["value"]


def headers():
    return {"Authorization": f"Bearer {token()}"}


def site_id(which):
    """which = 'SOURCE' or 'DEST'. Uses <which>_SITE_ID, else host + <which>_SITE_PATH."""
    sid = cfg(f"{which}_SITE_ID")
    if sid:
        return sid
    host, path = cfg("SHAREPOINT_HOST"), cfg(f"{which}_SITE_PATH").strip("/")
    resp = requests.get(f"{GRAPH}/sites/{host}:/{path}", headers=headers(), timeout=30)
    resp.raise_for_status()
    return resp.json()["id"]


def drive_id(sid):
    resp = requests.get(f"{GRAPH}/sites/{sid}/drive", headers=headers(), timeout=30)
    resp.raise_for_status()
    return resp.json()["id"]


def list_files(did, folder=""):
    """Yield (item, relative_path) for every file under folder (recursive)."""
    folder = folder.strip("/")
    url = (f"{GRAPH}/drives/{did}/root:/{folder}:/children" if folder
           else f"{GRAPH}/drives/{did}/root/children")
    yield from _walk(did, url, "")


def _walk(did, url, prefix):
    while url:
        resp = requests.get(url, headers=headers(), timeout=60)
        resp.raise_for_status()
        data = resp.json()
        for item in data.get("value", []):
            rel = f"{prefix}/{item['name']}" if prefix else item["name"]
            if "folder" in item:
                yield from _walk(did, f"{GRAPH}/drives/{did}/items/{item['id']}/children", rel)
            elif "file" in item:
                yield item, rel
        url = data.get("@odata.nextLink", "")


def download(did, item_id, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(f"{GRAPH}/drives/{did}/items/{item_id}/content",
                      headers=headers(), stream=True, timeout=600) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as fout:
            for chunk in resp.iter_content(1 << 16):
                fout.write(chunk)


def get_bytes(did, item_id):
    resp = requests.get(f"{GRAPH}/drives/{did}/items/{item_id}/content",
                        headers=headers(), timeout=60)
    resp.raise_for_status()
    return resp.content


def upload(did, dest_path, local_file):
    """Upload one file to dest_path (relative to drive root); return its eTag.

    Big files go in chunks. The returned eTag lets the caller seed sync's manifest
    so the just-uploaded file is not re-downloaded later.
    """
    size = local_file.stat().st_size
    base = f"{GRAPH}/drives/{did}/root:/{dest_path}:"
    if size < 4 * 1024 * 1024:                       # small file: single PUT
        with open(local_file, "rb") as fin:
            resp = requests.put(base + "/content", headers=headers(), data=fin, timeout=300)
        resp.raise_for_status()
        return resp.json().get("eTag", "")
    # large file (e.g. .mp4): upload session, 5 MB chunks
    resp = requests.post(base + "/createUploadSession", headers=headers(), timeout=60)
    resp.raise_for_status()
    upload_url, chunk_size, start = resp.json()["uploadUrl"], 5 * 1024 * 1024, 0
    etag = ""
    with open(local_file, "rb") as fin:
        while start < size:
            data = fin.read(chunk_size)
            end = start + len(data) - 1
            r = requests.put(upload_url, timeout=600, data=data,
                             headers={"Content-Length": str(len(data)),
                                      "Content-Range": f"bytes {start}-{end}/{size}"})
            r.raise_for_status()
            if r.status_code in (200, 201):          # final chunk returns the item
                etag = r.json().get("eTag", "")
            start = end + 1
    return etag


# ===========================================================================
# .lnk (Verknüpfung) -> the REAL file in the SAME library  (match by file name)
# ===========================================================================
_name_index = {"did": None, "map": None}


def library_index(did):
    """One-time map  filename.lower() -> item  for the whole library (cached)."""
    if _name_index["did"] == did and _name_index["map"] is not None:
        return _name_index["map"]
    print("      building library file-name index (one time)...")
    index, dups = {}, 0
    for it, _rel in list_files(did, ""):
        key = it["name"].lower()
        if key in index:
            dups += 1
        else:
            index[key] = it
    if dups:
        print(f"      note: {dups} duplicate file name(s) in library — kept first seen")
    _name_index.update(did=did, map=index)
    return index


def lnk_to_item(did, raw_bytes):
    """Parse .lnk bytes; return (item, real_name) for the file it points to.

    item is None if the target file name isn't found in the library.
    Raises ImportError if pylnk3 is missing.
    """
    import pylnk3
    lnk = pylnk3.parse(io.BytesIO(raw_bytes))
    target = (getattr(lnk, "path", "") or "").strip()
    real_name = target.replace("/", "\\").split("\\")[-1] if target else ""
    if not real_name:
        return None, ""
    return library_index(did).get(real_name.lower()), real_name


# ===========================================================================
# .url shortcut  ->  Confluence/Docupedia page HTML  (PAT + REST only, no browser)
# ===========================================================================
def _clean_link(link):
    second_http = link.find("http", 1)          # next 'http' after the start
    return (link[:second_http] if second_http != -1 else link).strip()


def _decode_tiny(code):
    s = code.replace("-", "+").replace("_", "/")
    s += "=" * (-len(s) % 4)
    return str(int.from_bytes(base64.b64decode(s), "little"))


def _page_id_from_link(link):
    m = re.search(r"/pages/(\d+)", link) or re.search(r"[?&]pageId=(\d+)", link)
    if m:
        return m.group(1)
    m = re.search(r"/x/([A-Za-z0-9_-]+)", link)
    if m:
        return _decode_tiny(m.group(1))
    m = re.search(r"/display/([^/]+)/([^?#]+)", link)
    if m:
        space = unquote(m.group(1))
        title = unquote(m.group(2).replace("+", " "))
        resp = requests.get(
            f"{cfg('DOCUPEDIA_BASE_URL').rstrip('/')}/rest/api/content",
            params={"spaceKey": space, "title": title},
            headers={"Authorization": f"Bearer {cfg('DOCUPEDIA_PAT')}"},
            timeout=30,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            raise RuntimeError(f"no page for space={space} title={title!r}")
        return results[0]["id"]
    raise RuntimeError(f"unrecognized Confluence link: {link}")


def url_to_html(raw_bytes, dest, debug_name=""):
    text = raw_bytes.decode("utf-8", errors="ignore")
    link = next((ln[4:].strip() for ln in text.splitlines()
                 if ln.strip().upper().startswith("URL=")), "")
    if not link:
        raise RuntimeError("no URL inside .url file")
    link = _clean_link(link)
    page_id = _page_id_from_link(link)
    api_url = (f"{cfg('DOCUPEDIA_BASE_URL').rstrip('/')}"
               f"/rest/api/content/{page_id}?expand=body.export_view")
    resp = requests.get(api_url, headers={"Authorization": f"Bearer {cfg('DOCUPEDIA_PAT')}"},
                        timeout=120)
    resp.raise_for_status()
    data = resp.json()
    title = data.get("title", "")
    body = data.get("body", {}).get("export_view", {}).get("value", "")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title></head><body>{body}</body></html>",
        encoding="utf-8",
    )


# ===========================================================================
# classify — topic / level / training from a file name
# ===========================================================================
BASIC = ["overview", "basics", "basic", "introduction", "quick reference",
         "general elements", "level 2", "process overview", "fundamentals", "quickreference"]
EXPERT = ["l4", "level 4", "e2e", "end to end", "end-to-end", "key user",
          "rollout consultant", "process control", "reverse processing", "tax elements",
          "vertex", "credit limit", "availability check", "apm number", "contract", "lls"]


def topic_of(name):
    """Topic = the 3-char code before the first '-'  ('O2C-03.036 ...' -> 'O2C')."""
    before = name.split("-", 1)[0].strip()
    return before.upper() if len(before) == 3 else "Other"


def level_of(name):
    low = name.lower()
    if any(k in low for k in EXPERT):
        return "Expert"
    if any(k in low for k in BASIC):
        return "Basic"
    if "level 3" in low:
        return "Advanced"
    return "Advanced"   # default


def training_of(name):
    """A single fixed 'training' folder per Topic/Level."""
    return "training"
