from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import os
import tempfile
import time

DEFAULT_QUERY = (
    "mimeType!='application/vnd.google-apps.folder' and "
    "not mimeType contains 'image/' and "
    "not mimeType contains 'video/' and "
    "not mimeType contains 'audio/' and "
    "not mimeType contains 'application/zip' and "
    "not mimeType contains 'application/x-rar' and "
    "not mimeType contains 'application/x-7z' and "
    "trashed=false"
)

GOOGLE_EXPORT_TYPES = {
    'application/vnd.google-apps.document': ('text/plain', '.txt'),
    'application/vnd.google-apps.spreadsheet': ('text/csv', '.csv'),
    'application/vnd.google-apps.presentation': ('text/plain', '.txt'),
}

def _build_service(access_token: str):
    creds = Credentials(token=access_token)
    return build('drive', 'v3', credentials=creds, cache_discovery=False)

def _list_files(service, page_size: int = 100, q: str = None):
    q_use = q if q is not None else DEFAULT_QUERY
    resp = service.files().list(q=q_use, pageSize=page_size, fields="files(id, name, mimeType)").execute()
    return resp.get("files", [])

def _download(service, file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fid = file.get("id")
    name = file.get("name", fid)
    mime = file.get("mimeType", "")
    safe = "".join(c for c in name if c.isalnum() or c in (' ','.','_')).rstrip()
    if mime in GOOGLE_EXPORT_TYPES:
        export_mime, ext = GOOGLE_EXPORT_TYPES[mime]
        req = service.files().export_media(fileId=fid, mimeType=export_mime)
        out = os.path.join(output_dir, safe + ext)
    else:
        _, ext = os.path.splitext(name)
        out = os.path.join(output_dir, safe + ext)
        req = service.files().get_media(fileId=fid)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, req)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    with open(out, "wb") as f:
        f.write(fh.getvalue())
    return out

def _path_to_text(path):
    with open(path, "rb") as f:
        raw = f.read()
    text = raw.decode("utf-8", errors="ignore").strip()
    return text if text else None

def fetch_and_extract(access_token: str, max_files: int = 100, q: str = None, output_dir: str = None):
    """
    Minimal helper. Returns (output_dir, docs)
    docs is a list of dicts: { "id": ..., "text": ..., "metadata": {"name":..., "mimeType":...}, "path": ... }
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix=f"drive_fetch_{int(time.time())}_", dir="/tmp")
    svc = _build_service(access_token)
    files = _list_files(svc, page_size=max_files, q=q)
    docs = []
    for f in files:
        fid = f.get("id")
        path = _download(svc, f, output_dir)
        text = _path_to_text(path)
        if not text:
            # keep the path even if text is empty; we will run OCR/cleaning on path in main
            text = ""
        docs.append({
            "id": fid,
            "text": text,
            "metadata": {"name": f.get("name"), "mimeType": f.get("mimeType")},
            "path": path
        })
    return output_dir, docs