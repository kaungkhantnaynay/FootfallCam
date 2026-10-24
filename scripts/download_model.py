"""Explicit one-time network step; inference itself only loads local weights."""
from pathlib import Path
import urllib.request
import hashlib

url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
target = Path("models/yolo11n.pt")
target.parent.mkdir(exist_ok=True)
if not target.exists():
    temporary = target.with_suffix(".download")
    urllib.request.urlretrieve(url, temporary)
    temporary.replace(target)
digest = hashlib.sha256(target.read_bytes()).hexdigest()
expected = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"
if digest != expected:
    raise RuntimeError(f"Model hash mismatch: expected {expected}, got {digest}")
print(f"{target}: {target.stat().st_size:,} bytes; sha256={digest}")
print("Source: " + url)
print("Review Ultralytics AGPL-3.0 / enterprise terms before distribution or commercial use.")
