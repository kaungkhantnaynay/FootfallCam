"""Extract the three tag patches supplied in the employer's assessment.

Crop rectangles refer to PDF reference images only, never the test video's frames.
"""
from pathlib import Path
from pypdf import PdfReader
import json
import hashlib

source = Path("data/reference/assessment.pdf")
out = source.parent
images = [entry.image for entry in PdfReader(source).pages[0].images]
specs = [(0, (112, 74, 151, 101), "tag-detail"),
         (1, (87, 98, 111, 112), "tag-dark-example"),
         (2, (48, 110, 73, 122), "tag-light-example")]
manifest = {"source": source.name, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "purpose": "Employer-provided tag references, not training labels", "patches": []}
for index, box, name in specs:
    path = out / f"{name}.png"
    images[index].crop(box).save(path)
    manifest["patches"].append({"path": path.name, "pdf_image_index": index, "crop_xyxy": box})
(out / "reference_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("Saved three employer-provided tag reference crops and provenance manifest.")
