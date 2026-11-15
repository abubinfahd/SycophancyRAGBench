"""
Creates a clean, distributable ZIP archive of the sycophancy_rag_bench codebase,
ready to upload directly to Google Colab or Kaggle.
"""

import os
import sys
import zipfile

pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
parent_dir = os.path.dirname(pkg_dir)
output_zip = os.path.join(parent_dir, "sycophancy_rag_bench.zip")

EXCLUDE_DIRS = {"__pycache__", ".ipynb_checkpoints", ".git", "artifacts", "index"}
EXCLUDE_EXTS = {".pyc", ".tmp", ".lock"}

def make_archive():
    print(f"[Packaging] Compressing {pkg_dir} into {output_zip} ...")
    count = 0
    total_bytes = 0

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(pkg_dir):
            # Prune excluded directories
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

            for file in files:
                if any(file.endswith(ext) for ext in EXCLUDE_EXTS):
                    continue
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, pkg_dir)
                zf.write(full_path, arcname=os.path.join("sycophancy_rag_bench", rel_path))
                count += 1
                total_bytes += os.path.getsize(full_path)

    zip_mb = os.path.getsize(output_zip) / (1024 * 1024)
    print(f"[Packaging] Successfully created: {output_zip}")
    print(f"[Packaging] Contained {count} files ({total_bytes / (1024*1024):.2f} MB uncompressed) -> Archive Size: {zip_mb:.2f} MB")
    return output_zip

if __name__ == "__main__":
    make_archive()
