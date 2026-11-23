"""Build the submission archive from the mirrored submission tree.

The delivered archive is built from ``submission/`` (not from the authoring
root), and it excludes LaTeX build artefacts.  Those artefacts exist in the
working tree because every compile regenerates them; shipping them is a
hygiene defect, not a convenience.

Usage:
    python code/scripts/make_submission_zip.py
"""
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "submission")

# LaTeX build artefacts and editor/OS debris. Never ship these.
EXCLUDE_EXT = {".aux", ".log", ".bbl", ".blg", ".out", ".abs", ".toc",
               ".lof", ".lot", ".synctex", ".fls", ".fdb_latexmk",
               ".pyc", ".pyo", ".DS_Store", ".bak", ".orig"}
# `thumbnails/` ships deliberately: cas-dc references the social-icon set,
# and dropping it on a clean checkout is exactly the kind of "it compiled
# here" breakage a reviewer should never have to debug.
EXCLUDE_DIR = {"__pycache__", ".ipynb_checkpoints"}
# Files that are useful in the working tree but are not part of the
# submission (preview dumps, driver logs, queue bookkeeping).
EXCLUDE_NAME = set()


def build():
    out = os.path.join(ROOT, "SpikeRPGNN_submission.zip")
    entries = []
    for dirpath, dirnames, filenames in os.walk(SRC):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIR]
        for fn in sorted(filenames):
            ext = os.path.splitext(fn)[1].lower()
            if ext in EXCLUDE_EXT or fn in EXCLUDE_NAME:
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, SRC).replace(os.sep, "/")
            entries.append((full, rel))
    entries.sort(key=lambda x: x[1])
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for full, rel in entries:
            z.write(full, rel)
    size = os.path.getsize(out)
    arte = [r for _, r in entries
            if os.path.splitext(r)[1].lower() in
            {".aux", ".log", ".bbl", ".blg", ".out", ".abs"}]
    print(f"wrote {out}")
    print(f"  entries : {len(entries)}")
    print(f"  size    : {size / 1048576:.2f} MB")
    print(f"  artefacts leaked: {len(arte)}")
    if arte:
        for a in arte[:10]:
            print(f"    !! {a}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(build())
