#!/usr/bin/env python3
"""
Download the micro-sam (Segment Anything for Microscopy) weights that get baked into the
micro-sam-inference image. Run this once before `docker compose build micro-sam-inference`.

    python scripts/fetch_microsam_weights.py            # ViT-B models (default, MICROSAM_SIZE=b)
    python scripts/fetch_microsam_weights.py --size l   # ViT-L models

Files land in services/microsam_inference/weights/ (git-ignored) under the names micro_sam's
pooch cache expects, so the container never downloads anything at runtime.
URLs and xxh128 hashes come from micro_sam.util.models() (micro_sam 1.8).
"""
import argparse
import os
import sys
import time
import urllib.request

BASE = "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io"
# bioimage.io (model id, version) per ViT size, as registered in micro_sam.util.models() (micro_sam 1.8):
# light microscopy (lm) and electron microscopy organelles (em). There is no ViT-H generalist pair.
IDS = {
    "t": {"lm": ("faithful-chicken", "1.1"), "em": ("greedy-whale", "1")},
    "b": {"lm": ("diplomatic-bug", "1.2"), "em": ("noisy-ox", "1.2")},
    "l": {"lm": ("idealistic-rat", "1.2"), "em": ("humorous-crab", "1.2")},
}
# known hashes (only the default size is pinned; other sizes are verified by micro_sam at load time)
XXH128 = {
    "vit_b_lm": "fe9252a29f3f4ea53c15a06de471e186",
    "vit_b_lm_decoder": "708b15ac620e235f90bb38612c4929ba",
    "vit_b_em_organelles": "f3bf2ed83d691456bae2c3f9a05fb438",
    "vit_b_em_organelles_decoder": "bb6398956a6b0132c26b631c14f95ce2",
}

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, "..", "services", "microsam_inference", "weights")


def targets(size):
    (lm_id, lm_v), (em_id, em_v) = IDS[size]["lm"], IDS[size]["em"]
    return {
        f"vit_{size}_lm": f"{BASE}/{lm_id}/{lm_v}/files/vit_{size}.pt",
        f"vit_{size}_lm_decoder": f"{BASE}/{lm_id}/{lm_v}/files/vit_{size}_decoder.pt",
        f"vit_{size}_em_organelles": f"{BASE}/{em_id}/{em_v}/files/vit_{size}.pt",
        f"vit_{size}_em_organelles_decoder": f"{BASE}/{em_id}/{em_v}/files/vit_{size}_decoder.pt",
    }


def xxh128(path):
    try:
        import xxhash
    except ImportError:
        return None
    h = xxhash.xxh128()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, path):
    """Resumable download with a progress line."""
    tmp = path + ".part"
    have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        total = have + int(r.headers.get("Content-Length", 0))
        mode = "ab" if have and r.status == 206 else "wb"
        if mode == "wb":
            have = 0
        t0 = time.time()
        with open(tmp, mode) as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                have += len(chunk)
                rate = have / max(time.time() - t0, 1e-6) / 1024
                sys.stdout.write(f"\r  {os.path.basename(path)}: {have/1e6:6.1f}/{total/1e6:.1f} MB ({rate:,.0f} kB/s)")
                sys.stdout.flush()
    print()
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default=os.getenv("MICROSAM_SIZE", "b"), choices=list(IDS))
    args = ap.parse_args()
    os.makedirs(DEST, exist_ok=True)
    for name, url in targets(args.size).items():
        path = os.path.join(DEST, name)
        if os.path.exists(path):
            digest = xxh128(path)
            if name in XXH128 and digest and digest != XXH128[name]:
                print(f"  {name}: hash mismatch, re-downloading")
                os.remove(path)
            else:
                print(f"  {name}: present ({os.path.getsize(path)/1e6:.1f} MB)")
                continue
        print(f"  {name}: downloading {url}")
        download(url, path)
        digest = xxh128(path)
        if name in XXH128 and digest and digest != XXH128[name]:
            sys.exit(f"ERROR: {name} hash mismatch after download ({digest})")
    print(f"weights ready in {os.path.normpath(DEST)}")


if __name__ == "__main__":
    main()
