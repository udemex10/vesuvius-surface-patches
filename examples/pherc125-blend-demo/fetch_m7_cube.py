"""Download the PHerc. 125 surface-m7 prediction cube used by the blend demo and check its hash.

    python examples/pherc125-blend-demo/fetch_m7_cube.py --out ../blend-demo-work/m7.npy

The cube bounds, source URL and expected SHA-256 are read from m7_cube.json next to this file.
Absent chunks in the sparse prediction take the array's fill value, as the Zarr spec defines.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "reconstruction" / "extensions" / "scripts"))
from zarr_crop import read_crop  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True, help="Destination .npy (ZYX uint8)")
    p.add_argument("--cache", type=Path, default=None, help="Chunk cache directory (default: next to --out)")
    args = p.parse_args()
    spec = json.loads((HERE / "m7_cube.json").read_text())
    (x0, y0, z0), (x1, y1, z1) = spec["global_xyz_half_open"]
    cache = args.cache or args.out.parent / "m7_chunk_cache"
    cube = read_crop(spec["url"], [z0, y0, x0, z1, y1, x1], cache, max_bytes=4 * 1024**3, missing_fill=True)
    if isinstance(cube, tuple):
        cube = cube[0]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, np.ascontiguousarray(cube))
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    ok = digest == spec["sha256_npy"]
    print(f"{args.out} shape={cube.shape} sha256={digest} {'matches' if ok else 'DIFFERS FROM'} the recorded cube")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
