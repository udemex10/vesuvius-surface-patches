# Ten interior surface meshes for PHerc. 125 and PHerc. 268

This release supplies **4.258803 cm²** of triangulated interior surfaces in the original CT coordinate systems. Three starting surfaces contribute 2.910842 cm². Seven PHerc. 125 extensions contribute a further 1.347961 cm² from disjoint outward footprints. Some meshes contain disconnected islands. The 0.813130 cm² subset with corroborating crossing-curve controls is already included in that total. The exact papyrus winding and writing face still require independent review. [Native-coordinate mesh index](geometry.csv) · [Geometry preview](geometry-preview.png) · [Source credits](source-credits.md)

The PHerc. 125 starting patches have seed XYZ coordinates (3400, 3310, 10467) and (3400, 3706, 10467) on volume 20250821151825. The PHerc. 268 starting patch uses (3800, 2991, 7405) on volume 20251110183117. Orthogonal source contours were followed separately, and agreement limited the retained footprint. The extensions grow from the PHerc. 125 Y3310 starter in four directions using overlapping routes on the M7 prediction. Flattening yields a regular chart while each vertex retains its original physical location. The area calculation sums two three-dimensional triangles for every quad with four valid coordinates. [All cases, original source URLs and array digests](index.json)

## Inspect a mesh

Open any indexed `.tifxyz` directory in a compatible surface viewer. Its `x.tif`, `y.tif` and `z.tif` files are float32 coordinate maps over the same intrinsic grid. The scan coordinates are global level-zero voxel centers; multiply by the acquisition pitch to obtain micrometres. An invalid knot has coordinates −1. One chart interval covers eight nominal native voxels. To use a quad, require four valid corners. Mesh links and individual areas appear in the CSV index. [Geometry index](geometry.csv)

```python
import json
from pathlib import Path
import numpy as np
import tifffile

case = json.loads(Path("index.json").read_text())["cases"][0]
mesh = Path(case["mesh"])
xyz = np.stack([tifffile.imread(mesh / f"{axis}.tif") for axis in "xyz"], axis=-1)
valid = np.isfinite(xyz).all(axis=-1) & (xyz >= 0).all(axis=-1)
print(case["id"], xyz[valid][0], case["pitch_um"])
```

## Reproduce geometry and CT imagery

Use Python 3.12 with the pinned dependencies. Work from the extracted release directory, with an empty work directory outside it. Verification checks all outer payload digests and recomputes each area from the TIFF coordinates.

```sh
python -m pip install -r requirements.txt
python review.py verify
```

Regenerate a native center-plane preview directly from the official scan using the supplied mesh. This also writes the losslessly sampled 17-plane CT stack and mask, then compares both array hashes with the archived measurements. The stack uses depth, row, column order. Depth 8 is the central surface. Plane spacing equals the scan pitch along the local mesh normal. The preview uses the first and 99th percentiles on supported center-plane pixels. Replace the case ID with any of the ten IDs in the CSV index.

```sh
python review.py render PHerc0125_west_2 --work ../fresh_render
```

Rebuild one original patch or a specified extension from public contour files, native CT and, for extensions, the public M7 volume. The extension command first reconstructs the old Y3310 starter from its original sources. Each wanted direction also builds all the neighboring components needed to rank its islands. A rebuilt extension must match the distributed x/y/z bytes and expected CT and mask digests. The final command builds all ten. These CPU jobs require network access and sufficient local working space for cached source chunks, routes and CT stacks.

```sh
python review.py rebuild PHerc0268_y2991 --work ../fresh_268
python review.py rebuild PHerc0125_east --work ../fresh_east
python review.py rebuild all --work ../fresh_all
```

Each geometry family retains its original build implementation. The source manifests identify the scroll, acquisition, URLs and native CT range digests. Regenerating imagery downloads upstream data into your own work directory. The outer release contains the derived XYZ coordinates, numerical records, figure code and area measurements.

The clean public-source imagery test reproduced all ten native stacks and masks, comprising 5,013,376 supported center-plane pixels. A separate run regenerated every corroborated face mask and matched its serialized digest. Starting with an empty cache, the directional geometry test rebuilt the Y3310 starter and both west components, then matched their three TIFF coordinate files, CT stacks and masks byte for byte. It exercised the released command for `PHerc0125_west_2`. [Reproduction record](source-replay.json)

## Run a published ink detector on regenerated CT

A frozen four-scroll detector accepted 111 complete windows on four PHerc. 125 extensions. Each depth direction yielded 626,688 scored pixels covering 0.5493 cm². Three narrower components produced no eligible windows. Replaying both directions recovered 1,253,376 identical finite output values on the recorded GPU. This demonstrates a downstream use of the surfaces. Reading letters requires separate evidence. [Detector replay measurements](ink-reader/reference_replay.json)

The example used Python 3.12, NumPy 2.5.4, PyTorch 2.11.0 with CUDA 12.8, and an RTX 5070 Ti Laptop GPU. Download the pinned seed-42 weight file directly from its publisher and save it as `../four_scrolls_scan_transfer_seed42.pth`. The runner verifies the checkpoint SHA-256 before loading weights, regenerates forward and reverse score maps from the native planes, and compares their hashes with the reference run. Different GPU software can affect floating-point agreement. Run this example for the east case; to run all seven, first render every extension into the same work directory and omit `--case east`. [TequilaYuen model card](https://huggingface.co/TequilaYuen/vesuvius-ink-9um-cross-scroll/blob/39eab1b26c7484c2975698b03d881049f9520af6/README.md) · [Pinned seed-42 checkpoint](https://huggingface.co/TequilaYuen/vesuvius-ink-9um-cross-scroll/resolve/39eab1b26c7484c2975698b03d881049f9520af6/production/four_scrolls_scan_transfer_seed42.pth)

```sh
python review.py render PHerc0125_east --work ../reader_ct
python ink-reader/run.py --renders ../reader_ct/renders --case east --checkpoint ../four_scrolls_scan_transfer_seed42.pth --output ../reader_output
```

## Inspect the corroborated subset and benchmark

The seven Boolean extension masks select native mesh quads whose tested corners follow the organizer curves across the old region’s boundary. They qualify **0.813130 cm²**, or 60.3% of the extension area. On the east side, 0.179645 cm² also meets the common-input GrowPatch agreement control. Both checks use the organizer M7. They do not independently label a winding. [Mask definitions and usage](curve-faces/README.md) · [Per-case face controls](curve-faces/bracketed_geometry.json)

The separate benchmark compares methods on older PHerc. Paris. 4 tissue with manually named sheets. On 16 held-out sites the fixed eight-voxel blend recovered 13.95% more seed-connected correct area than its threshold-128 GrowPatch comparator. Strict error fell from 26.12% to 18.09%. The wrong-sheet fraction among named assignments fell by 0.064 percentage points; over the full accepted footprint it rose by 0.053 points. These measurements describe a different scroll and do not adjudicate the ten exported surfaces. The companion archive supplies external source fetchers, frozen numerical outputs, code and replay instructions. [Benchmark companion](benchmark/growpatch-comparison-source.zip) · [Independent uncertainty audit](benchmark/external_resampling.md) · [All sixteen sites](benchmark/external_blend_changes.png)
