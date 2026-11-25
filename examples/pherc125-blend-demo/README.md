# Blend demo on a real PHerc. 125 GrowPatch mesh

The input is an unmodified official GrowPatch run: Villa `e875441`, `vc_grow_seg_from_seed`, step 8, 160 generations. It was seeded inside PHerc. 125 at native XYZ (3880, 3317, 10467) on volume `20250821151825`. `stock-growpatch/` holds its `tifxyz` output and parameters. Its coordinates are relative to the prediction cube origin (3584, 3008, 10176), hence `--stock-local`.

```sh
# from the repository root
python examples/pherc125-blend-demo/fetch_m7_cube.py --out ../blend-work/m7.npy
cd examples/pherc125-blend-demo
python ../../growpatch-blend/scripts/growbench_refine_mesh.py stock-growpatch ../../../blend-work/m7.npy site.json ../../../blend-work/out --stock-local
```

The fetch script downloads a 576 × 768 × 1408 crop of the public surface-m7 prediction (about 620 MB as uint8, ZYX). Its source URL and bounds are in `m7_cube.json`, and the script checks the SHA-256. The blend itself takes a few seconds on CPU. It writes:

- `local-blend.tifxyz`: a 97 × 97 chart (2-voxel spacing, 96-voxel radius) in global XYZ, editable in VC3D.
- `graph-confidence-cut.obj`: the independent graph trace, clipped at its confidence cutoff.
- `run.json`: the seed, the number of blended nodes and the graph receipt.

The x/y/z TIFFs should match `expected-output/` byte for byte. There is no sheet-ID annotation on PHerc. 125, so this example shows the tool running on real data. It does not measure an improvement. The measured comparison is the 16-region Paris. 4 benchmark described in the top-level README.

## Whole mesh

The tiled version refines the entire GrowPatch mesh with the locked settings (32-voxel tile lattice, shared-only combination; `growpatch-blend/scripts/tiled_blend_lock.json`):

```sh
# from the repository root, after the fetch step above
python growpatch-blend/scripts/growbench_tiled_blend.py examples/pherc125-blend-demo/stock-growpatch ../blend-work/m7.npy ../blend-work/whole --origin 3584 3008 10176 --stock-local
```

It writes:

- `tiled-blend.tifxyz`: the whole mesh, resampled from GrowPatch's 8-voxel grid to 2 voxels (`scale` 0.5, global XYZ).
- `displacement_xyz.npy`: the per-vertex move.
- `report.json`: tile counts and displacement statistics.

On 20 CPU cores this mesh takes about two minutes. There are 2,026 tiles; 794 solve and 1,232 abstain, mostly because 59% of this GrowPatch mesh was extrapolated outside the prediction cube it was grown on. The x/y/z TIFFs should match `whole-mesh-output/` byte for byte. `--workers` sets the number of processes.
