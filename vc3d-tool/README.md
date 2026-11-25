# The blend as a VC3D (villa) tool

`growpatch_blend/` is the tiled blend packaged the way villa's `segmentation/` tools are, ready to propose to
[ScrollPrize/villa](https://github.com/ScrollPrize/villa) as a draft pull request. `growpatch_blend.patch` adds it
there as seven new files; it changes nothing that already exists in villa.

Inside villa it runs as one command on a VC3D segment and writes a new tifxyz segment with the same files GrowPatch writes:

```sh
python -m segmentation.growpatch_blend /path/to/scroll.volpkg/paths/<segment> <prediction.zarr>/0 --workers 8
```

How it was checked:

- The core here, run on the PHerc. 125 example in this repository, gives exactly the same output as
  `growpatch-blend/scripts/growbench_tiled_blend.py`: every x, y and z value of the 817 × 805 grid matches.
- Two unit tests on a synthetic flat sheet pass (a point 3 voxels off moves halfway; a point 12 voxels off stays).
- Reading and writing segments and Zarr arrays goes through villa's `vc_reader`, as villa's AGENTS.md and the
  vc_reader README ask. vc_reader needs a native build, so that wrapper has not been run here yet, and the output
  has not yet been opened in VC3D.

The tool's [README](growpatch_blend/README.md) explains what it does and its limits in plain terms.
