# Stock GrowPatch baseline

This experiment ran the organizers’ released Windows x64 binaries for both normal-grid extraction and surface growth. Each invocation used an untouched predicted-foreground cube in ZYX order. The official CLI produced the native `tifxyz` and generation channels. The evaluation copy stores the same coordinates in the shared global XYZ frame, with invalid, outside-cube and ungenerated vertices excluded.

## Version and inputs

The tested release was `VC3D-e875441-2026-11-24-win64`. The SHA-256 of the portable ZIP is `dfdebbdf480f97da455ae821d2bafa8c4402d30006a37aff60a22237bc53f1ea`. The input helper verifies every voxel of the locally encoded Zarr against the shared source array. The official generator runs all three slice directions at each slice and retains the default foreground threshold of 1 on the binary M7 volume. This input is identical at any threshold from 1 through 255. [Release asset](https://api.github.com/repos/ScrollPrize/villa/releases/assets/787782221) · [Pinned GrowPatch source](https://github.com/ScrollPrize/villa/blob/e87544118ed046811635f62a47d04539e2a92c12/volume-cartographer/core/src/GrowPatch.cpp) · [VC3D command construction](https://github.com/ScrollPrize/villa/blob/e87544118ed046811635f62a47d04539e2a92c12/volume-cartographer/apps/VC3D/SegmentationCommandHandler.cpp)

The Paris.4 M7 test uses 257³ input windows at 9.6 µm. The secondary, manually annotated historical instance-ID test uses separate 256³ input windows at 7.91 µm. Its input is a shared foreground likelihood derived from the separately published semantic prediction; manually assigned target sheet IDs serve only as references for evaluation. [Primary frozen seeds](sources/seeds.json) · [Historical frozen seeds](sources/manual_ids/seeds.json)

## Configurations

- **default20** supplies a single XYZ seed and the 20-voxel spacing used in VC3D’s standard Create Segment pathway. Growth is allowed 11 generations. The minimum area gate is zero, so the CLI saves small outputs for measurement.
- **oriented8** gives the same stock optimizer the initial orientation available to the experimental method. Its seed is four corners of an 8-voxel square in the shared seed tangent plane. All four vertices lie 5.66 voxels from the supplied center, inside the 12-voxel exclusion around the seed. Stock resumes this small mesh for up to 23 generations. The official normal-grid generator uses matching knot spacing.
- The original stock loss weights remain at their compiled defaults. Each growth job uses random seed 42 and one OpenMP thread. Every `*_command.json` records its full arguments, duration and exit status. On Windows, a saved surface sometimes leaves a background thread alive. The wrapper then checks the terminal “saving” message, stable files, area metadata, matching shapes, and complete decodability of every geometry and generation TIFF before terminating that process. Its receipt explicitly records this path. Empty resulting surfaces remain in the benchmark.
- The two diagnostic replays enable the official `vis_losses` output. A larger `quality` is a smaller native `loss_snap`, the square root of GrowPatch’s evaluated Ceres snap residual cost. Each replay also preserves four additional raw loss channels. Equality of the diagnostic and original geometry is checked for every case. [Prediction lock](locks/stock/prediction_lock.json) · [Diagnostic replay lock](locks/stock/diagnostics_replay_lock.json)

## Reproduction

From the project root on Windows x64, first generate the shared benchmark input cubes. The helper installs the pinned stock portable binary in this directory if needed. Each output directory preserves original local-coordinate TIFFs, exact configuration, raw logs, per-file hashes and its global-coordinate evaluation copy. Standard Python 3.12 dependencies include NumPy, Numcodecs, Pillow, tifffile, requests and filelock.

```powershell
.\.venv\Scripts\python.exe scripts/growbench_stock.py --variant default20
.\.venv\Scripts\python.exe scripts/growbench_stock.py --variant oriented8
.\.venv\Scripts\python.exe scripts/growbench_stock_lock.py
.\.venv\Scripts\python.exe scripts/growbench_stock.py --variant default20 --diagnostics
.\.venv\Scripts\python.exe scripts/growbench_stock.py --variant oriented8 --diagnostics
.\.venv\Scripts\python.exe scripts/growbench_stock_lock.py --diagnostics
```

For the historical cohort, add `--manifest results/growbench/labels/manual_ids/seeds.json --output results/growbench/stock_manual --pitch-um 7.91` to either stock CLI configuration. To replay the additional historical half-maximum foreground thinning sensitivity, use `--variant oriented8 --grid-threshold 128 --diagnostics`. This threshold is applied only to the official contour generator; all arms receive the same original foreground input. The initial historical threshold-1 controls are preserved separately.

## Validation and additional cohorts

The initial diagnostic replays reproduced every XYZ value and validity bit on both 16-site sets. The historical development comparison also evaluated cutoff 192 on four development sites before carrying all three cutoffs to sixteen spatially distinct cubes with previously unseen target sheet identities. The second historical cohort uses the same preprocessor. The two Paris.4 pressure sets comprise eight high-competition cases and four high-gap cases. They use a 160-voxel radius in 385³ cubes and 17/37 stock generations at 20/8-voxel spacing. [Historical replays](locks/stock_manual/diagnostics_replay_lock.json) · [Historical threshold 128](locks/stock_manual/oriented8_th128_loss_lock.json) · [Historical threshold 192](locks/stock_manual/oriented8_th192_loss_lock.json) · [External historical seeds](sources/manual_external/seeds.json) · [Pressure seeds](sources/pressure/seeds.json) · [Gap seeds](sources/interrupted/seeds.json)

```powershell
.\.venv\Scripts\python.exe scripts/growbench_stock_supplement.py manual_external
.\.venv\Scripts\python.exe scripts/growbench_stock_supplement.py pressure
.\.venv\Scripts\python.exe scripts/growbench_stock_supplement.py interrupted
```

When the Windows native TIFF library's provisional filenames would approach its path-length limit, the runner supplies a short temporary output path, waits for native completion, and copies the unmodified TIFFs. The exact CLI call remains in each run's command receipt.

The public seed-alignment command was tested end to end on all eight held-out Paris.4 oriented-stock meshes. The replay converted the frozen global XYZ and validity mask to native float32 tifxyz, ran the CLI in a separate process, decoded the output TIFFs, and intersected their triangles with the benchmark rays. All eight masks and signed seed-normal shifts match the projection-only evaluation. At 2- and 4-voxel tolerances, every site score is identical. At 2 voxels, the connected-correct-area gain is 1.9491% and the error difference is −1.2925 percentage points. At the tightest 1-voxel tolerance, the CLI changes the reported gain by −0.0142 percentage point and error difference by +0.00357 percentage point compared with the ideal continuous translation. [Eight-site CLI verification](seed-alignment-replay.json) · [Replay script](scripts/growbench_stock_verify_align.py)
