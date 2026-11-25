# Recovering a seeded papyrus sheet from CT surface predictions

Sixteen previously untargeted PHerc. Paris. 4 regions have distinct manual sheet IDs, public source coordinates, reconstruction inputs, and frozen official GrowPatch outputs. This release lets a reviewer see where a change recovered more of the intended sheet and where it crossed onto a neighboring one. The reconstruction programs take a seed, an upstream foreground and, for the blend, a stock mesh. The evaluator alone uses the sheet annotations.

## Start with one real region

Use Python 3.12. Run the commands from the extracted archive. They fetch one frozen case directly from the organizer, reconstruct the two candidates, verify the measured areas, and produce actual coordinate meshes. Use a new output directory for each replay.

```text
python -m pip install -r requirements.txt
python scripts/growbench_labels_replay.py manual_external external_01 --manifests sources --output-root ../fresh_inputs
python scripts/growbench_review.py . ../case01-review --case external_01 --source-root ../fresh_inputs
```

For all 16 cases, fetch the remaining IDs with the replay script and run the last command without the case selector. This Python command fetches them in order.

```text
python -c "import subprocess,sys; [subprocess.run([sys.executable,'scripts/growbench_labels_replay.py','manual_external',f'external_{i:02d}','--manifests','sources','--output-root','../fresh_inputs'],check=True) for i in range(1,17)]"
python scripts/growbench_review.py . ../all-review --source-root ../fresh_inputs
```

The private offline edition also contains the organizer-derived inputs. From that edition the following one-command replay needs no network access.

```text
python scripts/growbench_review.py . ../all-offline
```

## Use the method on another local mesh

The local blend moves a GrowPatch point halfway toward a second, graph-traced sheet estimate wherever both surfaces have a single local intersection and agree within eight voxels. Otherwise it keeps the stock estimate. The 97 × 97 tangent grid uses two-voxel spacing and excludes multivalued output from the delivered chart. The separately exported OBJ clips the native graph triangles at the confidence cutoff. A new scan requires a registered surface-foreground cube and a trusted seed with its local orientation. The validation footprint was a 96-voxel-radius disc. Accuracy outside that disc or in a new scan requires a separate check.

```text
python scripts/growbench_refine_mesh.py cases/external_01/stock_128.npz ../fresh_inputs/manual_external/inputs/external_01/m7.npy cases/external_01/case.json ../case01-mesh
```

To use a normal GrowPatch coordinate triplet, supply its directory in the first argument. Add `--stock-local` if its XYZ coordinates are relative to the supplied cube origin. The output directory contains an editable `local-blend.tifxyz` mesh in the original global XYZ frame, `graph-confidence-cut.obj` and the seed and reconstruction receipt. Another CLI, `scripts/growbench_seed_align.py`, applies the separate seed-offset correction to a local mesh using XYZ and a trusted normal.

## What the comparisons establish

The original two-direction method lost 13.2% connected area on the eight held-out vetted patches, and lost again on the four reserved patches. On the fresh 16 manually labeled regions, the follow-up graph at confidence 0.575 increased measured connected correct area by 3.86% versus GrowPatch's cutoff-1 input configuration, with seven region wins and nine losses. A region bootstrap includes a loss. The local blend increased connected correct area by 13.95% versus the cutoff-128 configuration, in 15 regions with one tie. It retained less total connected area than cutoff-1 stock. Most of its benefit came from turning unlabeled placements into placements on the intended sheet. Different-sheet hits among named assignments fell from 1.308% to 1.244%. Different-sheet hits among all accepted points rose from 0.979% to 1.032%. These results support a local refinement, with limited evidence of reduced layer switching. [Paired results and uncertainty](sources/external_resampling.md) · [Area change in each region](prize_comparison.png) · [Export measurements](mesh-export-replay.json)

The source edition contains computed stock coordinates and per-case measurements. Both editions contain the three tested native input thresholds, the original losing comparisons, registration metadata, independently recorded resampling diagnostics, and code for the original multi-cohort experiment. The one-case instructions above are portable. Other project-level scripts retain their original working-directory conventions. Native stock runs used the official Volume Cartographer executable; the offline verification starts from its recorded coordinates. [Pinned GrowPatch source](https://github.com/ScrollPrize/villa/blob/e87544118ed046811635f62a47d04539e2a92c12/volume-cartographer/core/src/GrowPatch.cpp) · [Technical comparison](stock-methods.md) · [Frozen protocols](protocols)

## Credits and terms

Historical Scroll 1 data were obtained from EduceLab-Scrolls, copyright EduceLab/The University of Kentucky. The harmonized instance annotations and semantic foreground are from the Vesuvius Challenge public release. See Parsons, Parker, Chapman, Hayashida and Seales (2023), *EduceLab-Scrolls: Verifiable Recovery of Text from Herculaneum Papyri using X-ray CT*. The modern Paris.4 inputs are credited separately to Giorgio Angelotti and colleagues, *Vesuvius Challenge – CT Scans of Herculaneum Papyri*. The numerical measurements, local reconstruction code and reconstructed coordinates are project contributions. The source edition omits the original organizer arrays and source imagery and retrieves them from their official location. The offline edition is a private review copy. Its underlying source arrays are subject to the data server's redistribution terms. The new code is released under the MIT License in the repository root (LICENSE). [Dataset citation and licensing](https://scrollprize.org/data#how-to-cite) · [EduceLab paper](https://doi.org/10.48550/arXiv.2304.02084) · [Official legacy data terms](https://dl.ash2txt.org/LICENSE.txt) · [Annotation provenance](https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumetric-instance-labels/README.txt)
