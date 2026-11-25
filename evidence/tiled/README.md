# Whole-mesh tiling: evidence

| File | Contents |
|---|---|
| `tiled_blend_lock.json` | Locked setting (32-voxel lattice, shared-only), development ranking, timestamps and SHA-256 of the protocol, evaluator and tiling code at lock time |
| `dev_manual_ids.json` | Development run: 16 original `manual_ids` regions, 3 GrowPatch cutoffs, 4 candidates. Pooled summaries and every per-region row |
| `heldout_manual_external.json` | The single locked held-out run: 16 `manual_external` regions, 3 cutoffs. Stock, single-seed blend, single-tile check and tiled blend, scored within 96 voxels, within 128 voxels and in the 96–128 ring |
| `transitions_manual_external.json` | Ray-by-ray class changes (target, other named sheet, unlabeled, ambiguous, absent) from stock to tiled, within 12–128 voxels |
| `growbench_tiled_blend_at_lock.py` | The tiling code exactly as hashed in the lock |
| `fresh_manual_fresh.json`, `fresh_seeds.json`, `fresh_refs.json`, `fresh_sampling_rule.json` | The 4 fresh cubes that met the pre-registered rule, with seeds, target IDs and selection history. Fewer than 8 qualified, so they are descriptive only |

Protocol: [`growpatch-blend/protocols/growpatch-tiled-blend-protocol.md`](../../growpatch-blend/protocols/growpatch-tiled-blend-protocol.md). Code: `growpatch-blend/scripts/growbench_tiled_blend.py` (method and CLI), `growbench_tiled_eval.py` (scoring), `growbench_tiled_transitions.py` and `growbench_labels_manual_fresh.py`. The evaluators expect the project's data layout (`results/growbench/...`). The public inputs can be rebuilt with `growbench_labels_replay.py`, and the official GrowPatch meshes with `growbench_stock.py` and the pinned VC3D release.

**One post-lock code change.** After the lock, `growbench_tiled_blend.py` was changed in one place. Every `ValueError` from the frozen graph solver now counts as a tile abstention. Before, only two message patterns did. The full PHerc. 125 mesh raised a third solver message ("no connected ridges"). In the old code that would have stopped the run. Every labeled run in this folder completed, so none raised it, and its scores are unaffected. The file exactly as locked is kept as `growbench_tiled_blend_at_lock.py`. Its SHA-256 equals the lock's `tiled_sha256` (`810e512e…`), and it differs from the current script only in that `except` block.

Scoring areas are in cm² in the JSON files; the README converts them to mm². Intervals are region-paired bootstrap percentiles (200,000 draws, seed 20261124).
