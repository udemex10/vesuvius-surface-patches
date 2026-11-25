# A stricter tiled blend that must not raise the wrong-sheet share: protocol

Written 24 November 2026, before any of the rules below were scored. The tiled blend (`growpatch-tiled-blend-protocol.md`, locked at 16:04:49 the same day) found more target-sheet area on the held-out regions, but it failed the safety check. The share of named hits on a wrong sheet rose at every cutoff. This protocol tests stricter versions that try to keep most of the gain without that rise.

## What was looked at before writing this

Only the 16 development regions (`manual_ids`) were examined. For every scoring ray the locked tiled blend moved, the label before and after was recorded, along with simple features of the surface prediction along the ray (`dev_failure_analysis.py`, results summarised here).

- 511,026 moved rays. Of the moves off unlabeled voxels, 62,283 landed on the target and 11,578 on a wrong sheet (84% and 16%). 1,386 moved from the target to a wrong sheet.
- Checks that only look at one ray separate the two poorly. Call a move "bad" if it ends off the target (unlabeled to wrong sheet, target to wrong sheet, or target to unlabeled; 16,354 rays) and "good" if it ends on the target from elsewhere (63,141 rays). Refusing moves where another predicted ridge lies within 10 voxels of the landing point would drop 21% of the bad moves but also 14% of the good ones. Refusing moves where GrowPatch is closer to a different ridge than the landing ridge drops 11% of bad moves and 3% of good ones.
- The likely reason is that many wrong-sheet moves happen where GrowPatch is already following a neighbouring sheet. There, snapping onto the nearest ridge is locally right and globally wrong. A rule that compares overlapping tiles may catch this when the tiles disagree about which sheet they trace.

## Fixed parts

Everything in the locked tiled blend stays the same: the same GrowPatch meshes, the frozen graph solver, 32-voxel tile spacing, shared-only combination, the 8-voxel half-way rule, cos² weights, 4× resampling, and the same scorer (`growbench_tiled_eval.score_all`). Tiles are solved once per case, and every candidate is applied to the same solved tiles.

## Candidates

Each candidate only removes moves; none moves a point further or differently than the locked blend.

1. **locked**: the locked tiled blend, as a reference.
2. **ridge clearance c**, c ∈ {10, 14} voxels: a ray may move only if no other predicted ridge lies within c voxels of the landing point. Ridges are found as the graph solver finds them: local maxima of the prediction along the tile normal, normalised to the cube maximum, smoothed by 0.85 voxels, above 0.15.
3. **nearest ridge**: a ray may move only if the ridge nearest to GrowPatch's current depth is the landing ridge.
4. **tile consensus q**, q ∈ {0.5, 0.75, 1.0}: for every resampled vertex, among the tiles that see it (the existing visibility test), the cos²-weighted share of tiles that propose a move must be at least q. In addition, the proposed moves must agree: their weighted standard deviation must be at most 1 voxel. The move applied is the same weighted mean as in the locked blend.
5. **tile consensus 0.75 + nearest ridge**: candidates 3 and 4 (q = 0.75) together.

That is 8 candidates including the reference.

## Selection on the development regions

Score every candidate on the 16 development regions at cutoffs 1, 128 and 192, within 128 voxels of each seed.

- **Safety check:** the pooled share of named hits on a wrong sheet must not rise compared with official GrowPatch, at any of the three cutoffs (within 128 voxels).
- Among candidates that pass, choose the one with the largest mean gain in seed-connected target-sheet area within 128 voxels across the three cutoffs. Ties go to the earlier candidate in the list.
- If no candidate passes, stop. The stricter blend has failed at development, there is no held-out run, and that is reported.

The choice is written to `scripts/safe_blend_lock.json` with the hashes of this file, the evaluator and the blend code, before any held-out scoring.

## Held-out run (run once)

The locked candidate is run once on the 16 `manual_external` regions and the 4 `manual_fresh` cubes.

- The `manual_external` regions have already been scored twice (single-seed blend, then tiled blend). This is a third look. It is reported as such, and it carries less weight than a first look.
- The 4 fresh cubes are descriptive only (the fresh-cube rule required 8).
- **Success** means that on `manual_external`, within 128 voxels, the share of named hits on a wrong sheet does not rise at any cutoff, and the gain in seed-connected target-sheet area is positive at cutoff 128 with a region-paired 95% interval above zero. Results in the 96-voxel disc and the 96–128 ring are reported too.
- Whatever happens is reported, including a failure.

## Result (added after the development run, 24 November 2026, 18:47)

No candidate passed the safety check on the development regions, so the protocol stopped there. There was no held-out run, and the `manual_external` regions were not scored a third time. The lock file records `chosen: null` (`scripts/safe_blend_lock.json`, with the hashes of this protocol as written above, the evaluator and the blend code). As a built-in check, the "locked" reference reproduced the published development scores of the tiled blend exactly.

Within 128 voxels of each seed, on the 16 development regions:

| Candidate | Mean gain in target-sheet area | Wrong-sheet share at cutoffs 1 / 128 / 192 (GrowPatch → candidate) |
|---|---:|---|
| locked (reference) | +25.2% | 8.90→10.05 / 5.77→6.76 / 7.84→9.79 |
| ridge clearance 10 | +24.3% | 8.90→9.96 / 5.77→6.77 / 7.84→9.60 |
| ridge clearance 14 | +22.3% | 8.90→9.86 / 5.77→6.79 / 7.84→9.28 |
| nearest ridge | +25.3% | 8.90→10.13 / 5.77→6.79 / 7.84→9.87 |
| tile consensus 0.5 | +4.0% | 8.90→9.76 / 5.77→5.94 / 7.84→8.91 |
| tile consensus 0.75 | +3.9% | 8.90→9.54 / 5.77→5.86 / 7.84→8.52 |
| tile consensus 1.0 | +2.0% | 8.90→9.00 / 5.77→5.79 / 7.84→7.88 |
| tile consensus 0.75 + nearest ridge | +3.9% | 8.90→9.53 / 5.77→5.86 / 7.84→8.51 |

What this shows:

- Checks on single rays (ridge clearance, nearest ridge) keep the gain but barely change the wrong-sheet rise.
- Requiring the overlapping tiles to agree cuts the rise, almost completely only when every tile must agree, but it also removes most of the gain, and the share still rises slightly.
- A follow-up look at the same development results: at cutoffs 1 and 128, 61% and 52% of the extra wrong-sheet area comes from regions where official GrowPatch never connects the target sheet to the seed (none does at cutoff 192). A rule that sees only the mesh and the prediction cannot tell which sheet is the target, so where the mesh follows a neighbouring sheet it cannot avoid snapping onto it.

The tiled blend therefore stays as locked on 24 November at 16:04:49, with its trade-off reported as before.
