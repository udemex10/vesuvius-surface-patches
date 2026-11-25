# A stricter blend that did not work

The tiled blend finds more of the target sheet, but it also puts more surface on wrong sheets, so it fails the safety check. We tried to fix that with a stricter version. It did not work, and this folder records the attempt.

**What we tried.** Before scoring any of them, we wrote down seven stricter rules and how we would pick one ([protocol](../../growpatch-blend/protocols/growpatch-safe-blend-protocol.md)). Each rule only removes moves; none moves a point further. Some refuse a move when another predicted ridge is close to where the point would land, or when GrowPatch is already closer to a different ridge. Others only move a point when the overlapping tiles that see it agree on the move. A rule had to pass the safety check on the 16 older development regions, at all three cutoffs, before it could be tried on the test regions.

**What happened.** None passed, so we stopped there. The test regions were not scored again.

| Rule | Target-sheet gain | Wrong-sheet share at cutoff 128 |
|---|---:|---|
| Current tiled blend | +25.2% | 5.77% → 6.76% |
| No other sheet within 10 voxels of the landing point | +24.3% | 5.77% → 6.77% |
| Every overlapping tile must agree | +2.0% | 5.77% → 5.79% |

Gains are the mean across the three cutoffs, within 128 voxels of each seed, on the development regions. All eight rows are in the protocol.

The rules that keep the gain barely change the wrong-sheet rise. The rules that cut the rise also throw away almost all of the gain, and even when every tile must agree the share still creeps up.

**Why.** At cutoffs 1 and 128, more than half of the extra wrong-sheet area on the development regions (61% and 52%) comes from regions where GrowPatch's mesh never connects the target sheet to the seed. At cutoff 192 it is spread across regions. The blend only sees the mesh and the surface prediction. It has no way to know which sheet is the target, so where the mesh follows a neighbouring sheet, it tightens the mesh onto that sheet. Fixing that needs information the blend does not have, such as the seed point itself or a person checking the sheet.

## Files

| File | What it is |
|---|---|
| `safe_blend_lock.json` | The selection record: no rule chosen, the full ranking, the time, and the hashes of the protocol, evaluator and code |
| `growpatch-safe-blend-protocol-at-lock.md` | The protocol exactly as it was when the lock was written. Its hash matches the lock |
| `dev_manual_ids.json` | Every score for every rule and development region |

The code is `growpatch-blend/scripts/growbench_safe_blend.py` (the rules), `growbench_safe_eval.py` (scoring) and `growbench_safe_failure_analysis.py` (the first look at where wrong-sheet moves come from). As a check, the "current tiled blend" row reproduces the published development scores exactly.
