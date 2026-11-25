# Whole-mesh tiling of the 8-voxel local blend (pre-registered 24 November 2026)

## Purpose

The validated blend refines one 96-voxel-radius chart around one seed. This protocol extends it to an entire GrowPatch `tifxyz` mesh. The output is a single refined mesh on the stock grid's topology, and it does not depend on where an evaluator places a seed.

This protocol is written before any tiled output is scored on labeled papyrus.

## Fixed method (not tuned)

1. **Tile seeds.** Lay a square lattice over the stock mesh's own grid indices, offset by half a lattice step from index (0, 0). Use every valid grid node on the lattice that has at least six valid vertices within 24 voxels. The lattice is independent of any evaluation seed.
2. **Tile frame.** The normal is the smallest-variance principal axis of the stock vertices within 24 voxels of the tile seed. The u tangent is the mesh's column direction projected into the tangent plane.
3. **Per tile.** Within each tile, use exactly the validated rule. Run the unmodified graph solver (`growbench_recover.solve`) at the tile seed on the same foreground. Rasterize stock and graph onto the 97 × 97 ray grid at 2-voxel spacing within a 96-voxel radius. Where both are single-valued and the graph lies within 8 voxels of stock along the tile normal, the shift is half the gap. Elsewhere the shift is 0. A tile whose solver abstains contributes nothing.
4. **Resampling.** Bilinearly resample the stock mesh to 2-voxel node spacing, the blend's validated resolution. For an 8-voxel GrowPatch grid this is a factor of 4.
5. **Transfer.** A resampled vertex is visible in a tile when it lies within the 96-voxel radius, the nearest ray is single-valued for stock, and the vertex is within 3 voxels of that ray's stock depth. Its shift is bilinear in the tile's shift field and is applied along the tile normal. The weight is w = cos²(π r / 192), where r is the in-plane distance from the tile seed.
6. **Combination.** The final displacement is the weighted mean of the tile displacements over the contributing tiles. Vertices with no contributing tile keep their stock position.

## Development candidates (selected on the original manual cohort only)

There are four candidates: lattice spacing of 32 or 64 voxels, crossed with two ways of choosing contributing tiles.

- **all-visible:** every tile in which the vertex is visible contributes, including tiles where the rule kept stock (shift 0).
- **shared-only:** only tiles where the vertex's nearest ray met the shift condition contribute.

Development data are the 16 original `manual_ids` regions (dev, test and confirm). They already informed earlier hypotheses and are never used to claim held-out performance. All three GrowPatch presets are used: `oriented8_loss` (cutoff 1), `oriented8_th128_loss` and `oriented8_th192_loss`.

**Selection rule.** Score each candidate within a 128-voxel radius around each region's seed with the regional instance-ID scorer. Pool over regions. Choose the candidate with the largest mean relative gain in seed-connected correct area across the three presets. Ties go to the lower mean strict error. The choice is written to a lock file before any held-out scoring.

## Held-out evaluation (run once after the lock)

The 16 `manual_external` regions, three presets, compared with stock and with the single-seed blend:

- **Radius 96:** the same footprint and scorer as the published single-seed result.
- **Radius 128:** the regional scorer.
- **Annulus 96 < r ≤ 128:** area no blend variant has been scored on before.

Reported per preset are connected correct area, strict error, the different-sheet share of named hits and pooled different-sheet area, region wins, ties and losses, and region-paired bootstrap 95% intervals (200,000 resamples).

These 16 regions were scored once before for the single-seed blend. The tiled evaluation is therefore a second look at the same cubes for the inner disc. The annulus is new evaluation area.

**Implementation check.** A single tile placed at the evaluation seed with weight 1 should approximately reproduce the single-seed blend at radius 96. Differences come only from transferring to resampled vertices. This is reported as a correctness check, not as evidence.

## Optional fresh cohort

If time allows, draw a third cohort with the held-out cohort's salted rule and a new salt (`growbench-manual-fresh-v1:`). It excludes all 32 used cube names and target IDs and requires a Chebyshev cube-origin separation of at least 256 voxels. Official GrowPatch would be run at the three presets with the same executable. The locked tiled blend is then scored once. With fewer than 8 qualifying regions, report them as descriptive only.
