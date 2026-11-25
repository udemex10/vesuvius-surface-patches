# PHerc. 125 mesh faces between corroborating organizer curves

The seven native meshes provide 1.347961 cm². Of that area, **0.813130 cm² (60.3%)** meets the four-corner control below. These faces form a subset of the reported area.

| Export | All mesh area (cm²) | Bracketed area (cm²) | Largest bracketed component (cm²) |
|---|---:|---:|---:|
| east | 0.338667 | 0.274119 | 0.266444 |
| north | 0.229508 | 0.207309 | 0.207309 |
| south | 0.401486 | 0.137690 | 0.107877 |
| south_2 | 0.330639 | 0.188352 | 0.181495 |
| south_3 | 0.010649 | 0.000000 | 0.000000 |
| west | 0.026468 | 0.000000 | 0.000000 |
| west_2 | 0.010543 | 0.005659 | 0.005659 |

Of the exported east mesh, **0.179645 cm²** also falls completely inside the largest connected oriented-stock agreement mask. Every two-voxel stock face touched by an included quad's native XZ bounding box is qualified at the three-voxel tolerance. This yields a restricted subset where both the original curves and a separate official GrowPatch run corroborate the coordinates using the same source M7. See also_inside_stock_agreement_quads in east_faces.npz.


A retained quad lies entirely within a union of 8-by-32-native-voxel rectangles, each with an original organizer curve at every corner. Each curve was chosen solely from its unique agreement with the older surface through a 128-voxel interior band. Its original stored path continues across the boundary without a large reversal. The carried height agrees with the new estimate within three native Y voxels at each tested corner. We require every rectangle touched by the quad's entire projected bounding box to pass. This intentionally excludes both unobserved border strips and cells touching a failed control. Adjacent controls on the same plane may lie on the same stored polyline.

The controls measure boundary continuity at discrete positions. Surface identity between adjacent source planes and the winding identity of the common M7 prediction still require expert review. Every qualifying face is also in the previously reported seven-export total. Geometry areas use native 3D triangles at 9.362 µm per voxel.

Each *_faces.npz contains a two-dimensional qualified_quads Boolean mask indexed by the upper-left vertex of each cell in the corresponding addendum TIFXYZ. The report binds every mask to its three source TIFF SHA-256 digests. For a source-supported CT overlay, expand the mask eight times along both axes. From the root of this release, regenerate every face mask using the included derived route-footprint and stock-agreement diagnostics. To regenerate the underlying geometry from original public source, use the root review command.

```text
python curve-faces/growbench_recover_unread_brackets.py extensions ../fresh_masks
```
