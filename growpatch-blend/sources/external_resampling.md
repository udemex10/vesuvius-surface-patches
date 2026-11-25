# Independent check on sixteen previously untargeted sheet regions

Five area totals per method and site were recomputed from the frozen meshes and native 3D sheet-ID volumes. The independent implementation reproduced the archived area totals exactly for both comparisons. It reuses the common projection lattice and independently samples the native sheet IDs, then traces the seed-connected component by binary propagation. The spatial check grouped source cubes within two cube widths of each other, producing seven groups. Every bootstrap draw resampled paired sites or paired groups for both methods. [Diagnostic rule](external_resampling_rule.md) · [Reference and source data](README.md)

The table gives candidate minus official GrowPatch differences. Percentage-point differences refer to area-weighted rates. Interval columns show paired 95% percentile intervals from 200,000 resamples. Connected area is expressed as cm² across sixteen sites. A plus sign favors the candidate for area, and a minus sign favors it for error.

The [all-site native-ID maps](external_blend_changes.png) locate the gains and losses for the threshold 128 blend, including the persistent wrong sheet on external site 10 and the small new wrong-sheet assignments on site 08.

| Method and metric | Difference | Site interval | Spatial interval | Site changes, better / equal / worse |
|---|---:|---:|---:|---:|
| Graph, connected correct area | +0.00759 | −0.01802 to +0.04122 | −0.02773 to +0.02796 | 7 / 0 / 9 |
| Graph, strict error | −10.929 | −23.779 to −3.064 | −17.689 to −0.833 | 13 / 0 / 3 |
| Graph, wrong fraction among named assignments | −0.696 | −7.828 to +3.981 | −4.884 to +5.849 | 2 / 4 / 10 |
| Threshold 128 blend, connected correct area | +0.02292 | +0.01436 to +0.03263 | +0.01445 to +0.02705 | 15 / 1 / 0 |
| Threshold 128 blend, strict error | −8.039 | −10.945 to −5.449 | −8.962 to −5.679 | 15 / 1 / 0 |
| Threshold 128 blend, wrong fraction among named assignments | −0.0638 | −0.3823 to +0.0976 | −0.2737 to +0.0909 | 1 / 11 / 4 |
| Threshold 128 blend, wrong fraction of all accepted area | +0.0529 | −0.0023 to +0.1444 | 0.0000 to +0.1185 | 1 / 11 / 4 |

**Graph at confidence 0.575.** External site 10 accounts for the sign of the pooled known-wrong result. Omitting that site changes the difference to +2.116 points. Across all one-site omissions, the difference ranges from −1.822 to +2.116 points. The connected-area difference also changes sign. Strict error falls after every one-site omission. A coarse three-group division into absolute axial slabs gives a strict-error interval of −16.208 to +1.276 points. One slab contains a single case. These broad slabs provide a sensitivity check with very few units. [Graph results and every site omission](external_resampling.json)

**Ungated eight-voxel blend on official threshold 128.** The methods accepted the same 0.249595 cm². Seed-connected target area increased from 0.164341 to 0.187262 cm² (+13.95%). Strict error, which includes unknown, ambiguous and different named assignments, fell from 26.12% to 18.09%. Both effects retain their direction after omitting any one site, any one of the seven geographic groups, or any one of three coarse axial slabs. [Blend results and every site omission](external_resampling_blend128.json)

The fraction of positively named material assigned to another sheet fell from 1.3078% to 1.2440%. Its paired interval allows a small increase. The *area* assigned to another sheet rose from 0.00244339 to 0.00257544 cm² (+0.013205 mm²), or 0.9789% to 1.0318% of the identical accepted footprint. The named-only denominator grew as more target material was recovered. The fixed-footprint comparison supports greater target recovery and a lower strict error. Evidence on another named sheet depends on which denominator the question requires. The inference applies to these 16 ancient sheet regions from the same acquisition. [Native-ID totals](external_resampling_blend128.json)

Comparing the same rays before and after the blend explains where the additional target area came from. The shift took 2.05150 mm² from unassigned voxels into the target, and 0.03991 mm² in the reverse direction. Rays on another named sheet moved to the target across 0.00093 mm². Rays previously on the target moved to another named sheet across 0.00594 mm², confined to external sites 04 and 08. The principal measured change was bringing previously unassigned positions onto the intended sheet. [Paired ray transitions](external_resampling_blend128.json)
