"""Apply the pre-established recovery weights to a 385-voxel source cube."""
import argparse
import json
from pathlib import Path

import growbench_recover as base
from growbench_recover_phase import run as preserve_seed_phase


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--site",action="append")
    args=p.parse_args()
    base.RADIUS=160
    base.DEPTH=175
    for s in json.loads((args.cohort/"seeds.json").read_text())["sites"]:
        if args.site and s["id"] not in args.site:continue
        base.run_case(s,args.cohort/"inputs",args.out)
        preserve_seed_phase(s,args.out)
