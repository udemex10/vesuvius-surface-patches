"""Run a frozen ink reader on native stacks reproduced with review.py render."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
CHECKPOINT = "87ad4b751bc914ed8ae22aae1d81ad214712300ec0d0f8966f23d9a80b175364"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renders", type=Path, required=True, help="The work/renders directory")
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--case",help="One direction/component or all seven if omitted")
    args=parser.parse_args()
    if sha(args.checkpoint) != CHECKPOINT:raise ValueError("Checkpoint digest differs from the published model")
    sys.path.insert(0,str(HERE / "reader_code"))
    import torch
    from hybrid_model import make_hybrid
    from transfer_core import supported_starts, score
    expected=json.loads((HERE/"array_hashes.json").read_text())
    if args.case and args.case not in expected:parser.error("Unknown extension case")
    torch.set_num_threads(3)
    model = make_hybrid(args.checkpoint)[0].cuda().eval()
    reports={}
    for name in ([args.case] if args.case else sorted(expected)):
        row=expected[name]
        folder=args.renders/("PHerc0125_"+name)
        for key in ("ct17","mask"):
            if sha(folder/(key+".npy"))!=row[key+"_npy_sha256"]:
                raise ValueError(f"Source image differs: {name} {key}")
        ct=np.load(folder/"ct17.npy",mmap_mode="r")
        mask=np.load(folder/"mask.npy",mmap_mode="r")
        positions=supported_starts(mask)
        if len(positions)!=row["windows"]:raise ValueError(f"Window-count mismatch for {name}")
        report={"valid_input_windows":len(positions),"finite_values":0,"hash_matches":True}
        if positions:
            dest=args.output/name
            dest.mkdir(parents=True,exist_ok=True)
            forward,reverse=score([model],ct,positions,False)
            finite=np.isfinite(forward)
            if not np.array_equal(finite,np.isfinite(reverse)) or np.any(finite & ~mask):
                raise ValueError(f"Invalid output support {name}")
            hashes={}
            for key,array in (("forward",forward),("reverse",reverse),("scored_support",finite)):
                path=dest/(key+".npy")
                np.save(path,array)
                hashes[key]=sha(path)
                report["hash_matches"] &= hashes[key]==row[key+"_npy_sha256"]
            report.update(finite_values=int(2*finite.sum()),actual_hashes=hashes)
        reports[name]=report
        print(name,report["valid_input_windows"],report["finite_values"],report["hash_matches"])
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"receipt.json").write_text(json.dumps(reports,indent=2)+"\n",encoding="utf-8")
    if not all(v["hash_matches"] for v in reports.values()):
        raise ValueError("Output files were saved. They differ from the reference GPU's exact NumPy hashes.")


if __name__=="__main__":
    main()
