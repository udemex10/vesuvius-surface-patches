"""Lock the saved stock outputs before scoring ground truth and compare diagnostic reruns."""
import argparse
import json
from pathlib import Path

import numpy as np
from growbench_stock import BIN, OUTPUT, SEEDS, digest, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=SEEDS)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--diagnostics',action='store_true')
    parser.add_argument('--single',help='Hash-lock one additional stock sensitivity output without replaying it.')
    args=parser.parse_args()
    rows=json.loads(args.manifest.read_text())['sites']
    lock={'seed_manifest_sha256':digest(args.manifest),'grow_exe_sha256':digest(BIN/'vc_grow_seg_from_seed.exe'),
          'gen_normalgrids_exe_sha256':digest(BIN/'vc_gen_normalgrids.exe'),'cases':[]}
    for row in rows:
        for variant in ([args.single] if args.single else ('default20','oriented8')):
            original=args.output/variant/row['id']/'surface.npz'
            src=original
            entry={'id':row['id'],'split':row['split'],'variant':variant,
                   'surface_sha256':digest(original)}
            if args.diagnostics:
                src=args.output/(variant+'_loss')/row['id']/'surface.npz'
                with np.load(src) as rerun,np.load(original) as first:
                    entry['xyz_and_valid_identical']=bool(np.array_equal(first['xyz'],rerun['xyz'],equal_nan=True) and np.array_equal(first['valid'],rerun['valid']))
                    entry['diagnostic_surface_sha256']=digest(src)
                    valid=rerun['valid']
                    q=rerun['quality'][valid]
                    entry['quality_minmax']=[float(q.min()),float(q.max())] if len(q) else []
            record=json.loads((src.parent/'record.json').read_text())
            entry.update(input_sha256=record['input_sha256'],valid_vertices=record['valid_vertices'])
            lock['cases'].append(entry)
    name=f'{args.single}_lock.json' if args.single else ('diagnostics_replay_lock.json' if args.diagnostics else 'prediction_lock.json')
    save(args.output/name,lock)
    print(name,digest(args.output/name),len(lock['cases']),flush=True)


if __name__=='__main__': main()
