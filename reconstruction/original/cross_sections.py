"""Reproduce or verify the local cross-section geometry package."""
import argparse
import hashlib,io,json,os,sys
from pathlib import Path
import numpy as np
PACKAGE=Path(__file__).resolve().parent
CASES=['PHerc0125_y3310','PHerc0125_y3706','PHerc0268_y2193','PHerc0268_y2991']
def digest(path):
    with path.open('rb')as s:return hashlib.file_digest(s,'sha256').hexdigest()
def hash_as_npy(arr):
    data=io.BytesIO();np.save(data,arr);return hashlib.sha256(data.getbuffer()).hexdigest()
def verify_manifest():
    """Check packaged inputs, code, measurements, and data with literal relative paths."""
    manifest = PACKAGE / 'SHA256SUMS'
    count = 0
    for line in manifest.read_text(encoding='utf-8').splitlines():
        expected, relative_path = line.split('  ', 1)
        item = (PACKAGE / relative_path).resolve()
        if not item.is_relative_to(PACKAGE) or digest(item) != expected:
            raise RuntimeError(f'File checksum differs: {relative_path}')
        count += 1
    print(f'Verified {count} package-file checksums.')
def assert_same_locks(loc,ref):
    def sig(p):
        rows=json.loads(p.read_text())['chunks']
        return sorted((r['url'],tuple(r.get('l0_z_range_in_chunk',[])),r.get('sha256'),r.get('missing',False)) for r in rows)
    if sig(loc)!=sig(ref):raise RuntimeError(f'Raw source range lock mismatch in {loc}')
def verify(case):
    d=PACKAGE/'cases'/case;md=json.loads((d/'native_meta.json').read_text())
    raw=np.load(d/'native.npz');ct=raw['ct17'];mask=raw['mask']
    if hash_as_npy(ct)!=md['ct17_sha256'] or hash_as_npy(mask)!=md['mask_sha256']:raise RuntimeError(f'Native payload differs in {case}')
    checks=json.loads((d/'distortion.json').read_text())
    for k,w in checks['mesh_hashes'].items():
        if digest(d/(case+'.tifxyz')/(k+'.tif'))!=w:raise RuntimeError(f'Mesh hash differs in {case}')
    if ct.shape[0]!=17 or ct.dtype!=np.uint8 or mask.shape!=ct.shape[1:] or not mask.any():raise RuntimeError(f'Invalid CT or support in {case}')
    print(f'{case}: {ct.shape}, {int(mask.sum()):,} supported pixels, mesh and CT digests verified.')
def reproduce(case,work,cache):
    specimen=case.split('_')[0];seed=int(case.split('_y')[1])
    os.environ['CROSS_SECTIONS_WORK']=str(work)
    os.environ['FIRST_LETTERS_GRID_CASE']='PHerc0268_inner' if case=='PHerc0268_y2991' else specimen
    if cache:os.environ['CROSS_SECTIONS_RAW_CACHE']=str(cache)
    sys.path.insert(0,str(PACKAGE/'src'))
    import first_letters_0125_surface as surface
    import first_letters_0125_render as projection
    import first_letters_0125_validation as validation
    import first_letters_surface_ablation as ablation
    import first_letters_intrinsic as intrinsic
    import first_letters_intrinsic_render as native
    import first_letters_surface_figure as figure
    surface.run()
    projection.main(seed)
    validation.main()
    ablation.run(seed)
    intrinsic.main(seed)
    native.main(seed)
    figure.main(seed)
    output=work/'results/first_letters_deep'/case
    ref=PACKAGE/'cases'/case
    assert_same_locks(output/'source_ranges.json',ref/'projection_raw_ranges.json')
    assert_same_locks(output/'intrinsic/source_ranges.json',ref/'raw_ranges.json')
    for k in 'xyz':
        if digest(output/'intrinsic'/(case+'.tifxyz')/(k+'.tif'))!=digest(ref/(case+'.tifxyz')/(k+'.tif')):raise RuntimeError(f'Output mesh differs, {k}. Check the pinned library versions.')
    expected=json.loads((ref/'native_meta.json').read_text())
    if digest(output/'intrinsic/ct17.npy')!=expected['ct17_sha256'] or digest(output/'intrinsic/mask.npy')!=expected['mask_sha256']:raise RuntimeError('Native CT or mask differs. See the generated source manifests and numerical versions.')
    print('Reproduced locked mesh, raw range hashes, CT and mask:',output)
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    v=sub.add_parser('verify',help='Verify the included geometry and surface volumes offline.');v.add_argument('--case',choices=CASES)
    r=sub.add_parser('reproduce',help='Rebuild one surface from public grid and CT sources.');r.add_argument('--case',required=True,choices=CASES);r.add_argument('--work-dir',required=True,type=Path);r.add_argument('--raw-cache',type=Path,help='Optional directory containing the same public native CT byte ranges from a prior run.')
    a=parser.parse_args()
    if a.command=='verify':
        verify_manifest()
        for case in [a.case]if a.case else CASES:verify(case)
    else:reproduce(a.case,a.work_dir.resolve(),a.raw_cache.resolve() if a.raw_cache else None)
if __name__=='__main__':main()
