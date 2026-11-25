"""Run the organizer's unmodified release of GrowPatch on the blinded shared M7 cubes."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import zipfile

import numpy as np
import requests
from numcodecs import Blosc
from PIL import Image
import tifffile
from filelock import FileLock

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'results/growbench/stock'
UPSTREAM = OUTPUT / 'upstream'
RELEASE = 'VC3D-e875441-2026-11-24-win64'
BIN = UPSTREAM / RELEASE / 'bin'
SEEDS = ROOT / 'results/growbench/labels/seeds.json'
ARCHIVE_SHA256 = 'dfdebbdf480f97da455ae821d2bafa8c4402d30006a37aff60a22237bc53f1ea'
ASSET = 'https://api.github.com/repos/ScrollPrize/villa/releases/assets/787782221'


def install_release():
    if os.name!='nt':
        raise RuntimeError('This replay script reproduces the pinned Windows x64 stock build.')
    UPSTREAM.mkdir(parents=True,exist_ok=True)
    archive = UPSTREAM/'stock.zip'
    with FileLock(str(UPSTREAM/'install.lock')):
        if not archive.exists():
            temp = UPSTREAM/'stock.zip.partial'
            with requests.get(ASSET,headers={'Accept':'application/octet-stream'},stream=True,timeout=90) as response:
                response.raise_for_status()
                with temp.open('wb') as dest:
                    for block in response.iter_content(chunk_size=1<<20): dest.write(block)
            if digest(temp)!=ARCHIVE_SHA256:
                raise ValueError('Downloaded asset did not match the pinned release.')
            temp.replace(archive)
        if digest(archive)!=ARCHIVE_SHA256:
            raise ValueError('Release archive hash mismatch.')
        if not (BIN/'vc_grow_seg_from_seed.exe').exists() or not (BIN/'vc_gen_normalgrids.exe').exists():
            with zipfile.ZipFile(archive) as source:
                for member in source.infolist():
                    if member.filename.startswith(RELEASE+'/bin/'):
                        source.extract(member,UPSTREAM)


def digest(file):
    with file.open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def save(path, record):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    temp.replace(path)


def write_zarr(volume, path, pitch, source_hash):
    path.mkdir(parents=True,exist_ok=True)
    depth,height,width=volume.shape
    save(path/'meta.json',{'uuid':path.stem,'type':'vol','name':path.stem,'voxelsize':pitch,'format':'zarr',
                           'width':width,'height':height,'slices':depth})
    level = path/'0'
    step = 64
    comp = Blosc(cname='zstd',clevel=3,shuffle=0)
    marker=path/'complete.json'
    if marker.exists():
        cached=json.loads(marker.read_text())
        if 'source_sha256' in cached and (cached['source_sha256']!=source_hash or cached['pitch_um']!=pitch):
            raise ValueError(f'Stale cached input: {path}')
        if 'source_sha256' not in cached:
            verify_zarr(volume,level,step,comp)
            save(marker,{'shape':list(volume.shape),'source_sha256':source_hash,'pitch_um':pitch})
        return
    level.mkdir(parents=True,exist_ok=True)
    save(path/'.zgroup',{'zarr_format':2})
    save(path/'.zattrs',{'multiscales':[{'version':'0.4','axes':[{'name':name,'type':'space','unit':'micrometer'} for name in 'zyx'],'datasets':[{'path':'0','coordinateTransformations':[{'type':'scale','scale':[pitch]*3}]}]}]})
    save(level/'.zarray',{'shape':list(volume.shape),'chunks':[step]*3,'dtype':'|u1','zarr_format':2,
                         'compressor':comp.get_config(),'order':'C','dimension_separator':'.','filters':None,'fill_value':0})
    save(level/'.zattrs',{})
    for xyz in np.ndindex(*[int(np.ceil(n/step)) for n in volume.shape]):
        roi = tuple(slice(i*step,(i+1)*step) for i in xyz)
        data = volume[roi]
        chunk = np.zeros((step,step,step),'u1')
        chunk[tuple(slice(0,n) for n in data.shape)] = data
        (level/'.'.join(map(str,xyz))).write_bytes(comp.encode(chunk))
    verify_zarr(volume,level,step,comp)
    save(marker,{'shape':list(volume.shape),'source_sha256':source_hash,'pitch_um':pitch})


def verify_zarr(volume,level,step,comp):
    for idx in np.ndindex(*[int(np.ceil(n/step)) for n in volume.shape]):
        src=volume[tuple(slice(i*step,(i+1)*step) for i in idx)]
        stored=np.frombuffer(comp.decode((level/'.'.join(map(str,idx))).read_bytes()),dtype='u1').reshape((step,)*3)
        if not np.array_equal(stored[tuple(slice(0,n) for n in src.shape)],src):
            raise ValueError('Cached stock input differs from shared array.')


def command(args, base, name, threads, result_dir=None):
    log = base/(name+'.log')
    env = dict(os.environ,OMP_NUM_THREADS=str(threads),OPENBLAS_NUM_THREADS='1')
    started = time.monotonic()
    completed_output = False
    with log.open('wb') as stream:
        with subprocess.Popen([str(v) for v in args],stdout=stream,stderr=subprocess.STDOUT,env=env,cwd=base) as proc:
            latest = None
            while proc.poll() is None:
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    if time.monotonic()-started>1200:
                        proc.kill()
                        raise
                    # Some Windows builds strand one idle thread after vc3d saves its
                    # mesh. Accept its native output only after complete TIFF decoding,
                    # footer metadata, the final "saving" message and four stable polls.
                    files = [(result_dir or base/'surface')/f for f in ('x.tif','y.tif','z.tif','generations.tif','meta.json')]
                    if os.name!='nt' or name!='grow' or not all(f.exists() for f in files):
                        continue
                    state=tuple((f.stat().st_mtime_ns,f.stat().st_size) for f in files)
                    if latest is None or state!=latest[0]:
                        latest=state,time.monotonic()
                    if time.monotonic()-latest[1]<4 or 'saving ' not in log.read_text(errors='replace')[-450:]:
                        continue
                    meta=json.loads(files[-1].read_text())
                    planes=[np.array(Image.open(f)) for f in files[:-1]]
                    if 'area_vx2' not in meta or not all(plane.shape==planes[0].shape for plane in planes):
                        raise RuntimeError('Incomplete GrowPatch output during Windows teardown')
                    proc.terminate()
                    proc.wait()
                    completed_output=True
    save(base/(name+'_command.json'),{'command':list(map(str,args)),'OMP_NUM_THREADS':threads,'elapsed_seconds':time.monotonic()-started,'exit_code':proc.returncode,'ended_after_completed_output_salvage':completed_output})
    if proc.returncode and not completed_output:
        raise RuntimeError(f'{name} failed ({proc.returncode}). {log.read_text(errors="replace")[-1600:]}')


def tangent_patch(site, local, dest, step):
    # Four seed-plane corners, all <=6 voxels from the common seed. Neither mesh nor
    # truth samples outside the permitted local tangent estimate are consulted.
    u,v = (np.array(site[k],float) for k in ['tangent_u_xyz','tangent_v_xyz'])
    xyz = np.full((8,8,3),-1.0,dtype='f4')
    for j,i in np.ndindex(2,2):
        xyz[j+3,i+3] = local + (i-0.5)*step*u + (j-0.5)*step*v
    dest.mkdir(exist_ok=True)
    save(dest/'meta.json',{'uuid':dest.name,'type':'seg','name':dest.name,'format':'tifxyz','scale':[1/step,1/step]})
    for i,n in enumerate('xyz'):
        tifffile.imwrite(dest/(n+'.tif'),xyz[:,:,i])


def run(site, volume_path, variant, output, pitch, frame, diagnostics, threshold, radius, generation_limit=None):
    id_ = site['id']
    shared = output/'volumes'/id_
    shared.mkdir(exist_ok=True,parents=True)
    cube = np.load(volume_path,mmap_mode='r')
    vol = shared/'m7.zarr'
    source_hash = digest(volume_path)
    with FileLock(str(shared/'volume.lock')):
        write_zarr(cube,vol,pitch,source_hash)
    tuning = '' if threshold==1 else f'_th{threshold}'
    label=variant+tuning+(f'_g{generation_limit}' if generation_limit is not None else '')
    base = output/(label+('_loss' if diagnostics else ''))/id_
    base.mkdir(exist_ok=True,parents=True)
    if (base/'record.json').exists() and (base/'surface.npz').exists():
        record=json.loads((base/'record.json').read_text())
        if record['input_sha256']!=source_hash or record['site']!=site or digest(base/'surface.npz')!=record['output_sha256']:
            raise ValueError(f'Stale baseline receipt: {base}')
        return
    step = 20 if variant=='default20' else 8
    cache_key = f'grid{step}{tuning}'
    grid = shared/cache_key
    gen = BIN/'vc_gen_normalgrids.exe'
    grow = BIN/'vc_grow_seg_from_seed.exe'
    with FileLock(str(shared/(cache_key+'.lock'))):
        finished=shared/(cache_key+'.ok')
        if finished.exists() and finished.read_text()!=source_hash:
            raise ValueError(f'Grid source mismatch at {finished}')
        if not finished.exists():
            command([gen,'-i',vol,'-o',grid,'--spiral-step',step,'--threshold',threshold,'--preview-every',0,
                     '--io-threads',2],shared,f'normal{cache_key}',2)
            finished.write_text(source_hash)
    local = np.asarray(site['seed_xyz']) - np.asarray(site['origin_xyz'])
    segment = base/'surface'
    segment.mkdir(exist_ok=True)
    # Constant growth allowance relative to the evaluation radius, frozen geometry only.
    config = {'mode':'seed','voxelsize':pitch,'step_size':step,'generations':generation_limit or int(np.ceil((5/3)*radius/step))+3,
              'normal_grid_path':str(grid),'min_area_cm':0,'thread_limit':1}
    if diagnostics:
        config['vis_losses'] = True
    cmd = [grow,'--volume',vol,'--target-dir',segment,'--params',base/'params.json',
           '--segment-name','surface','--rng-seed','42']
    if variant=='oriented8':
        tangent_patch(site,local,base/'local_tangent',step)
        config['mode'] = 'resume'
        cmd.extend(['--resume',base/'local_tangent','--resume-opt','skip'])
    else:
        cmd.extend(['--seed',*[f'{v:.8f}' for v in local]])
    save(base/'params.json',config)
    # Native TIFF saves use nested temporary paths that can exceed MAX_PATH on
    # Windows despite the final filenames fitting. Stage only output I/O and
    # preserve the native bytes plus exact invocation alongside the permanent copy.
    if os.name=='nt' and len(str(segment))>146:
        with tempfile.TemporaryDirectory(prefix='vcgp_') as temporary:
            staged=Path(temporary)/'surface'
            staged.mkdir()
            cmd[cmd.index('--target-dir')+1]=staged
            command(cmd,base,'grow',1,result_dir=staged)
            shutil.copytree(staged,segment,dirs_exist_ok=True)
    else:
        command(cmd,base,'grow',1)
    files = [segment/(n+'.tif') for n in 'xyz']
    exists = all(f.is_file() for f in files)
    if exists:
        points = np.stack([np.array(Image.open(f)) for f in files],-1).astype('f4')
        valid = np.isfinite(points).all(-1) & (points >= 0).all(-1) & (points < cube.shape[::-1]).all(-1)
        for name in ('mask','generations'):
            channel = segment/(name+'.tif')
            if channel.exists():
                sample = np.asarray(Image.open(channel))
                if sample.shape!=valid.shape:
                    raise ValueError(f'Misaligned {name} channel')
                valid &= sample>0
    else:
        points = np.full((2,2,3),np.nan,dtype='f4')
        valid = np.zeros((2,2),bool)
    points += np.asarray(site['origin_xyz'],dtype='f4')
    points[~valid] = np.nan
    extras={}
    if diagnostics and exists:
        for name in ('loss_dist','loss_straight','loss_normal','loss_snap','loss_sdir'):
            extras[name]=np.array(Image.open(segment/(name+'.tif')),dtype='f4')
            extras[name][~valid]=np.nan
        extras['quality']=-extras['loss_snap']
    np.savez_compressed(base/'surface.npz',xyz=points,valid=valid,**extras)
    save(base/'record.json',{'id':id_,'split':site['split'],'variant':variant+tuning,'site':site,'grid_threshold':threshold,
                             'input_sha256':source_hash,'xyz_units':frame,'physical_voxel_um':pitch,
                             'returned_tifxyz':exists,'valid_vertices':int(valid.sum()),
                             'source_commit':'e87544118ed046811635f62a47d04539e2a92c12',
                             'binary_sha256':digest(grow),'grid_binary_sha256':digest(gen),
                             'tifxyz_sha256':{f.name:digest(f) for f in files if f.exists()},
                             'output_sha256':digest(base/'surface.npz')})
    print(id_,variant,int(valid.sum()),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=SEEDS)
    p.add_argument('--inputs',type=Path)
    p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--pitch-um',type=float,default=9.6)
    p.add_argument('--ids',nargs='+')
    p.add_argument('--split',nargs='+')
    p.add_argument('--variant',choices=['default20','oriented8','point8'],required=True)
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--diagnostics',action='store_true',help='Save native loss channels without changing the optimizer or source arm')
    p.add_argument('--grid-threshold',type=int,choices=range(1,256),metavar='1..255',default=1)
    p.add_argument('--generations',type=int,help='Separate replication sensitivity with an explicit growth ceiling; preserves original outputs.')
    args = p.parse_args()
    install_release()
    manifest = json.loads(args.manifest.read_text())
    sites = manifest['sites']
    home = args.inputs or args.manifest.parent/'inputs'
    selected = [s for s in sites if (not args.ids or s['id'] in args.ids) and
                (not args.split or s['split'] in args.split)]
    if not selected: raise ValueError('No selected sites.')
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(run,s,home/s['id']/'m7.npy',args.variant,args.output.resolve(),
                               args.pitch_um,manifest.get('frame','global cube source XYZ'),args.diagnostics,
                               args.grid_threshold,manifest.get('area_radius_voxels',96),args.generations) for s in selected]
        for f in futures: f.result()


if __name__=='__main__': main()
