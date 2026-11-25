"""Round 3: on-demand chunked access to a public Zarr v2 volume with an in-memory LRU of decoded chunks.

Chunks are persisted in data/cache under the same sha256(url) names as zarr_crop.py. Absent chunks decode to the
array's fill value only when allow_missing=True (for sparse prediction arrays); raw CT callers keep the default.
"""
import collections
import hashlib
import json
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numcodecs
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'data/cache'


def _get(url, timeout=120):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


class ChunkVolume:
    def __init__(self, url, max_bytes=6 * 1024 ** 3, allow_missing=False, workers=12):
        self.url = url.rstrip('/')
        self.meta = json.loads(_get(self.url + '/.zarray'))
        self.shape = np.array(self.meta['shape']); self.chunk = np.array(self.meta['chunks'])
        self.dtype = np.dtype(self.meta['dtype']); self.sep = self.meta.get('dimension_separator', '.')
        self.codec = numcodecs.get_codec(self.meta['compressor']) if self.meta.get('compressor') else None
        self.fill = self.meta.get('fill_value') or 0
        self.lru = collections.OrderedDict(); self.bytes = 0; self.max_bytes = max_bytes
        self.allow_missing = allow_missing; self.lock = threading.Lock(); self.pool = ThreadPoolExecutor(workers)
        self.downloaded = 0; self.missing = set()

    def _load(self, key):
        address = self.url + '/' + self.sep.join(map(str, key))
        path = CACHE / hashlib.sha256(address.encode()).hexdigest()
        if path.exists():
            data = path.read_bytes()
        else:
            try:
                data = _get(address)
            except urllib.error.HTTPError as e:
                if self.allow_missing and e.code in (403, 404):
                    self.missing.add(key)
                    return np.full(tuple(self.chunk), self.fill, self.dtype)
                raise
            tmp = path.with_suffix('.tmp' + str(threading.get_ident()))
            tmp.write_bytes(data); tmp.replace(path)
            self.downloaded += len(data)
        dec = self.codec.decode(data) if self.codec else data
        return np.frombuffer(dec, self.dtype).reshape(tuple(self.chunk))

    def chunks(self, keys):
        need = [k for k in keys if k not in self.lru]
        if need:
            for k, arr in zip(need, self.pool.map(self._load, need)):
                self.lru[k] = arr; self.bytes += arr.nbytes
        for k in keys:
            self.lru.move_to_end(k)
        while self.bytes > self.max_bytes and len(self.lru) > len(keys):
            _, a = self.lru.popitem(last=False); self.bytes -= a.nbytes
        return {k: self.lru[k] for k in keys}

    def gather(self, iz, iy, ix):
        """Integer voxel coordinates (any shape) -> values; outside the array -> fill."""
        shp = iz.shape
        iz, iy, ix = iz.ravel(), iy.ravel(), ix.ravel()
        out = np.full(iz.shape, self.fill, self.dtype)
        inside = (iz >= 0) & (iy >= 0) & (ix >= 0) & (iz < self.shape[0]) & (iy < self.shape[1]) & (ix < self.shape[2])
        if inside.any():
            cz, cy, cx = iz[inside] // self.chunk[0], iy[inside] // self.chunk[1], ix[inside] // self.chunk[2]
            cid = (cz.astype(np.int64) * 100000 + cy) * 100000 + cx
            uniq, inv = np.unique(cid, return_inverse=True)
            keys = [(int(u // 10 ** 10), int((u // 100000) % 100000), int(u % 100000)) for u in uniq]
            arrs = self.chunks(keys)
            vals = np.empty(inside.sum(), self.dtype)
            lz, ly, lx = iz[inside] % self.chunk[0], iy[inside] % self.chunk[1], ix[inside] % self.chunk[2]
            order = np.argsort(inv, kind='stable'); bounds = np.searchsorted(inv[order], np.arange(len(uniq) + 1))
            for j, k in enumerate(keys):
                sel = order[bounds[j]:bounds[j + 1]]
                vals[sel] = arrs[k][lz[sel], ly[sel], lx[sel]]
            out[inside] = vals
        return out.reshape(shp)

    def trilinear(self, z, y, x):
        """Float voxel coordinates (voxel centres at integers) -> trilinear values (float32)."""
        z0, y0, x0 = np.floor(z).astype(np.int64), np.floor(y).astype(np.int64), np.floor(x).astype(np.int64)
        fz, fy, fx = (z - z0).astype(np.float32), (y - y0).astype(np.float32), (x - x0).astype(np.float32)
        acc = np.zeros(z.shape, np.float32)
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    w = (fz if dz else 1 - fz) * (fy if dy else 1 - fy) * (fx if dx else 1 - fx)
                    acc += w * self.gather(z0 + dz, y0 + dy, x0 + dx).astype(np.float32)
        return acc
