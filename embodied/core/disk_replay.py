"""Bounded-memory local SSD replay with streaming, immutable checkpoints.

Only metadata stays resident. pread/pwrite use the reclaimable OS page cache;
there is no anonymous full-buffer copy or long-lived mmap. Checkpoint shards
are owned and cleaned up by elements.Checkpoint, just like model shards.
"""

import atexit
from collections import defaultdict, deque
from contextlib import contextmanager
import fcntl
import functools
import hashlib
import os
from pathlib import Path
import shutil
import threading
import uuid

import elements
import numpy as np

from . import limiters, selectors
from .chunk import Chunk
from .replay import Replay


def _write(fd, data, offset):
  data = memoryview(data).cast('B')
  while data:
    written = os.pwrite(fd, data, offset)
    if written <= 0:
      raise OSError('Short SSD replay write')
    offset += written
    data = data[written:]


def _read(fd, size, offset):
  data = os.pread(fd, size, offset)
  if len(data) != size:
    raise OSError(f'Truncated SSD replay: expected {size}, got {len(data)}')
  return data


class DiskChunk(Chunk):

  __slots__ = ('path', 'fields', '_nbytes')

  def __init__(self, directory, size=1024):
    super().__init__(size)
    self.path = Path(directory) / (bytes(self.uuid).hex() + '.bin')
    self.fields = {}
    self._nbytes = 0

  @property
  def nbytes(self):
    return self._nbytes

  def append(self, step):
    assert self.length < self.size
    if not self.fields:
      offset = 0
      for key, value in step.items():
        if value.dtype.hasobject:
          raise TypeError('Object arrays are not supported in SSD replay')
        stride = value.nbytes
        offset = (offset + 63) // 64 * 64
        self.fields[key] = (value.dtype.str, value.shape, stride, offset)
        offset += self.size * stride
      self._nbytes = offset
      with self.path.open('xb') as handle:
        handle.truncate(offset)
    if set(step) != set(self.fields):
      raise ValueError('Replay transition keys changed within a chunk')
    with self._open() as fd:
      for key, value in step.items():
        self._put(fd, key, self.length, 1, np.asarray(value)[None])
    self.length += 1

  @contextmanager
  def _open(self):
    fd = os.open(self.path, os.O_RDWR)
    try:
      yield fd
    finally:
      os.close(fd)

  def _put(self, fd, key, index, length, value):
    dtype, shape, stride, offset = self.fields[key]
    value = np.ascontiguousarray(value, dtype=dtype)
    if value.shape != (length, *shape):
      raise ValueError(f'Replay shape mismatch for {key}: {value.shape}')
    _write(fd, value, offset + index * stride)

  def update(self, index, length, mapping):
    assert 0 <= index <= index + length <= self.length
    with self._open() as fd:
      for key, value in mapping.items():
        self._put(fd, key, index, length, value)

  def slice(self, index, length, keys=None):
    assert 0 <= index <= index + length <= self.length
    result = {}
    with self._open() as fd:
      for key in self.fields if keys is None else keys:
        dtype, shape, stride, offset = self.fields[key]
        data = _read(fd, length * stride, offset + index * stride)
        result[key] = np.frombuffer(data, dtype).reshape((length, *shape))
    return result

  def record(self):
    return dict(time=self.time, uuid=bytes(self.uuid), succ=bytes(self.succ),
                length=self.length, size=self.size, fields=self.fields,
                nbytes=self.nbytes)

  @classmethod
  def restore(cls, directory, record, data):
    obj = cls(directory, record['size'])
    obj.time = record['time']
    obj.uuid = elements.UUID(record['uuid'])
    obj.succ = elements.UUID(record['succ'])
    obj.length = record['length']
    obj.fields = record['fields']
    obj._nbytes = record['nbytes']
    obj.path = Path(directory) / (bytes(obj.uuid).hex() + '.bin')
    if len(data) != obj.nbytes:
      raise ValueError('SSD checkpoint chunk length mismatch')
    if obj.fields:
      with obj.path.open('xb') as handle:
        handle.write(data)
    return obj


def _locked(fn):
  @functools.wraps(fn)
  def wrapped(self, *args, **kwargs):
    with self._lock:
      return fn(self, *args, **kwargs)
  return wrapped


class DiskReplay(Replay):

  FORMAT = 'ssd-replay-v1'

  def __init__(self, *args, directory=None, selector=None, **kwargs):
    if directory is None:
      raise ValueError('SSD replay requires a local directory')
    if selector is not None and type(selector) is not selectors.Uniform:
      raise ValueError('SSD replay currently supports Uniform + online only')
    self._lock = threading.RLock()
    self._closed = False
    root = Path(str(directory)).resolve()
    root.mkdir(parents=True, exist_ok=True)
    self._lease = (root / 'ssd.lock').open('a+b')
    try:
      fcntl.flock(self._lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
      self._lease.close()
      raise RuntimeError(f'SSD replay already in use: {root}') from None
    # Scratch is never a checkpoint. A new process loads committed shards.
    workroot = root / 'ssd_work'
    workroot.mkdir(exist_ok=True)
    for old in workroot.glob('session-*'):
      if old.is_dir() and not old.is_symlink():
        shutil.rmtree(old)
    self._workdir = workroot / ('session-' + uuid.uuid4().hex)
    self._workdir.mkdir()
    super().__init__(*args, directory=None, selector=selector, **kwargs)
    self.directory = elements.Path(str(root))
    self.counts = dict(transitions=0, resets=0, actual_actions=0, context_update_calls=0)
    atexit.register(self.close)

  def _new_chunk(self):
    return DiskChunk(self._workdir, self.chunksize)

  @_locked
  def add(self, step, worker=0):
    super().add(step, worker)
    reset = int(bool(step.get('is_first', False)))
    self.counts['transitions'] += 1
    self.counts['resets'] += reset
    self.counts['actual_actions'] += 1 - reset

  def sample(self, batch, mode='train'):
    limiters.wait(lambda: len(self.sampler), f'Replay {self.name} is empty')
    with self._lock:
      return super().sample(batch, mode)

  @_locked
  def update(self, data):
    super().update(data)
    self.counts['context_update_calls'] += 1

  def _remove(self):
    chunkid, _ = self.items[self.fifo[0]]
    chunk = self.chunks[chunkid]
    super()._remove()
    if chunkid not in self.chunks:
      chunk.path.unlink(missing_ok=True)

  @_locked
  def stats(self):
    stats = super().stats()
    stats['disk_gb'] = stats.pop('ram_gb')
    # Resident metadata and sample batches are included in process RSS, not here.
    stats['ram_gb'] = 0.0
    stats.update(self.counts)
    return stats

  def save(self):
    # Lock spans generator consumption, including all context bytes. Shards
    # belong to the enclosing atomic checkpoint, not mutable working files.
    with self._lock:
      pair = lambda value: (bytes(value[0]), value[1])
      metadata = dict(
          format=self.FORMAT, length=self.length, capacity=self.capacity,
          chunksize=self.chunksize, online=self.online, itemid=self.itemid,
          items={k: pair(v) for k, v in self.items.items()}, fifo=list(self.fifo),
          refs={bytes(k): v for k, v in self.refs.items()},
          current={k: pair(v) for k, v in self.current.items()},
          streams={k: [pair(v) for v in vs] for k, vs in self.streams.items()},
          sampler_keys=list(self.sampler.keys), rng=self.sampler.rng.bit_generator.state,
          queue=[pair(v) for v in self.queue] if self.online else [],
          lengths=dict(self.lengths) if self.online else {},
          chunks=len(self.chunks), metrics=self.metrics.copy(), counts=self.counts.copy())
      yield metadata
      for chunk in self.chunks.values():
        data = chunk.path.read_bytes() if chunk.fields else b''
        yield dict(record=chunk.record(), data=data,
                   sha256=hashlib.sha256(data).hexdigest())

  @_locked
  def load(self, data=None, directory=None, amount=None):
    if data is None or directory is not None or amount is not None:
      raise ValueError('SSD replay must load its complete sharded checkpoint')
    if self.chunks or self.items:
      raise ValueError('Load SSD replay into an empty instance')
    shards = iter(data)
    metadata = next(shards)
    expected = dict(format=self.FORMAT, length=self.length, capacity=self.capacity,
                    chunksize=self.chunksize, online=self.online)
    for key, value in expected.items():
      if metadata[key] != value:
        raise ValueError(f'SSD checkpoint configuration mismatch: {key}')
    chunks = {}
    try:
      for shard in shards:
        if hashlib.sha256(shard['data']).hexdigest() != shard['sha256']:
          raise ValueError('SSD checkpoint chunk checksum mismatch')
        chunk = DiskChunk.restore(self._workdir, shard['record'], shard['data'])
        if chunk.uuid in chunks:
          raise ValueError('Duplicate SSD checkpoint chunk')
        chunks[chunk.uuid] = chunk
      if len(chunks) != metadata['chunks']:
        raise ValueError('Missing SSD checkpoint shards')
    except Exception:
      for path in self._workdir.glob('*.bin'):
        path.unlink()
      raise
    pair = lambda value: (elements.UUID(value[0]), value[1])
    self.chunks = chunks
    self.refs = {elements.UUID(k): v for k, v in metadata['refs'].items()}
    self.items = {k: pair(v) for k, v in metadata['items'].items()}
    self.fifo = deque(metadata['fifo'])
    self.itemid = metadata['itemid']
    self.current = {k: pair(v) for k, v in metadata['current'].items()}
    self.streams = defaultdict(deque, {
        k: deque(pair(v) for v in vs) for k, vs in metadata['streams'].items()})
    self.sampler.keys = metadata['sampler_keys']
    self.sampler.indices = {key: i for i, key in enumerate(self.sampler.keys)}
    self.sampler.rng.bit_generator.state = metadata['rng']
    self.metrics = metadata['metrics']
    self.counts = metadata['counts']
    if self.online:
      self.queue = deque(pair(v) for v in metadata['queue'])
      self.lengths = defaultdict(int, metadata['lengths'])

  @_locked
  def close(self):
    if not self._closed:
      self._closed = True
      shutil.rmtree(self._workdir, ignore_errors=True)
      self._lease.close()
      atexit.unregister(self.close)
