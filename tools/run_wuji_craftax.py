#!/usr/bin/env python3
"""Run the isolated, GPU WUJI Craftax port without changing Dreamer packages."""

import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
for path in reversed([
    ROOT, ROOT / '.deps/wuji_upstream', ROOT / '.deps/torch311cu126',
    ROOT / '.deps/wuji_probe_pkgs', ROOT / '.deps/craftax_pkgs',
]):
    sys.path.insert(0, str(path))
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

if __name__ == '__main__':
    from dreamerv3.wuji_craftax import main
    main()
