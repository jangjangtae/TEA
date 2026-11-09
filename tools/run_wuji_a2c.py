#!/usr/bin/env python3
"""Isolated SB3 A2C launcher; never changes the Dreamer environment."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in reversed([
    ROOT, ROOT / 'tools', ROOT / 'third_party', ROOT / '.deps/wuji_upstream',
    ROOT / '.deps/torch311cu126', ROOT / '.deps/wuji_probe_pkgs',
    ROOT / '.deps/craftax_pkgs', ROOT / '.deps/meta_rl_pkgs',
]):
    sys.path.insert(0, str(path))
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

if __name__ == '__main__':
    from dreamerv3.wuji_a2c_craftax import main
    main()
