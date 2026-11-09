#!/usr/bin/env python3
"""Validate an archived TEA YAML and print its command; opt in with --execute."""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

import elements
from ruamel.yaml import YAML


ROOT = Path(__file__).resolve().parents[1]


def build_command(path, overrides=()):
    yaml = YAML(typ='safe')
    with Path(path).open() as stream:
        saved = elements.Config(yaml.load(stream))
    flags = []
    for item in overrides:
        key, separator, value = item.partition('=')
        if not separator or not key or key.startswith('--'):
            raise ValueError('Use --set dotted.key=value for each override.')
        if key not in saved.flat:
            raise KeyError(f'Unknown archived configuration field: {key}')
        flags.append(f'--{key}={value}')
    configured = elements.Flags(saved).parse(flags)
    argv = ['--configs', 'defaults']
    for key, value in configured.flat.items():
        if isinstance(value, tuple):
            if not value:
                raise ValueError(f'Cannot encode an empty tuple: {key}')
            argv.extend([f'--{key}', *(str(item) for item in value)])
        else:
            argv.append(f'--{key}={value}')
    with (ROOT / 'dreamerv3/configs.yaml').open() as stream:
        schema = elements.Config(yaml.load(stream)['defaults'])
    checked = elements.Flags(schema).parse(argv[2:])
    if checked.flat != configured.flat:
        raise ValueError('Archived configuration did not round-trip through CLI flags.')
    return [sys.executable, '-m', 'dreamerv3.main', *argv]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    parser.add_argument('--execute', action='store_true',
                        help='Run the command instead of only printing it.')
    args = parser.parse_args(argv)
    command = build_command(args.config, args.set)
    print(shlex.join(command), flush=True)
    if args.execute:
        return subprocess.call(command, cwd=ROOT)
    print('Dry run only. Environment variables are inherited, not inferred from YAML.',
          file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
