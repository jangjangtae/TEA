"""Check public instructions without executing installation or training commands."""

import ast
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / 'README.md', *(ROOT / 'docs').glob('tea_*.md'),
        ROOT / 'docs/wuji_craftax_port.md']


class PublicDocumentationTests(unittest.TestCase):
    def test_shell_examples_are_syntactically_valid(self):
        count = 0
        for path in DOCS:
            for body in re.findall(r'^```bash\n(.*?)^```', path.read_text(), re.M | re.S):
                with self.subTest(document=path.name, block=count):
                    result = subprocess.run(['bash', '-n'], input=body, text=True,
                                            capture_output=True, check=False)
                    self.assertEqual(result.returncode, 0, result.stderr)
                count += 1
        self.assertGreaterEqual(count, 20)

    def test_embedded_replay_manifest_script_compiles_without_running(self):
        text = (ROOT / 'docs/tea_reproduction.md').read_text()
        blocks = re.findall(r"python - <<'PY'\n(.*?)^PY$", text, re.M | re.S)
        self.assertEqual(len(blocks), 1)
        ast.parse(blocks[0])

    def test_relative_document_links_exist(self):
        for path in DOCS:
            for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
                if '://' in target or target.startswith('#'):
                    continue
                with self.subTest(document=path.name, target=target):
                    self.assertTrue((path.parent / target.split('#')[0]).exists())

    def test_released_provenance_hashes_match(self):
        manifest = json.loads((ROOT / 'results/tea/SOURCE_MANIFEST.json').read_text())
        for row in manifest['files']:
            with self.subTest(file=row['target']):
                actual = hashlib.sha256((ROOT / row['target']).read_bytes()).hexdigest()
                self.assertEqual(actual, row['public_sha256'])

    def test_rnd_environment_covers_both_training_cohorts(self):
        with (ROOT / 'configs/tea/rnd_environment.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 10)
        expected = {(method, str(seed)) for method in ('CleanAdaptRND', 'FaultAdaptRND')
                    for seed in range(5)}
        self.assertEqual({(r['method'], r['adapt_seed']) for r in rows}, expected)
        for row in rows:
            self.assertEqual(row['adapt_seed'], row['CRAFTAX_RND_SEED'])
            self.assertGreater(float(row['CRAFTAX_RND_ALPHA']), 0)
        fault = [r['CRAFTAX_RND_ALPHA'] for r in rows if r['method'] == 'FaultAdaptRND']
        clean = [r['CRAFTAX_RND_ALPHA'] for r in rows if r['method'] == 'CleanAdaptRND']
        self.assertEqual(len(set(fault)), 1)
        self.assertEqual(len(set(clean)), 5)


if __name__ == '__main__':
    unittest.main()
