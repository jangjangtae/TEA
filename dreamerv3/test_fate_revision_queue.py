"""Exercise real shell queue functions without launching a learning process."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class QueueProtocol(unittest.TestCase):

  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.path = Path(self.temp.name)
    self.repo = Path(__file__).resolve().parents[1]
    script = self.repo / 'dreamerv3/run_craftax_multiseed_fault_queue.sh'
    self.source = script.read_text()
    self.library = self.source.split('\nvalidate_milestones\n', 1)[0]
    self.clean = self.path / 'clean/train/ckpt/20260101'
    self.clean.mkdir(parents=True)
    (self.clean / 'done').touch()
    # Command-capture tests must not inherit a production queue's audit,
    # calibration, budget, or fault switches. Tests enable those explicitly.
    inherited = {key: value for key, value in os.environ.items() if key in (
        'PATH', 'HOME', 'TMPDIR', 'LD_LIBRARY_PATH', 'PYTHONPATH',
        'CUDA_VISIBLE_DEVICES', 'JAX_PLATFORMS', 'XLA_FLAGS',
        'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}
    self.env = dict(inherited, PROJECT_ROOT=str(self.repo),
        TRAIN_ROOT=str(self.path / 'clean'), ROOT=str(self.path / 'queue'),
        PYTHON_BIN=sys.executable, MIN_FREE_GB='0', VARIANTS='taskonly',
        DENSE_MATCHED_BETA='0.001', RND_MATCHED_ALPHA='0.002',
        TASK_REWARD_SOURCE='delivered', RESTART_INCOMPLETE_FROM_SCRATCH='1',
        AUDIT_REVISION_JOBS='0', CALIBRATION_ROOT='')
    self.capture = self.path / 'capture.py'
    self.capture.write_text(
        'import os, sys, json\n'
        'with open(os.environ["CAPTURE"], "a") as f:\n'
        ' f.write(json.dumps(dict(argv=sys.argv[1:], env={k:v for k,v in os.environ.items() '
        'if k.startswith("CRAFTAX_")})) + "\\n")\n')
    self.env['CAPTURE'] = str(self.path / 'calls.jsonl')

  def run_shell(self, body, success=True):
    path = self.path / 'test.sh'
    path.write_text(self.library + '\n' + body)
    result = subprocess.run(['bash', str(path)], env=self.env,
                            capture_output=True, text=True, timeout=20)
    if success:
      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    else:
      self.assertNotEqual(result.returncode, 0)
    return result

  def test_only_completed_checkpoints_selected(self):
    partial = self.clean.parent / '20260102'
    partial.mkdir()
    (partial.parent / 'latest').write_text(partial.name)
    self.run_shell(f'latest_ckpt "$TRAIN_ROOT/train" > "{self.path}/selected"')
    self.assertEqual((self.path / 'selected').read_text().strip(), str(self.clean))

  def test_completed_jobs_are_not_executed_twice(self):
    self.run_shell(f'''count() {{ echo x >> "{self.path}/count"; }}
run_job 0 test count
run_job 0 test count
''')
    self.assertEqual((self.path / 'count').read_text(), 'x\n')

  def test_failed_job_stops_followup(self):
    self.run_shell(f'run_job 0 broken false\ntouch "{self.path}/must_not_exist"', False)
    self.assertFalse((self.path / 'must_not_exist').exists())
    self.assertIn('FAILED:1', (self.path / 'queue/status.tsv').read_text())

  def test_interrupted_attempt_is_archived_not_partially_resumed(self):
    output = self.path / 'unfinished'
    output.mkdir()
    (output / 'marker').write_text('preserve')
    self.run_shell(f'prepare_attempt "{output}"')
    archived = list(self.path.glob('unfinished.interrupted_*'))
    self.assertEqual(len(archived), 1)
    self.assertEqual((archived[0] / 'marker').read_text(), 'preserve')
    self.assertFalse(output.exists())

  def test_partial_calibration_trace_is_not_a_completion_marker(self):
    block = self.source.split('  STATS_FILE="$seed_root/', 1)[1]
    block = '  STATS_FILE="$seed_root/' + block.split('\n  if [[ "$RUN_VARIANTS"', 1)[0]
    self.run_shell('''seed=0
seed_root="$ROOT/seed_0"
mkdir -p "$seed_root/base_clean_eval"
echo partial > "$seed_root/base_clean_eval/fault_score_trace.jsonl"
run_eval() { echo complete > "$seed_root/base_clean_eval/fault_score_trace.jsonl"; }
make_norm_stats() {
  [[ "$(cat "$1")" == complete ]] || return 1
  echo '{}' > "$2"
}
''' + block)
    status = (self.path / 'queue/status.tsv').read_text()
    self.assertIn('base_clean_eval_for_stats\tDONE', status)
    self.assertIn('make_norm_stats\tDONE', status)

  def test_calibration_and_six_weight_commands_are_consistent(self):
    self.run_shell(f'''MAIN_PY="{self.capture}"
run_eval 0 base_clean_eval "$REF_CKPT" 256 "" "" ""
for variant in dense_matched_half dense_matched dense_matched_double rnd_matched_half rnd_matched rnd_matched_double; do
  run_train_variant 0 "$variant" stats.json 4096
done
''')
    records = [json.loads(line) for line in Path(self.env['CAPTURE']).read_text().splitlines()]
    self.assertEqual(len(records), 7)
    for record in records:
      args = record['argv']
      self.assertEqual(args[args.index('--fault.w_reward') + 1], '0.0')
      self.assertEqual(args[args.index('--fault.task_reward_source') + 1], 'delivered')
      self.assertEqual(args[args.index('--fault.clip') + 1], '2.0')
    for record, expected in zip(records[1:4], (0.0005, 0.001, 0.002)):
      args = record['argv']
      self.assertAlmostEqual(float(args[args.index('--fault.beta') + 1]), expected)
      self.assertEqual(args[args.index('--fault.log_only') + 1], 'False')
      self.assertEqual(record['env']['CRAFTAX_USE_RND'], '0')
    for record, expected in zip(records[4:], (0.001, 0.002, 0.004)):
      args = record['argv']
      self.assertAlmostEqual(float(record['env']['CRAFTAX_RND_ALPHA']), expected)
      self.assertEqual(record['env']['CRAFTAX_USE_RND'], '1')
      self.assertEqual(args[args.index('--fault.log_only') + 1], 'True')

  def test_matched_evaluation_budgets_and_fate_diagnostics(self):
    self.env.update(BASE_EVAL_STEPS='30000', BASE_FAULT_EVAL_STEPS='60000',
                    ADAPT_CLEAN_EVAL_STEPS='30000', ADAPT_EVAL_STEPS='60000',
                    SPARSE_EVAL_STEPS='60000', EVAL_SPLITS='clean seen holdout sparse')
    self.run_shell(f'''MAIN_PY="{self.capture}"
run_variant_evals 0 excess_delta_p95_beta02 "$REF_CKPT" stats.json
''')
    records = [json.loads(x) for x in Path(self.env['CAPTURE']).read_text().splitlines()]
    self.assertEqual(len(records), 4)
    for record, steps in zip(records, (30000, 60000, 60000, 60000)):
      args = record['argv']
      self.assertEqual(int(args[args.index('--run.steps') + 1]), steps)
      self.assertEqual(args[args.index('--fault.reward_mode') + 1], 'excess_delta_threshold')
      self.assertEqual(args[args.index('--fault.log_only') + 1], 'True')
      self.assertEqual(record['env']['CRAFTAX_USE_RND'], '0')
    self.assertEqual(records[0]['env']['CRAFTAX_FAULT'], '0')
    for record in records[1:]:
      self.assertEqual(record['env']['CRAFTAX_FAULT_EP_PROB'], '0.5')
      self.assertEqual(record['env']['CRAFTAX_FAULT_MANIFEST_PROB'], '1.0')
    base = self.source.split('  if [[ "$RUN_BASE_EVALS" == "1" ]]; then', 1)[1].split('\n  STATS_FILE=', 1)[0]
    self.run_shell(f'MAIN_PY="{self.capture}"\nseed=1\nif true; then\n' + base)
    records = [json.loads(x) for x in Path(self.env['CAPTURE']).read_text().splitlines()]
    self.assertEqual([int(r['argv'][r['argv'].index('--run.steps') + 1]) for r in records[4:]],
                     [30000, 60000, 60000, 60000])

  def test_failed_audit_does_not_mark_job_done(self):
    self.env['AUDIT_REVISION_JOBS'] = '1'
    self.env['PYTHON_BIN'] = '/bin/false'
    self.run_shell(f'run_job 0 test true\ntouch "{self.path}/after"', False)
    self.assertFalse((self.path / 'after').exists())
    status = (self.path / 'queue/status.tsv').read_text()
    self.assertIn('FAILED:1', status)
    self.assertNotIn('\tDONE', status)

  def test_clean_adaptation_and_faulted_evaluation_switches(self):
    self.env.update(TRAIN_FAULT_PROFILE='clean', EVAL_SPLITS='clean seen holdout sparse')
    self.run_shell(f'''MAIN_PY="{self.capture}"
for variant in taskonly excess_delta_p95_beta02; do
  run_train_variant 0 "$variant" stats.json 4096
  run_variant_evals 0 "$variant" "$REF_CKPT" stats.json
done
''')
    records = [json.loads(x) for x in Path(self.env['CAPTURE']).read_text().splitlines()]
    self.assertEqual(len(records), 10)
    for offset, log_only in ((0, 'True'), (5, 'False')):
      train = records[offset]
      args = train['argv']
      self.assertEqual(args[args.index('--fault.log_only') + 1], log_only)
      for record in records[offset:offset + 2]:
        self.assertEqual(record['env']['CRAFTAX_FAULT'], '0')
        self.assertEqual(record['env']['CRAFTAX_FAULT_SAMPLER'], '0')
        self.assertNotIn('CRAFTAX_FAULT_PROFILE', record['env'])
        self.assertEqual(record['env']['CRAFTAX_USE_RND'], '0')
      for record, profile in zip(records[offset + 2:offset + 5],
                                ('benchmark_seen', 'benchmark_holdout', 'benchmark_sparse')):
        self.assertEqual(record['env']['CRAFTAX_FAULT_SAMPLER'], '1')
        self.assertEqual(record['env']['CRAFTAX_FAULT_PROFILE'], profile)
        self.assertEqual(record['env']['CRAFTAX_FAULT_EP_PROB'], '0.5')

  def test_invalid_calibration_fails_closed(self):
    for content in ('', '{"fault_score_raw": NaN}\n'):
      trace = self.path / 'bad.jsonl'
      trace.write_text(content)
      self.run_shell(f'make_norm_stats "{trace}" "{self.path}/stats.json"', False)
      self.assertFalse((self.path / 'stats.json').exists())

  def test_shared_calibration_requires_audit_and_is_copied_exactly(self):
    source = self.path / 'shared'
    (source / 'seed_0').mkdir(parents=True)
    payload = '{"fault_score_p95": 123.456}\n'
    (source / 'seed_0/craftax_clean_fault_stats.json').write_text(payload)
    self.env['CALIBRATION_ROOT'] = str(source)
    command = f'import_norm_stats 0 "{self.path}/stats.json"'
    self.run_shell(command, False)
    self.assertFalse((self.path / 'stats.json').exists())
    (source / 'job_audits').mkdir()
    (source / 'job_audits/seed_0_make_norm_stats.json').write_text('{}')
    self.run_shell(command)
    self.assertEqual((self.path / 'stats.json').read_text(), payload)


if __name__ == '__main__':
  unittest.main()
