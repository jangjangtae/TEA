"""Configuration validation must never start an agent in the default mode."""

from contextlib import redirect_stderr, redirect_stdout
import io
import unittest
from unittest.mock import patch

from tools.run_tea_config import ROOT, build_command, main


class PublicConfigurationTests(unittest.TestCase):
    def test_all_archived_configurations_round_trip(self):
        paths = list((ROOT / 'configs/tea').rglob('*.yaml'))
        self.assertEqual(len(paths), 86)
        for path in paths:
            with self.subTest(config=str(path.relative_to(ROOT))):
                self.assertEqual(build_command(path)[1:3], ['-m', 'dreamerv3.main'])

    def test_overrides_preserve_boolean_numeric_and_path_types(self):
        path = ROOT / 'configs/tea/FaultAdaptFATE/seed_0_train.yaml'
        command = build_command(path, ['seed=4', 'run.steps=2048',
            'fault.log_only=True', 'logdir=/tmp/tea path with spaces'])
        self.assertIn('--seed=4', command)
        self.assertIn('--run.steps=2048.0', command)
        self.assertIn('--fault.log_only=True', command)
        self.assertIn('--logdir=/tmp/tea path with spaces', command)

    def test_unknown_and_malformed_overrides_are_rejected(self):
        path = ROOT / 'configs/tea/P0_pretraining.yaml'
        with self.assertRaises(KeyError):
            build_command(path, ['not_a_parameter=1'])
        for override in ('seed', '=4', '--seed=4'):
            with self.subTest(value=override), self.assertRaises(ValueError):
                build_command(path, [override])

    def test_wrong_type_is_rejected(self):
        with self.assertRaises(TypeError):
            build_command(ROOT / 'configs/tea/P0_pretraining.yaml', ['seed=abc'])

    def test_default_cli_does_not_run_subprocess(self):
        with patch('tools.run_tea_config.subprocess.call') as call:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main([str(ROOT / 'configs/tea/P0_pretraining.yaml')]), 0)
            call.assert_not_called()

    def test_execute_is_explicit_and_preserves_exit_status(self):
        with patch('tools.run_tea_config.subprocess.call', return_value=7) as call:
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main([str(ROOT / 'configs/tea/P0_pretraining.yaml'),
                                       '--execute']), 7)
            self.assertEqual(call.call_args.kwargs['cwd'], ROOT)


if __name__ == '__main__':
    unittest.main()
