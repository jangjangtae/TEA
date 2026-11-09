"""Guard publication layout, condition inclusion, and caption escaping."""

import unittest

from dreamerv3.plot_fate_completed_results import METHODS
from dreamerv3.plot_fate_tog_figures import (
    SINGLE, DOUBLE, STYLE, PaperFigures, canvas, dot, paired_rows, setup_style,
    short, tex_escape, validate_layout, plt)


class PaperFiguresTest(unittest.TestCase):
    def setUp(self):
        setup_style()

    def tearDown(self):
        plt.close('all')

    def test_all_thirteen_conditions(self):
        rows = paired_rows()
        self.assertEqual({r[0] for r in rows}, set(METHODS))
        self.assertEqual(len(rows), 13)
        self.assertEqual(len({r[1] for r in rows}), 13)

    def test_display_name_does_not_change_historical_ids(self):
        for name in ('FATE', 'CleanAdaptFATE', 'FaultAdaptFATE'):
            self.assertEqual(short(name), 'TEA')
        self.assertIn('FaultAdaptFATE', {r[0] for r in paired_rows()})
        self.assertNotIn('FaultAdaptTEA', {r[0] for r in paired_rows()})

    def test_legacy_visible_label_rejected_before_saving(self):
        fig, _ = canvas()
        fig.text(.5, .5, 'FATE + R0', fontsize=9)
        writer = object.__new__(PaperFigures)
        with self.assertRaisesRegex(ValueError, 'Legacy method label in figure'):
            writer.save_paper(fig, 'test', '', 'TEA', '', [], '')

    def test_legacy_caption_rejected_before_saving(self):
        fig, _ = canvas()
        writer = object.__new__(PaperFigures)
        with self.assertRaisesRegex(ValueError, 'Legacy method label in caption'):
            writer.save_paper(fig, 'test', '', 'FATE bonus', '', [], '')

    def test_ablation_has_both_profiles(self):
        rows = paired_rows(False, ['FATE', 'AbsoluteSameShape'])
        self.assertEqual(len(rows), 4)
        self.assertNotIn('CleanEval', [r[0] for r in rows])

    def test_native_widths(self):
        for width in (SINGLE, DOUBLE):
            fig, axes = canvas(width=width)
            self.assertAlmostEqual(fig.get_figwidth(), width)
            self.assertEqual(axes[0].xaxis.label.get_fontsize(), 9)
            self.assertGreater(validate_layout(fig), 0)

    def test_small_font_rejected(self):
        fig, _ = canvas()
        fig.text(.5, .5, 'too small', fontsize=7)
        with self.assertRaisesRegex(ValueError, 'smaller than 9'):
            validate_layout(fig)

    def test_clipping_rejected(self):
        fig, _ = canvas()
        fig.text(-.5, .5, 'outside')
        with self.assertRaisesRegex(ValueError, 'Clipped'):
            validate_layout(fig)

    def test_caption_tex(self):
        self.assertEqual(tex_escape('5% & 2_seeds'), r'5\% \& 2\_seeds')

    def test_hidden_data_rejected(self):
        _, axes = canvas()
        axes[0].set_xlim(0, 1)
        with self.assertRaisesRegex(ValueError, 'hide a data point'):
            dot(axes[0], 0, [0, 2], STYLE['Fault'])

    def test_hidden_errorbar_rejected(self):
        _, axes = canvas()
        axes[0].set_xlim(0, 1)
        with self.assertRaisesRegex(ValueError, 'hide a data point'):
            dot(axes[0], 0, [0, 1], STYLE['Fault'])


if __name__ == '__main__':
    unittest.main()
