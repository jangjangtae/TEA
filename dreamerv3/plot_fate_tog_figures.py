"""Typeset immutable historical results with TEA labels; no learning or rollout."""

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from dreamerv3.plot_fate_completed_results import METHODS, SUBTYPES, kind, policy_values
from dreamerv3.plot_fate_manuscript_materials import Figures, Inputs, core, plt
from dreamerv3.plot_fate_professor_discussion import one, require, seeded, select, stats
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.text import Text


REPO = Path(__file__).resolve().parents[1]
SINGLE, DOUBLE = 3.5, 7.16
BLUE, ORANGE, GRAY = '#0072B2', '#D55E00', '#555555'
STYLE = {'Fault': (BLUE, 'o', BLUE), 'Clean': (ORANGE, 's', 'white'),
         'Initial': (GRAY, 'D', GRAY)}
BASE = ['TaskOnly', 'RND', 'DenseSurprise', 'SparseSurprise', 'FATE', 'AbsoluteSameShape']
SHORT = {'CleanEval': 'Initial', 'DenseSurprise': 'Dense', 'SparseSurprise': 'Sparse',
         'AbsoluteSameShape': 'Abs. level', 'FATE': 'TEA'}
GUIDE = 'https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-graphics-for-your-article/resolution-and-size/'
TYPE_GUIDE = 'https://proceedingsoftheieee.ieee.org/resources/guidelines-for-figures-and-tables/'


def short(method):
    return SHORT.get(core(method), core(method))


def paired_rows(initial=True, methods=BASE):
    result = [('CleanEval', 0, 'Initial')] if initial else []
    for i, method in enumerate(methods, int(initial)):
        result.extend([(f'FaultAdapt{method}', i - .16, 'Fault'),
                       (f'CleanAdapt{method}', i + .16, 'Clean')])
    return result


def setup_style():
    plt.rcParams.update({
        'font.family': 'Liberation Serif', 'font.size': 9, 'mathtext.fontset': 'stix',
        'axes.titlesize': 9, 'axes.labelsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'legend.fontsize': 9, 'axes.labelcolor': 'black', 'text.color': 'black',
        'xtick.color': 'black', 'ytick.color': 'black', 'axes.edgecolor': '#555555',
        'axes.linewidth': .6, 'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.major.size': 2.5, 'ytick.major.size': 0, 'xtick.major.width': .6,
        'axes.labelpad': 3, 'xtick.major.pad': 2.5, 'ytick.major.pad': 4,
        'lines.linewidth': 1, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.fonttype': 'none', 'savefig.facecolor': 'white',
        'axes.formatter.useoffset': False, 'figure.dpi': 150,
    })


def canvas(cols=1, width=DOUBLE, height=3.25, left=1.0, right=.12, top=.42,
           bottom=.57, gap=.33):
    fig = plt.figure(figsize=(width, height), facecolor='white')
    aw = (width - left - right - gap * (cols - 1)) / cols
    axes = [fig.add_axes([(left + i * (aw + gap)) / width, bottom / height,
                         aw / width, (height - top - bottom) / height]) for i in range(cols)]
    return fig, axes


def row_axis(ax, labels, limits, ticks, xlabel, show=True):
    ax.set_ylim(len(labels) - .55, -.55)
    ax.set_yticks(range(len(labels)), labels if show else [])
    ax.set_xlim(*limits)
    ax.set_xticks(ticks)
    ax.set_xlabel(xlabel)
    ax.set_axisbelow(True)
    ax.grid(axis='x', color='#e3e3e3', linewidth=.5)
    ax.spines['left'].set_visible(False)


def key_handle(c, marker, face, label):
    return Line2D([], [], color=c, marker=marker, markerfacecolor=face,
                  markeredgewidth=.8, markersize=4.2, linewidth=0, label=label)


def legend(fig, entries=None):
    entries = entries or [(STYLE['Fault'], 'Fault adaptation'), (STYLE['Clean'], 'Clean adaptation')]
    fig.legend(handles=[key_handle(*style, label) for style, label in entries],
               loc='upper center', bbox_to_anchor=(.5, .995), ncol=len(entries),
               frameon=False, handletextpad=.4, columnspacing=1.2, borderaxespad=.2)


def dot(ax, y, values, style, individual=True, error=True):
    mean, sd = stats(values)
    lo, hi = sorted(ax.get_xlim())
    spread = sd if error and sd is not None else 0
    require(min(min(values), mean - spread) >= lo - 1e-9
            and max(max(values), mean + spread) <= hi + 1e-9,
            'Axis limits would hide a data point or error bar')
    c, marker, face = style
    if individual:
        ax.scatter(values, y + np.linspace(-.065, .065, len(values)),
                   color=c, s=4, alpha=.45, linewidths=0, zorder=2)
    ax.errorbar(mean, y, xerr=sd if error else None, fmt=marker, color=c,
                markerfacecolor=face, markeredgewidth=.8, ms=4.1, capsize=2,
                capthick=.65, elinewidth=.8, zorder=3)


def visible_text(fig):
    # Tick Text objects outside axis limits exist even when Matplotlib never draws them.
    outside = set()
    for ax in fig.axes:
        for axis, limits in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = sorted(limits)
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not lo <= tick.get_loc() <= hi:
                    outside.update((tick.label1, tick.label2))
    return [t for t in fig.findobj(match=Text)
            if t.get_visible() and t.get_text() and t not in outside]


def validate_layout(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = visible_text(fig)
    for text in texts:
        box = text.get_window_extent(renderer)
        require(box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.x1 + 1
                and box.y1 <= fig.bbox.y1 + 1, f'Clipped label: {text.get_text()}')
        require(text.get_fontsize() >= 8.99, f'Text smaller than 9 pt: {text.get_text()}')
    require(any(abs(fig.get_figwidth() - w) < 1e-9 for w in (SINGLE, DOUBLE)), 'Wrong column width')
    for ax in fig.axes:
        for labels in (ax.get_xticklabels(), ax.get_yticklabels()):
            boxes = [(t, t.get_window_extent(renderer)) for t in labels if t in texts]
            for i, (ta, a) in enumerate(boxes):
                for tb, b in boxes[i + 1:]:
                    require(not a.overlaps(b), f'Overlapping ticks: {ta.get_text()} / {tb.get_text()}')
    return len(texts)


class PaperFigures(Figures):
    """Reuse numeric provenance recording, but not the report-page layout."""

    def __init__(self, out):
        self.out = out
        self.points, self.summaries, self.catalog = [], [], []
        self.pdf = PdfPages(out / 'tea_tog_figures.pdf')
        setup_style()

    def save_paper(self, fig, name, title, caption, note, sources, placement):
        count = validate_layout(fig)
        require(all('FATE' not in t.get_text() for t in visible_text(fig)), 'Legacy method label in figure')
        require('FATE' not in caption, 'Legacy method label in caption')
        for ext in ('pdf', 'svg', 'png'):
            fig.savefig(self.out / f'{name}.{ext}', dpi=600)
        self.pdf.savefig(fig)
        width, height = fig.get_size_inches()
        with Image.open(self.out / f'{name}.png') as im:
            gray = im.convert('L')
            ink = float(np.mean(np.asarray(gray) < 230))
            require(.01 < ink < .7, f'Blank or overfilled image: {name}')
            gray.save(self.out / f'{name}_gray.png', dpi=(600, 600))
        self.catalog.append(dict(name=name, title=title, caption=caption, note=note,
            sources=sources, placement=placement, columns=1 if width == SINGLE else 2,
            width_inches=float(width), height_inches=float(height), minimum_font_pt=9,
            visible_text_count=count, text_bounds='PASS', tick_overlap='PASS', nonwhite_fraction=ink))
        plt.close(fig)
        print('saved', name, flush=True)


def main_results(f, inputs):
    rows = inputs.csv('p0_per_policy.csv')
    labels = ['Initial'] + [short(m) for m in BASE]
    definitions = [
        ('01_clean_return', SINGLE, 3.45, [('clean', 'task_return', 1, (0, 4.7), [0, 1, 2, 3, 4], 'Clean task return')]),
        ('02_fault_exposure', DOUBLE, 3.25, [(s, 'coverage', 100, (0, 100), [0, 25, 50, 75, 100],
            f'Subtype exposure (%)\n({chr(97+i)}) {s.title()}') for i, s in enumerate(('seen', 'holdout', 'sparse'))]),
    ]
    for name, width, height, specs in definitions:
        fig, axes = canvas(len(specs), width=width, height=height)
        legend(fig)
        for i, (ax, (split, metric, scale, lim, ticks, xlabel)) in enumerate(zip(axes, specs)):
            row_axis(ax, labels, lim, ticks, xlabel, i == 0)
            ax.axhline(.5, color='#aaaaaa', linewidth=.6)
            for method, y, profile in paired_rows():
                values = np.array(policy_values(rows, method, split, metric)) * scale
                f.record(name, method, split + '/' + metric, values, kind=kind(method))
                dot(ax, y, values, STYLE[profile])
            if metric == 'task_return':
                ax.axvline(np.mean(policy_values(rows, 'CleanEval', 'clean', metric)),
                           color=GRAY, ls=':', lw=.75, zorder=1)
        if name.startswith('01'):
            caption = ('Clean-evaluation task return after adaptation from P0. Circles and open squares denote fault- '
                'and clean-environment adaptation, respectively; the diamond and dotted line denote the initial policy. '
                'Small points show five adaptation seeds (five evaluation slots of one policy for Initial); large symbols '
                'and bars show the mean and sample standard deviation. Each policy is evaluated for 30K callbacks '
                'without intrinsic reward. Dense, Sparse, and Abs. level denote DenseSurprise, SparseSurprise, '
                'and AbsoluteSameShape, respectively.')
            title, note = 'P0 게임 수행 능력', '기존 13조건을 생략하지 않고 같은 방법의 두 적응 환경을 한 행에 배치했다.'
        else:
            caption = ('Fault-subtype exposure after adaptation from P0. Symbols and bars show the mean and sample '
                'standard deviation across five policy slots, with individual slots shown as small points. Seen uses one '
                '60K-callback evaluation per policy; Holdout and Sparse average two separate 60K evaluations within '
                'each policy before aggregating across slots. Exposure counts wrapper-recorded operator applications, '
                'not independently verified failures. Initial is one fixed policy evaluated in five slots. Abbreviations '
                'Dense, Sparse, and Abs. level denote DenseSurprise, SparseSurprise, and AbsoluteSameShape.')
            title, note = '평가 profile별 결함 유형 노출', '반복 합집합이 아닌 정책 내부 평균이다. TEA의 일관된 최고 성능을 주장하지 않는다.'
        f.save_paper(fig, name, title, caption, note, ['p0_per_policy.csv'], '본문 주 결과')


def behavior(f, inputs):
    rows = inputs.csv('p0_behavior_per_seed.csv')
    name = '03_behavior_repertoire'
    fig, axes = canvas(3)
    legend(fig)
    for i, (ax, key, lim, ticks, xlabel) in enumerate(zip(axes,
            ['contexts', 'context_actions', 'transition_patterns'], [(0, 160), (0, 1350), (0, 3100)],
            [[0, 40, 80, 120, 160], [0, 400, 800, 1200], [0, 1000, 2000, 3000]],
            ['Unique contexts\n(a)', 'Context-action pairs\n(b)', 'Transition patterns\n(c)'])):
        row_axis(ax, ['Initial'] + [short(m) for m in BASE], lim, ticks, xlabel, i == 0)
        ax.axhline(.5, color='#aaaaaa', lw=.6)
        for method, y, profile in paired_rows():
            chosen = seeded(select(rows, method=method))
            require(all(int(r['steps']) == 30000 for r in chosen), 'Behavior budget mismatch')
            values = [float(r[key]) for r in chosen]
            f.record(name, method, key, values, kind=kind(method))
            dot(ax, y, values, STYLE[profile])
    f.save_paper(fig, name, '관측 행동 조합의 다양성',
        'Observed behavioral repertoire in 30K-callback Clean evaluations of P0 policies. Contexts encode inventory, '
        'front tile, achievement stage, and nearby-object information; actions are aligned to pre-transition contexts. '
        'Symbols and bars show means and sample standard deviations over five adaptation seeds, or five evaluation '
        'slots of the initial policy. These counts measure observed combinations, not spatial or code coverage, and '
        'are not controlled for progression or visitation frequency. Dense, Sparse, and Abs. level are the shortened '
        'method names defined in the performance comparison.',
        '넓은 행동 조합과 결함 노출 사이의 인과관계를 입증하는 그림은 아니다.',
        ['p0_behavior_per_seed.csv'], '본문 주 결과')


def repeats(f, inputs):
    rows = inputs.csv('p0_per_policy.csv')
    summary = inputs.csv('p0_summary.csv')
    name = '04_sparse_repeat'
    fig, axes = canvas(2, height=4.45, left=1.35, gap=.48)
    legend(fig, [(STYLE['Fault'], 'First evaluation'), (STYLE['Clean'], 'Repeated evaluation')])
    labels = ['Initial'] + [('F: ' if m.startswith('Fault') else 'C: ') + short(m) for m in METHODS[1:]]
    for i, ax in enumerate(axes):
        row_axis(ax, labels, [(0, 100), (-.2, 7.2)][i], [list(range(0, 101, 25)), list(range(8))][i],
                 ['Subtype exposure (%)\n(a)', 'Common subtypes (of 7)\n(b)'][i], i == 0)
        for y in (.5, 6.5):
            ax.axhline(y, color='#aaaaaa', lw=.6)
    for y, method in enumerate(METHODS):
        r = one(summary, method=method, split='sparse')
        for key, offset, style in [('first', -.15, STYLE['Fault']), ('repeat', .15, STYLE['Clean'])]:
            values = 100 * np.array(policy_values(rows, method, 'sparse', key + '_coverage'))
            f.record(name, method, key + '_coverage_pct', values, kind=kind(method))
            dot(axes[0], y + offset, values, style, individual=False)
            count = int(r[key + '_consistent_types'])
            f.summary(name, method, key + '_common_types', count, None, 5, 'across_slot_intersection')
            dot(axes[1], y + offset, [count], style, individual=False, error=False)
    f.save_paper(fig, name, 'Sparse 반복 평가 안정성',
        'Sparse exposure in two separate fixed-policy evaluation rounds. F and C indicate fault- and clean-environment '
        'adaptation; Initial denotes one fixed pretrained policy. (a) Mean and sample standard deviation over five '
        'policy slots in each 60K-callback round. (b) Number of subtypes exposed in all five slots, computed separately '
        'for each round. The intersection is not a mean, a best-seed result, or a pooled two-round union. The labels '
        'Dense, Sparse, and Abs. level abbreviate DenseSurprise, SparseSurprise, and AbsoluteSameShape.',
        '유리한 첫 평가만 보여주지 않는다. 공통 유형 수에는 SD를 붙이지 않는다.',
        ['p0_per_policy.csv', 'p0_summary.csv'], '본문 주 결과 또는 반복 평가 표 대체')


def temporal(f, inputs):
    rows = inputs.csv('p0_training_reward_per_seed.csv')
    policy = inputs.csv('p0_per_policy.csv')
    name = '05_temporal_ablation'
    fig, axes = canvas(4, height=2.25, left=.92, gap=.38, bottom=.68)
    legend(fig)
    configs = [('task_return', 1, (0, 4.8), [0, 2, 4], 'Clean return\n(a)'),
               ('bonus_mean', 1000, (0, 45), [0, 20, 40], 'Bonus per step\n' + r'($10^{-3}$ units)' + '\n(b)'),
               ('zero_fraction', 100, (0, 100), [0, 50, 100], 'Bonus-positive\ntransitions (%)\n(c)'),
               ('movement_step_fraction', 100, (0, 100), [0, 50, 100], 'Movement\nactions (%)\n(d)')]
    for i, (ax, (key, scale, lim, ticks, xlabel)) in enumerate(zip(axes, configs)):
        row_axis(ax, [short('FATE'), 'Abs. level'], lim, ticks, xlabel, i == 0)
        for method, y, profile in paired_rows(False, ['FATE', 'AbsoluteSameShape']):
            values = np.array(policy_values(policy, method, 'clean', key) if key == 'task_return'
                else [float(r[key]) for r in seeded(select(rows, method=method))])
            if key == 'zero_fraction':
                values = 1 - values
            values *= scale
            metric = {'bonus_mean': 'bonus_milliunits', 'zero_fraction': 'positive_bonus_pct'}.get(key, key)
            f.record(name, method, metric, values)
            dot(ax, y, values, STYLE[profile])
    f.save_paper(fig, name, '시간 차분 제거 대조',
        'Temporal-differencing ablation from P0. AbsoluteSameShape (Abs. level) replaces the positive temporal '
        'increase with absolute normalized surprise, keeping the coefficient (0.2), ramp threshold (0.5), and clipping '
        'level (2.0) fixed. Symbols and bars show means and sample standard deviations across five adaptation seeds. '
        'Panels (b)-(d) summarize non-reset training transitions. Realized bonus magnitudes are not matched: the '
        'absolute-level variant pays approximately 21 times the mean bonus of TEA. Movement frequency alone does '
        'not establish ineffective behavior.',
        '실현 보상 강도까지 동일한 대조군으로 표현하지 않는다. 차분만의 인과 이득을 확정할 수 없다.',
        ['p0_training_reward_per_seed.csv', 'p0_per_policy.csv'], '본문 기전 분석')


def cross(f, inputs):
    rows = inputs.csv('cross_primary_per_policy.csv')
    name = '06_initialization_reference'
    fig, axes = canvas(2, height=2.65, left=1.02, gap=.5)
    legend(fig, [(STYLE['Fault'], 'P0 initialization'), (STYLE['Clean'], 'P1 initialization')])
    methods = ['CleanEval', 'TaskOnly', 'FATE_R0', 'FATE_R1']
    for i, (ax, split, metric, scale, lim, ticks, xlabel) in enumerate(zip(axes,
            ['clean', 'sparse'], ['task_return', 'coverage'], [1, 100], [(0, 4.6), (0, 100)],
            [[0, 1, 2, 3, 4], [0, 25, 50, 75, 100]], ['Clean task return\n(a)', 'Sparse exposure (%)\n(b)'])):
        row_axis(ax, ['Initial', 'TaskOnly', 'TEA + R0', 'TEA + R1'], lim, ticks, xlabel, i == 0)
        ax.axhline(.5, color='#aaaaaa', lw=.6)
        for y, method in enumerate(methods):
            for prior, offset, style in [('P0', -.16, STYLE['Fault']), ('P1', .16, STYLE['Clean'])]:
                chosen = seeded(select(rows, prior=prior, method=method, split=split), 3, 'adapt_seed')
                require(all(r['eval_repeats'] == '2' for r in chosen), 'Cross repeats mismatch')
                values = [float(r[metric]) * scale for r in chosen]
                f.record(name, prior + '/' + method, metric, values, kind=kind(method))
                dot(ax, y + offset, values, style)
    f.save_paper(fig, name, '초기 정책과 reference의 교차 비교',
        'Initialization-reference cross comparison under clean-environment adaptation. P0 and P1 are distinct '
        'pretrained checkpoints; R0 and R1 are their frozen reference models. Each point summarizes two evaluations '
        'of a policy, and bars show sample standard deviations across three completed adaptation seeds per condition '
        '(evaluation slots for Initial). This is a separate cohort from the five-seed P0 comparison. The incomplete '
        'seed-3/4 extension is excluded.',
        '3시드 결과로 명시한다. P1 하락을 reference만의 문제로 단정하지 않는다.',
        ['cross_primary_per_policy.csv'], '본문 한계와 조건 의존성')


def restore(f, inputs):
    rows = inputs.csv('joint_returns.csv')
    name = '07_joint_restoration'
    fig, axes = canvas(1, width=SINGLE, height=2.7, left=1.08, bottom=.48)
    ax = axes[0]
    legend(fig, [(STYLE['Fault'], 'P0'), (STYLE['Clean'], 'P1')])
    row_axis(ax, ['Initial', 'TaskOnly', 'Joint restored'], (0, 4.5), [0, 1, 2, 3, 4], 'Clean task return')
    for prior, offset, style in [('P0', -.16, STYLE['Fault']), ('P1', .16, STYLE['Clean'])]:
        for y, (cohort, method, rounds) in enumerate([
                ('existing', 'CleanEval', ('base', 'base_repeat')),
                ('existing', 'TaskOnly', ('final', 'repeat_final')),
                ('joint_restored', 'TaskOnly', ('final', 'repeat_final'))]):
            chosen = sorted([r for r in rows if r['prior'] == prior and r['cohort'] == cohort
                and r['method'] == method and r['evaluation_round'] in rounds], key=lambda r: int(r['eval_seed']))
            require(len(chosen) == 2 and {r['eval_seed'] for r in chosen} == {'0', '20000'}, 'Pilot evaluations mismatch')
            values = [float(r['task_return']) for r in chosen]
            f.record(name, prior + '/' + cohort + '/' + method, 'task_return', values,
                     ids=[0, 20000], kind='evaluation_repeat_one_trained_policy')
            dot(ax, y + offset, values, style, error=False)
    f.save_paper(fig, name, '공동 복원 파일럿',
        'TaskOnly joint-restoration pilot with one adaptation seed for each initialization. Joint restoration changes '
        'the train ratio from 128 to 512, replay capacity from 100K to 200K, and restores stored Clean experience '
        'together. Small points are two evaluation seeds of each policy; large symbols are their means. No '
        'training-seed error bars are shown. This pilot changes both the protocol and computation and does not '
        'identify an individual causal factor.',
        'P0와 P1 각각 1학습 시드다. 두 평가를 독립 학습으로 세지 않는다.',
        ['joint_returns.csv'], '진단 보조 그림')


def wuji(f, inputs):
    rows = inputs.csv('p0_per_policy.csv')
    ports = ['DDQN_update1', 'DDQN_update8', 'A2C']
    data = {p: inputs.csv(f'wuji_{p}_per_seed.csv') for p in ports}
    dreamer = ['CleanEval', 'FaultAdaptTaskOnly', 'FaultAdaptRND', 'FaultAdaptDenseSurprise',
               'FaultAdaptFATE', 'CleanAdaptFATE']
    names = ['WUJI DDQN (u=1)', 'WUJI DDQN (u=8)', 'WUJI A2C', 'Dreamer initial',
             'TaskOnly (F)', 'RND (F)', 'Dense (F)', 'TEA (F)', 'TEA (C)']
    name = '08_external_comparison'
    fig, axes = canvas(3, height=3.6, left=1.57, top=.14, gap=.33)
    for i, (ax, split, metric, scale, lim, ticks, xlabel) in enumerate(zip(axes,
            ['clean', 'sparse', 'sparse'], ['task_return', 'coverage', 'events_per_10k'], [1, 100, 1],
            [(0, 4.8), (0, 100), (0, 21)], [[0, 2, 4], [0, 50, 100], [0, 10, 20]],
            ['Clean return\n(a)', 'Sparse exposure (%)\n(b)', 'Events per 10K\n(c)'])):
        row_axis(ax, names, lim, ticks, xlabel, i == 0)
        ax.axhline(2.5, color='#aaaaaa', lw=.6)
        for y, port in enumerate(ports):
            values = [float(r[metric]) * scale for r in seeded(select(data[port], split=split))]
            f.record(name, port, split + '/' + metric, values, kind='independent_population_seed')
            dot(ax, y, values, (GRAY, '^', 'white'))
        for y, method in enumerate(dreamer, 3):
            values = np.array(policy_values(rows, method, split, metric)) * scale
            f.record(name, method, split + '/' + metric, values, kind=kind(method))
            profile = 'Initial' if method == 'CleanEval' else 'Fault' if method.startswith('Fault') else 'Clean'
            dot(ax, y, values, STYLE[profile])
    f.save_paper(fig, name, 'WUJI 외부 시스템 비교',
        'External comparison of WUJI Craftax ports and selected Dreamer conditions. Means and sample standard '
        'deviations summarize five WUJI population seeds or five Dreamer adaptation seeds (evaluation slots for '
        'the initial policy). The DDQN update multiplier is denoted by u; F and C indicate fault- and clean-environment '
        'adaptation. WUJI uses 2.1M actual actions per population including fitness evaluation, while Dreamer uses '
        'approximately 1.1M pretraining plus 1M adaptation callbacks. WUJI Sparse results use one evaluation round '
        'and Dreamer results average two rounds per policy. Architecture, population selection, updates, and '
        'computation are not matched; this is not a model-free versus model-based ablation.',
        '외부 시스템 비교다. 모든 Dreamer 방법은 01~04에서 제시하며 여기에는 주요 비교군을 표시했다.',
        [f'wuji_{p}_per_seed.csv' for p in ports] + ['p0_per_policy.csv'], '본문 외부 비교')


def subtype(f, inputs):
    rows = inputs.csv('p0_sparse_repeat_opportunities.csv')
    name = '09_sparse_subtypes'
    fig, axes = canvas(1, height=4.1, left=1.3, right=.63, top=.15, bottom=.7)
    ax = axes[0]
    mat = np.empty((len(METHODS), len(SUBTYPES)))
    for y, method in enumerate(METHODS):
        for x, (key, _) in enumerate(SUBTYPES):
            r = one(rows, method=method, subtype=key)
            require(int(r['n']) == 5, 'Subtype slot count mismatch')
            mat[y, x] = int(r['hit_seeds'])
            f.summary(name, method, key + '/hit_slots', mat[y, x], None, 5, 'observed_slot_count')
    im = ax.imshow(mat, cmap='Blues', vmin=0, vmax=5, aspect='auto', interpolation='nearest')
    for y in range(len(METHODS)):
        for x in range(len(SUBTYPES)):
            ax.text(x, y, str(int(mat[y, x])), ha='center', va='center',
                    color='white' if mat[y, x] >= 4 else 'black', fontsize=9)
    ax.set_yticks(range(len(METHODS)), ['Initial'] + [
        ('F: ' if m.startswith('Fault') else 'C: ') + short(m) for m in METHODS[1:]])
    ax.set_xticks(range(7), ['Action\nremap', 'Reward\nzero', 'Reward\ndelay', 'Revisit\ndelay',
                            'Switch\nfailure', 'Achievement\nomission', 'Inventory\ndesync'])
    ax.tick_params(length=0)
    for y in (.5, 6.5):
        ax.axhline(y, color='white', lw=1.2)
    ax.spines[['left', 'bottom']].set_visible(False)
    cb = fig.colorbar(im, cax=fig.add_axes([.934, .7/4.1, .015, (4.1-.85)/4.1]), ticks=range(6))
    cb.set_label('Exposed slots (of 5)', labelpad=4)
    f.save_paper(fig, name, 'Sparse 유형별 발견 슬롯',
        'Subtype exposure in the common repeated Sparse evaluation round. Each cell gives the number of five '
        'policy slots with at least one wrapper-recorded application. F and C indicate fault- and clean-environment '
        'adaptation. The inventory-desynchronization subtype has no condition-met occurrences in these 65 '
        'evaluations; it remains in the coverage denominator. Reward delay denotes the two-reward delay subtype. '
        'A successful smoke test does not establish that trained policies reach a subtype trigger.',
        '모든 방법에서 발견하지 못한 유형을 삭제하지 않았다. 진한 칸은 빈도가 아니라 발견한 슬롯 수다.',
        ['p0_sparse_repeat_opportunities.csv'], '본문 subtype 분석 또는 보조 그림')


def drift(f, inputs):
    rows = inputs.csv('diagnostic_training_bins_summary.csv')
    name = '10_bonus_drift'
    fig, axes = canvas(3, height=2.65, left=.5, top=.4, bottom=.68, gap=.49)
    handles = []
    for i, (ax, key, scale, xlabel) in enumerate(zip(axes,
            ['mean_bonus', 'bonus_abs_task_ratio', 'mean_task_episode_return'], [1000, 100, 1],
            ['Adaptation callbacks (M)\n(a) Bonus ' + r'($10^{-3}$ units)', 'Adaptation callbacks (M)\n(b) Bonus / abs. task (%)',
             'Adaptation callbacks (M)\n(c) Task episode return'])):
        for prior, style, ls in [('P0', STYLE['Fault'], '-'), ('P1', STYLE['Clean'], '--')]:
            chosen = sorted(select(rows, prior=prior, method='FATE'), key=lambda r: int(r['bin']))
            require(len(chosen) == 10 and all(r['n'] == '3' for r in chosen), 'Diagnostic cohort incomplete')
            x = np.arange(1, 11) / 10
            means = np.array([float(r[key + '_mean']) for r in chosen]) * scale
            sd = np.array([float(r[key + '_sd']) for r in chosen]) * scale
            c, marker, face = style
            line, = ax.plot(x, means, color=c, marker=marker, mfc=face, ms=3.3, mew=.7,
                            lw=1, ls=ls, label=prior)
            ax.fill_between(x, means - sd, means + sd, color=c, alpha=.12, linewidth=0)
            if i == 0:
                handles.append(line)
            for r, mean, spread in zip(chosen, means, sd):
                f.summary(name, prior + '/bin' + r['bin'], key + ('_milliunits' if i == 0 else ''),
                          mean, spread, 3, 'adaptation_seed')
        ax.set_xlim(.07, 1.03)
        ax.set_xticks([.1, .4, .7, 1])
        ax.set_xlabel(xlabel)
        ax.set_ylim(bottom=0)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#e3e3e3', lw=.5)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .995), ncol=2, frameon=False)
    f.save_paper(fig, name, '학습 보너스와 task return의 변화',
        'Training diagnostics for TEA under clean-environment adaptation in an earlier, separate three-seed '
        'cohort. Lines show seed means in successive 100K-callback bins; shading shows sample standard deviations. '
        'Panels report the mean intrinsic bonus per non-reset transition, the bonus-to-absolute-task-reward ratio, '
        'and task-only training episode return. These training returns are not fixed-policy evaluation returns. '
        'The co-occurrence of bonus growth and task-return degradation does not establish a causal direction.',
        '교차 reference 실험과 합치지 않는다. 실제 보너스 증가와 수행의 관계를 진단하는 자료다.',
        ['diagnostic_training_bins_summary.csv'], '보상 drift 보조 그림')


def tex_escape(text):
    return ''.join({'&': r'\&', '%': r'\%', '_': r'\_', '#': r'\#',
                    '$': r'\$', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
                    '^': r'\textasciicircum{}', '\\': r'\textbackslash{}'}.get(c, c) for c in text)


def finish(f, inputs):
    f.pdf.close()
    for name, values in [('plotted_points.csv', f.points), ('plotted_summaries.csv', f.summaries)]:
        with (f.out / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    preview = Image.new('RGB', (1800, 620 * 5), 'white')
    draw = ImageDraw.Draw(preview)
    tex = ['% Requires \\usepackage{graphicx}; run LaTeX from this directory or set \\graphicspath.',
           '% Figure numbers follow manuscript placement, not asset filenames.']
    lines = ['# TEA IEEE ToG 논문용 그림', '',
        '2026-10-19 완료 결과를 인쇄 크기에 맞춰 다시 그렸다. 대형 제목과 해석 문구를 그림에서 제거하고 영문 캡션으로 분리했다. '
        '기존 보고용 그림과 수치는 그대로 보존했다.', '',
        '[그림 전체 PDF](tea_tog_figures.pdf) | [미리보기](overview.jpg) | [영문 캡션과 LaTeX 삽입문](figures.tex)', '',
        '## 규격', '',
        f'- [IEEE 크기 가이드]({GUIDE})에 맞춘 1단 3.5인치, 2단 7.16인치. 여백 잘라내기로 파일 폭이 달라지지 않게 저장했다.',
        f'- [IEEE 글자 및 흑백 가독성 가이드]({TYPE_GUIDE})를 참고해 9pt 글자, 색과 마커 모양의 중복 인코딩을 사용했다.',
        '- 폰트는 시스템에 설치된 Liberation Serif를 PDF에 임베딩했다. Times New Roman으로 잘못 표기하지 않는다.',
        '- 제출·원고 삽입은 벡터 PDF 권장. PNG는 600dpi, SVG는 편집용이며 회색조 PNG도 제공한다.',
        '- 실제 원고의 columnwidth/textwidth에 넣은 뒤 최종 크기를 확인한다. 2단 그림을 1단으로 줄이면 9pt 글자가 작아진다.', '',
        '## 본문 구성 제안', '',
        '**01 수행 → 02 결함 노출 → 03 행동 다양성 → 05 차분 제거 → 06 초기 정책 의존성 → 08 외부 비교**를 중심으로 배치한다. '
        '04 반복 평가 결과는 본문 표로 대체할 수 있다. 07·09·10은 진단 자료이며 지면에 맞춰 선택하되 '
        '반복 평가의 불안정성과 P1 결과를 해석에서 누락하지 않는다. 10개를 모두 본문에 넣으라는 의미는 아니다.', '',
        '| 파일 | 내용 | 폭 | 권장 위치 |', '|---|---|---|---|']
    for i, item in enumerate(f.catalog):
        name = item['name']
        lines.append(f'| [{name}.pdf]({name}.pdf) | {item["title"]} | {item["columns"]}단 | {item["placement"]} |')
        environment = 'figure' if item['columns'] == 1 else 'figure*'
        width = r'\columnwidth' if item['columns'] == 1 else r'\textwidth'
        tex += ['', f'\\begin{{{environment}}}[t]', r'  \centering',
                f'  \\includegraphics[width={width}]{{{name}.pdf}}',
                r'  \caption{' + tex_escape(item['caption']) + '}',
                f'  \\label{{fig:tea-{name.replace("_", "-")}}}', f'\\end{{{environment}}}']
        with Image.open(f.out / f'{name}.png') as im:
            im.thumbnail((870, 560))
            x, y = (i % 2) * 900, (i // 2) * 620
            preview.paste(im, (x + (900 - im.width) // 2, y + 40))
            draw.text((x + 30, y + 10), name, fill='black')
    preview.save(f.out / 'overview.jpg', quality=95)
    for start in range(0, 10, 4):
        preview.crop((0, (start // 2) * 620, 1800, min(((start + 4) // 2) * 620, preview.height))).save(
            f.out / f'preview_{start // 4 + 1:02d}.jpg', quality=95)
    for item in f.catalog:
        name = item['name']
        lines += ['', f'## {item["title"]}', '',
            f'[PDF]({name}.pdf) / [PNG]({name}.png) / [SVG]({name}.svg) / [흑백 확인]({name}_gray.png)', '',
            f'![{item["title"]}]({name}.png)', '', '**영문 캡션:** ' + item['caption'], '',
            '**작성 시 주의:** ' + item['note'], '',
            '**원자료:** ' + ', '.join(f'[{s}](inputs/{s})' for s in item['sources'])]
    lines += ['', '## 통계 및 재현', '',
        '오차선은 sample SD이며 CI/SEM이 아니다. P0의 5 adaptation seeds, 교차 비교의 3 seeds, '
        '공동 복원 1 seed, WUJI 5 population seeds를 분리했다. Initial은 한 정책의 평가 슬롯이다. '
        '반복 평가는 각 정책 내부에서 평균하고 학습 표본 수를 늘리지 않았다.', '',
        '입력 SHA256은 기존 스냅샷과 대조했다. 표시 수치와 표본 단위는 `plotted_points.csv`, '
        '`plotted_summaries.csv`에 보존했다. `manifest.json`에 그래프별 폭·높이·글자 경계·입력 해시를 기록했다.', '',
        '실행 코드: `dreamerv3/plot_fate_tog_figures.py`. 새 학습·rollout·큐 변경 없이 CPU만 사용한다. '
        'LaTeX 삽입문은 제공하지만 이 시스템에 LaTeX가 없어 원고 자체를 컴파일한 것은 아니다.']
    (f.out / 'figures.tex').write_text('\n'.join(tex) + '\n')
    (f.out / 'README.md').write_text('\n'.join(lines) + '\n')
    manifest = dict(sources=inputs.sources, figures=f.catalog, no_training_or_rollout=True,
        statistics='sample SD; nested evaluations averaged within policy; cohorts not pooled',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        outputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in f.out.iterdir() if p.is_file()})
    (f.out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')


def core_pack(materials, source, out):
    """Regenerate the four core figures; retain historical data IDs and numbers."""
    require(not out.exists(), 'Select a new output directory')
    manifest = json.loads((source / 'manifest.json').read_text())
    selected = ['05_temporal_ablation', '06_initialization_reference', '08_external_comparison']
    for name in [*(n + '.pdf' for n in selected), 'plotted_points.csv', 'plotted_summaries.csv']:
        path = source / name
        require(hashlib.sha256(path.read_bytes()).hexdigest() == manifest['outputs'][name],
                f'Original figure evidence changed: {path}')
    png_names = ['01_main_results', '02_temporal_ablation', '03_p0_p1_comparison', '04_wuji_comparison']
    out.mkdir(parents=True)
    setup_style()
    with tempfile.TemporaryDirectory(prefix='tea-core-') as temporary:
        work = Path(temporary)
        (work / 'inputs').mkdir()
        inputs = Inputs(materials, work)
        figures = PaperFigures(work)
        policies = inputs.csv('p0_per_policy.csv')
        behaviors = inputs.csv('p0_behavior_per_seed.csv')
        policy_summary = inputs.csv('p0_summary.csv')
        behavior_summary = inputs.csv('p0_behavior_summary.csv')
        fig, axes = canvas(3)
        legend(fig)
        specs = [('clean', 'task_return', 1, (0, 4.7), [0, 1, 2, 3, 4], 'Clean task return\n(a)'),
                 ('clean', 'context_actions', 1, (0, 1350), [0, 400, 800, 1200], 'Context-action pairs\n(b)'),
                 ('sparse', 'coverage', 100, (0, 100), [0, 25, 50, 75, 100], 'Sparse exposure (%)\n(c)')]
        for i, (ax, (split, metric, scale, lim, ticks, xlabel)) in enumerate(zip(axes, specs)):
            row_axis(ax, ['Initial'] + [short(m) for m in BASE], lim, ticks, xlabel, i == 0)
            ax.axhline(.5, color='#aaaaaa', lw=.6)
            for method, y, profile in paired_rows():
                if metric == 'context_actions':
                    chosen = seeded(select(behaviors, method=method))
                    require(all(int(r['steps']) == 30000 for r in chosen), 'Behavior budget mismatch')
                    values = np.array([float(r[metric]) for r in chosen])
                    expected = one(behavior_summary, method=method)
                else:
                    values = np.array(policy_values(policies, method, split, metric)) * scale
                    expected = one(policy_summary, method=method, split=split)
                mean, sd = stats(values)
                require(np.isclose(mean, scale * float(expected[metric + '_mean']))
                        and np.isclose(sd, scale * float(expected[metric + '_sd'])),
                        f'Core summary mismatch: {method}/{metric}')
                figures.record(png_names[0], method, split + '/' + metric, values, kind=kind(method))
                dot(ax, y, values, STYLE[profile])
        validate_layout(fig)
        require(all('FATE' not in t.get_text() for t in visible_text(fig)), 'Legacy label in main figure')
        overview = work / 'overview.pdf'
        fig.savefig(overview)
        fig.savefig(out / f'{png_names[0]}.png', dpi=600)
        plt.close(fig)
        for plot in (temporal, cross, wuji):
            plot(figures, inputs)
        figures.pdf.close()
        items = figures.catalog
        # The three regenerated panels must preserve every historical plotted value.
        for filename, rows in [('plotted_points.csv', figures.points), ('plotted_summaries.csv', figures.summaries)]:
            with (source / filename).open() as stream:
                original = [r for r in csv.DictReader(stream) if r['figure'] in selected]
            current = [r for r in rows if r['figure'] in selected]
            require(len(original) == len(current), 'Changed plotted row count')
            require(all({k: '' if v is None else str(v) for k, v in b.items()} == a
                        for a, b in zip(original, current)), f'Changed plotted values: {filename}')
        for item in items:
            old = one(manifest['figures'], name=item['name'])
            require(item['caption'] == old['caption'].replace('FATE', 'TEA'), 'Unexpected caption change')
        for old_name, new_name in zip(selected, png_names[1:]):
            shutil.copyfile(work / f'{old_name}.png', out / f'{new_name}.png')
        subprocess.run(['pdfunite', str(overview),
                        *[str(work / (name + '.pdf')) for name in selected],
                        str(out / 'tea_core.pdf')], check=True)
        provenance = dict(display_name='TEA', historical_method_id='FATE', name_change_only=True,
            sources=inputs.sources, reference_manifest=str((source / 'manifest.json').resolve()),
            reference_manifest_sha256=hashlib.sha256((source / 'manifest.json').read_bytes()).hexdigest(),
            points=figures.points, summaries=figures.summaries,
            old_selected_points_and_summaries='identical', main_summary_checks=39,
            no_training_or_rollout=True, completed_results_only=True,
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    caption = (
        'Performance, behavioral repertoire, and fault exposure after adaptation from P0. Circles and open squares '
        'denote fault- and clean-environment adaptation; the diamond denotes the initial policy. Large symbols '
        'and bars show means and sample standard deviations across five adaptation seeds (five evaluation slots '
        'of one policy for Initial); small points show individual slots. Clean return excludes intrinsic reward. '
        'Context-action counts use pre-transition contexts and are not controlled for progression. Panels (a)-(b) '
        'use 30K-callback Clean evaluations. Sparse exposure averages two 60K evaluations within each policy and '
        'counts wrapper-recorded subtype applications, not independently verified failures. Dense, Sparse, and '
        'Abs. level abbreviate DenseSurprise, SparseSurprise, and AbsoluteSameShape.')
    titles = ['주 결과: 수행·행동 다양성·Sparse 노출', '시간 차분 제거 대조',
              'P0/P1 초기 정책 의존성', 'WUJI 외부 시스템 비교']
    lines = ['# TEA 논문 핵심 그림', '',
        '**[핵심 그림 4장 PDF](tea_core.pdf).** 같은 그림의 600dpi PNG 4개도 제공한다. '
        'TEA는 Temporal-excess adaptation이며, 이전 그림의 방법 표시와 캡션만 변경했다. '
        '수치·오차선·구성은 유지했고, 실행 중인 추가 큐의 결과는 포함하지 않았다. '
        '원자료의 FATE 실행 ID는 재현성을 위해 그대로 보존한다.', '',
        '| PDF 페이지 | 고해상도 PNG | 본문에서 보여줄 내용 |', '|---|---|---|',
        '| 1 | [수행·다양성·Sparse 노출](01_main_results.png) | P0 전체 13조건의 이득과 한계를 한눈에 비교 |',
        '| 2 | [시간 차분 제거](02_temporal_ablation.png) | 같은 ramp/cap에서 보상·행동·수행이 어떻게 달라지는가 |',
        '| 3 | [P0/P1 교차 비교](03_p0_p1_comparison.png) | 이득이 초기 정책 전반에 일관되지는 않는다는 한계 |',
        '| 4 | [WUJI 비교](04_wuji_comparison.png) | 외부 테스팅 시스템과 수행·노출을 함께 비교 |', '',
        '복원 파일럿·subtype heatmap·보너스 drift는 본문 핵심 그림에서 뺐다. Seen/Holdout 수치와 '
        'Sparse 반복별 공통 유형은 표로 정리하면 된다. Sparse의 첫 평가만 고르지 않고 두 평가 평균을 유지했다. '
        '원자료와 나머지 후보는 기존 위치에 그대로 보존했다.', '',
        '## 원고에 넣기', '',
        '모두 2단 폭이다. PDF를 분리하지 않아도 페이지를 지정해서 삽입할 수 있다. 아래 page 값을 1~4로 바꾼다.', '',
        '```latex', r'\includegraphics[page=1,width=\textwidth]{tea_core.pdf}', '```', '',
        '## 영문 캡션', '']
    for i, (title, text) in enumerate(zip(titles, [caption] + [item['caption'] for item in items]), 1):
        lines += [f'### {i}. {title}', '', text, '']
    lines += ['## 해석과 원자료', '',
        '- 오차선은 표본 SD다. P0는 5 adaptation seeds, 교차 비교는 3 seeds, WUJI는 5 population seeds다.',
        '- 시간 차분 제거 실험은 실현 보상 강도까지 맞춘 비교가 아니다. WUJI는 matched backbone ablation이 아니다.',
        '- [전체 결과표](../fate_all_results_20261019/tables.md)',
        '- [보관된 그림·원자료·캡션](../fate_tog_figures_20261019_final/README.md)', '',
        '주 결과 39개 평균·SD를 기존 집계와 대조하고, 나머지 세 그림도 원자료에서 다시 그린 뒤 '
        '모든 개별 점과 요약 수치가 이전 그림 기록과 동일한지 검사했다. 입력 해시와 수치는 `manifest.json`에 보존한다. '
        '재생성은 `dreamerv3.plot_fate_tog_figures --core-only --out <새 경로>`로 한다.']
    (out / 'README.md').write_text('\n'.join(lines) + '\n')
    provenance['outputs'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in out.iterdir() if p.is_file()}
    (out / 'manifest.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n')
    print(f'Wrote four TEA figures as PDF and 600dpi PNG: {out}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--materials', type=Path, default=REPO / 'analysis_outputs/fate_all_results_20261019')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--core-only', action='store_true')
    parser.add_argument('--paper-source', type=Path,
                        default=REPO / 'analysis_outputs/fate_tog_figures_20261019_final')
    args = parser.parse_args()
    if args.core_only:
        core_pack(args.materials, args.paper_source, args.out)
        return
    require(not args.out.exists(), 'Select a new output directory')
    (args.out / 'inputs').mkdir(parents=True)
    inputs, figures = Inputs(args.materials, args.out), PaperFigures(args.out)
    for plot in (main_results, behavior, repeats, temporal, cross, restore, wuji, subtype, drift):
        plot(figures, inputs)
    finish(figures, inputs)
    print(f'Wrote {len(figures.catalog)} paper figures to {args.out}', flush=True)


if __name__ == '__main__':
    main()
