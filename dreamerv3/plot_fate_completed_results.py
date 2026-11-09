"""Plot the October 19 completed-result snapshot, without RL or GPU execution."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from dreamerv3.plot_fate_manuscript_materials import (
    Figures, Inputs, PALETTE, INK, MUTED, axis, core, dotrow, hlabels)
from dreamerv3.plot_fate_professor_discussion import one, require, seeded, select, stats


REPO = Path(__file__).resolve().parents[1]
METHODS = ['CleanEval'] + [f'{profile}Adapt{method}' for profile in ('Fault', 'Clean')
    for method in ('TaskOnly', 'RND', 'DenseSurprise', 'SparseSurprise', 'FATE',
                   'AbsoluteSameShape')]
COLORS = dict(PALETTE, AbsoluteSameShape='#bf5360')
SHORT = {'DenseSurprise': 'Dense', 'SparseSurprise': 'Sparse surprise',
         'AbsoluteSameShape': 'Abs. same-shape'}
SUBTYPES = [
    ('remap_after_repeat_switch', 'Action\nremap'),
    ('reward_zero_after_repeat_switch', 'Reward\nzero'),
    ('reward_delay_after_two_rewards', 'Two-reward\ndelay'),
    ('revisit_action_delay', 'Revisit\ndelay'),
    ('delayed_switch_failure', 'Delayed\nswitch'),
    ('achievement_unlock_missing_after_valid_progress', 'Achievement\nomission'),
    ('delayed_inventory_desync_after_station_use', 'Inventory\ndesync'),
]


def label(method):
    if method == 'CleanEval':
        return 'CleanEval'
    return SHORT.get(core(method), core(method)) + (' | Clean' if method.startswith('CleanAdapt') else ' | Fault')


def color(method):
    return COLORS[core(method)]


def kind(method):
    return 'evaluation_slot_one_policy' if method == 'CleanEval' else 'adaptation_seed'


def policy_values(rows, method, split, metric):
    chosen = seeded(select(rows, method=method, split=split))
    repeats = 2 if split in ('holdout', 'sparse') else 1
    require(all(int(r['eval_repeats']) == repeats for r in chosen), 'Wrong nested evaluation count')
    return [float(r[metric]) for r in chosen]


def panels(fig, count, left=.245, right=.97, wspace=.2):
    return fig.subplots(1, count, squeeze=False,
        gridspec_kw={'left': left, 'right': right, 'top': .82, 'bottom': .23, 'wspace': wspace})[0]


def row_axes(ax, title, limit, show=True):
    hlabels(ax, [label(m) for m in METHODS], title, limit, show)
    for at in (.5, 6.5):
        ax.axhline(at, color='#c9d1d6', lw=.9)
    for i, method in enumerate(METHODS):
        if core(method) == 'FATE':
            ax.axhspan(i - .45, i + .45, color='#008779', alpha=.055, zorder=0)


def record_rows(f, name, ax, rows, split, metric, scale=1):
    for y, method in enumerate(METHODS):
        vals = np.array(policy_values(rows, method, split, metric)) * scale
        f.record(name, method, split + '/' + metric, vals, kind=kind(method))
        dotrow(ax, y, vals, color(method), '^' if method.startswith('CleanAdapt') else 'o')


def main_results(f, inputs):
    rows = inputs.csv('p0_per_policy.csv')
    summary = inputs.csv('p0_summary.csv')
    name = '01_p0_task_performance'
    fig = f.page('Game performance after adaptation',
        'P0 initialization | 5 adaptation seeds per method | points: seeds; large marker: mean +/- sample SD',
        ['Clean evaluation: 30K callbacks per policy; no intrinsic bonus paid.',
         'CleanEval shows five evaluation slots of one fixed policy, not five pretrained agents.'], size=(12.5, 8.5))
    ax = panels(fig, 1, right=.9)[0]
    row_axes(ax, 'Clean task return', (0, 4.65))
    record_rows(f, name, ax, rows, 'clean', 'task_return')
    baseline = float(one(summary, method='CleanEval', split='clean')['task_return_mean'])
    ax.axvline(baseline, color=MUTED, linestyle='--', linewidth=1, alpha=.65)
    for y, method in enumerate(METHODS):
        mean = float(one(summary, method=method, split='clean')['task_return_mean'])
        ax.text(1.02, y, f'{mean:.3f}', transform=ax.get_yaxis_transform(), va='center', fontsize=10)
    ax.set_xlabel('Episode task return')
    f.save(fig, name, 'P0 게임 수행 점수',
        '13조건 전체를 포함했다. Fault FATE의 평균이 높지만 P0 하나에서의 결과다. 점수에는 intrinsic bonus가 없다.',
        ['p0_per_policy.csv', 'p0_summary.csv'], '작은 점은 adaptation seed, CleanEval의 작은 점은 한 정책의 평가 슬롯이다.')

    name = '02_p0_fault_exposure'
    fig = f.page('Fault exposure across evaluation profiles',
        'P0 | all 13 conditions | mean +/- sample SD across five policy slots',
        ['Seen: one 60K evaluation/policy. Holdout and Sparse: mean of two separate 60K evaluations within each policy.',
         'Coverage = wrapper-recorded operator subtype applications, not independent error detection.'], size=(15, 8.5))
    for i, (ax, split) in enumerate(zip(panels(fig, 3, left=.205), ('seen', 'holdout', 'sparse'))):
        row_axes(ax, split.title(), (0, 103), i == 0)
        record_rows(f, name, ax, rows, split, 'coverage', 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel('Subtype coverage (%)')
    f.save(fig, name, 'Seen·Holdout·Sparse의 전체 비교',
        'Holdout/Sparse는 모든 비교군의 반복 평가까지 반영했다. FATE가 모든 profile에서 최고인 결과는 아니다.',
        ['p0_per_policy.csv'], '두 반복을 합친 발견 집합이 아니라 정책 안의 평균이다. 반복을 n=10 학습으로 세지 않는다.')

    name = '03_sparse_repeat_stability'
    fig = f.page('Sparse coverage changes when evaluation is repeated',
        'Same fixed policies | five policy slots | first vs repeated evaluation',
        ['Blue: first round; orange: repeat round. Left: mean coverage; right: intersection of discovered types across all five slots.',
         'The intersection is recomputed in each round. Neither panel reports the best seed or a pooled 120K union.'], size=(13.5, 8.5))
    axes = panels(fig, 2, left=.23, wspace=.28)
    for i, ax in enumerate(axes):
        row_axes(ax, ('Round-wise mean coverage', 'Types common to all five slots')[i], ((0, 100), (-.2, 7.2))[i], i == 0)
    for y, method in enumerate(METHODS):
        r = one(summary, method=method, split='sparse')
        first, repeat = [100 * float(r[k + '_coverage_mean']) for k in ('first', 'repeat')]
        axes[0].plot([first, repeat], [y, y], color='#a7afb5', lw=2)
        for j, (key, value, c, marker) in enumerate((('first', first, '#3972b4', 'o'), ('repeat', repeat, '#d27436', 's'))):
            axes[0].scatter(value, y, s=45, color=c, marker=marker, zorder=4)
            common = int(r[key + '_consistent_types'])
            axes[1].scatter(common, y + (-.13 if j == 0 else .13), s=48, color=c, marker=marker, zorder=4)
            f.summary(name, method, key + '_common_types', common, None, 5, 'across_slot_intersection')
            f.record(name, method, key + '_coverage_pct',
                     100 * np.array(policy_values(rows, method, 'sparse', key + '_coverage')), kind=kind(method))
    axes[0].set_xlabel('Coverage (%)')
    axes[1].set_xticks(range(8)); axes[1].set_xlabel('Common subtypes / 7')
    f.save(fig, name, 'Sparse 첫 평가와 반복 평가의 차이',
        'Fault FATE는 74.3%에서 54.3%로 바뀌었다. Clean FATE의 공통 유형은 4/4, Fault FATE는 3/2다.',
        ['p0_per_policy.csv', 'p0_summary.csv'], '공통 유형은 평균 coverage와 다른 지표다. 이 두 라운드만으로 모집단의 발견 확률을 확정하지 않는다.')


def behavior(f, inputs):
    rows = inputs.csv('p0_behavior_per_seed.csv')
    name = '04_p0_behavior_diversity'
    fig = f.page('Observed behavioral repertoire in Clean evaluation',
        'P0 | 30K callbacks per policy | five slots; mean +/- sample SD',
        ['Contexts compress inventory, front tile, achievement stage and nearby-object information; actions align to pre-transition contexts.',
         'These are observed combinations, not spatial/code coverage. Progression and observation frequency are not controlled here.'], size=(15, 8.5))
    for i, (ax, key, title, limit) in enumerate(zip(panels(fig, 3, left=.205),
            ('contexts', 'context_actions', 'transition_patterns'),
            ('Contexts', 'Context-action pairs', 'Transition patterns'),
            ((0, 160), (0, 1350), (0, 3100)))):
        row_axes(ax, title, limit, i == 0)
        ax.set_xticks(([0, 40, 80, 120, 160], [0, 300, 600, 900, 1200],
                      [0, 750, 1500, 2250, 3000])[i])
        for y, method in enumerate(METHODS):
            chosen = seeded(select(rows, method=method))
            require(all(int(r['steps']) == 30000 for r in chosen), 'Behavior budget mismatch')
            vals = [float(r[key]) for r in chosen]
            f.record(name, method, key, vals, kind=kind(method))
            dotrow(ax, y, vals, color(method))
        ax.set_xlabel('Unique observed combinations')
    f.save(fig, name, 'P0 행동 조합과 전이 패턴',
        'Fault FATE는 TaskOnly·RND·Dense보다 넓은 조합을 보였다. Clean SparseSurprise는 일부 지표에서 Clean FATE보다 높다.',
        ['p0_behavior_per_seed.csv'], '다양성 증가가 결함 노출 개선을 유발했다는 인과 증거는 아니다. P1에 대한 일반화도 아니다.')


def temporal(f, inputs):
    rewards = inputs.csv('p0_training_reward_per_seed.csv')
    policies = inputs.csv('p0_per_policy.csv')
    methods = [p + 'Adapt' + m for p in ('Clean', 'Fault') for m in ('FATE', 'AbsoluteSameShape')]
    name = '05_temporal_excess_ablation'
    fig = f.page('Removing temporal differencing at fixed ramp and cap',
        'Same beta = 0.2, threshold = 0.5, clip = 2.0 | five adaptation seeds per condition',
        ['FATE uses positive temporal increase; AbsoluteSameShape uses the absolute normalized surprise level.',
         'Realized mean bonus differs by about 21x: this is NOT a reward-budget-matched ablation. Error bars: sample SD.'], size=(15, 7.5))
    axes = panels(fig, 4, left=.21, wspace=.3)
    configs = [('task_return', 'Clean task return', 1, (0, 4.8)),
               ('bonus_mean', 'Mean bonus / transition', 1, (0, .043)),
               ('zero_fraction', 'Transitions paid bonus', 100, (0, 100)),
               ('movement_step_fraction', 'Movement action share', 100, (0, 100))]
    for i, (ax, (key, title, scale, lim)) in enumerate(zip(axes, configs)):
        hlabels(ax, [label(m) for m in methods], title, lim, i == 0)
        ax.axhline(1.5, color='#c9d1d6', lw=.9)
        for y, method in enumerate(methods):
            if key == 'task_return':
                vals = np.array(policy_values(policies, method, 'clean', key))
            else:
                vals = np.array([float(r[key]) for r in seeded(select(rewards, method=method))])
                if key == 'zero_fraction':
                    vals = 1 - vals
            vals *= scale
            f.record(name, method, key if key != 'zero_fraction' else 'positive_bonus_pct', vals)
            dotrow(ax, y, vals, color(method))
        ax.set_xlabel('% of non-reset transitions' if scale == 100 else 'Reward units')
    f.save(fig, name, '시간 차분 제거와 보상·행동·수행 변화',
        '차분 제거 조건의 평균 보너스는 약 21배, 이동 비중은 약 92%였다. 수행 점수는 두 적응 환경 모두 낮았다.',
        ['p0_training_reward_per_seed.csv', 'p0_per_policy.csv'], '계수를 고정했지만 실현 보상 강도는 다르다. 이동 비중만으로 무의미한 반복이나 원인을 확정하지 않는다.')


def cross(f, inputs):
    rows = inputs.csv('cross_primary_per_policy.csv')
    methods = ['CleanEval', 'TaskOnly', 'FATE_R0', 'FATE_R1']
    names = ['Initial policy', 'TaskOnly', 'FATE + R0', 'FATE + R1']
    name = '06_initialization_reference_cross'
    fig = f.page('Initialization and reference effects are not uniform',
        'Separate cross-reference cohort | Clean adaptation | three completed adaptation seeds per condition',
        ['Two pretrained checkpoints: P0 and P1. Clean and Sparse each average two evaluations within policy.',
         'Mean +/- sample SD; initial-policy dots are evaluation slots. The incomplete seed-3/4 extension is excluded.'], size=(13.5, 8.5))
    axes = fig.subplots(2, 2, gridspec_kw={'left': .16, 'right': .96, 'top': .79, 'bottom': .24, 'wspace': .35, 'hspace': .72})
    for i, prior in enumerate(('P0', 'P1')):
        for j, (split, metric, scale, title, lim) in enumerate((
                ('clean', 'task_return', 1, 'Clean return', (0, 4.6)),
                ('sparse', 'coverage', 100, 'Sparse coverage (%)', (0, 100)))):
            ax = axes[i, j]
            hlabels(ax, names, prior + ' | ' + title, lim)
            for y, method in enumerate(methods):
                chosen = seeded(select(rows, prior=prior, method=method, split=split), 3, 'adapt_seed')
                require(all(r['eval_repeats'] == '2' for r in chosen), 'Cross repeats mismatch')
                vals = [float(r[metric]) * scale for r in chosen]
                c = {'CleanEval': COLORS['CleanEval'], 'TaskOnly': COLORS['TaskOnly'], 'FATE_R0': '#008779', 'FATE_R1': '#3972b4'}[method]
                f.record(name, prior + '/' + method, metric, vals, kind=kind(method))
                dotrow(ax, y, vals, c)
    f.save(fig, name, 'P0/P1 시작 정책과 R0/R1 교차 결과',
        'P0에서는 평균 수행이 올랐지만 P1은 TaskOnly부터 낮아졌다. FATE의 TaskOnly 대비 이득도 초기 정책에 의존했다.',
        ['cross_primary_per_policy.csv'], '완료 3시드 결과다. P0 본 비교와 합치지 않았으며 하락의 원인을 reference 하나로 특정하지 않는다.')


def restore(f, inputs):
    rows = inputs.csv('joint_returns.csv')
    name = '07_joint_restore_pilot'
    fig = f.page('Joint restoration changes the continuation outcome',
        'TaskOnly pilot | ONE adaptation seed for P0 and ONE for P1',
        ['Ratio 128 -> 512, replay capacity 100K -> 200K, and stored Clean replay are restored together.',
         'Dots: two evaluation seeds of each fixed policy; black marker: their mean. No training-seed error bar.'], size=(12.5, 7.5))
    axes = panels(fig, 2, left=.10, wspace=.24)
    for ax, prior in zip(axes, ('P0', 'P1')):
        all_vals = []
        for x, (cohort, method, rounds) in enumerate((
                ('existing', 'CleanEval', ('base', 'base_repeat')),
                ('existing', 'TaskOnly', ('final', 'repeat_final')),
                ('joint_restored', 'TaskOnly', ('final', 'repeat_final')))):
            chosen = sorted([r for r in rows if r['prior'] == prior and r['cohort'] == cohort
                             and r['method'] == method and r['evaluation_round'] in rounds], key=lambda r: int(r['eval_seed']))
            require(len(chosen) == 2 and {r['eval_seed'] for r in chosen} == {'0', '20000'}, 'Pilot evaluations mismatch')
            vals = [float(r['task_return']) for r in chosen]
            all_vals.append(vals)
            mean, _ = f.record(name, prior + '/' + cohort + '/' + method, 'task_return', vals,
                              ids=[0, 20000], kind='evaluation_repeat_one_trained_policy')
            ax.scatter([x, x], vals, s=40, c=['#3972b4', '#d27436'], zorder=4)
            ax.scatter(x, mean, color=INK, marker='_', s=300, zorder=5)
            ax.text(x, max(vals) + .16, f'{mean:.3f}', ha='center', weight='bold')
        for j, c in enumerate(('#3972b4', '#d27436')):
            ax.plot(range(3), np.array(all_vals)[:, j], color=c, alpha=.35)
        ax.set_xticks(range(3), ['Initial', 'Existing\nTaskOnly', 'Joint-restored\nTaskOnly'])
        ax.set_xlim(-.4, 2.4); ax.set_ylim(0, 4.5)
        ax.set_title(prior, loc='left', weight='bold'); ax.set_ylabel('Clean task return'); axis(ax, 'y')
    f.save(fig, name, '공동 복원 TaskOnly 1시드 진단',
        '최종 두 평가 평균은 P0 3.102→3.396→3.913, P1 3.600→3.260→3.858이다.',
        ['joint_returns.csv'], '세 설정과 계산량이 함께 변했다. 평가 반복은 학습 시드가 아니며 개별 원인 또는 일반적인 회복을 증명하지 않는다.')


def wuji(f, inputs):
    policies = inputs.csv('p0_per_policy.csv')
    ports = ['DDQN_update1', 'DDQN_update8', 'A2C']
    data = {p: inputs.csv(f'wuji_{p}_per_seed.csv') for p in ports}
    dreamer = ['CleanEval', 'FaultAdaptRND', 'FaultAdaptFATE', 'CleanAdaptFATE']
    names = ['WUJI DDQN | update1', 'WUJI DDQN | update8', 'WUJI A2C',
             'Dreamer | CleanEval', 'Dreamer | RND, Fault', 'Dreamer | FATE, Fault', 'Dreamer | FATE, Clean']
    name = '08_wuji_external_comparison'
    fig = f.page('External testing-system comparison',
        'WUJI: 5 independent populations/port | Dreamer: P0, five adaptation/evaluation slots',
        ['WUJI budget: 2.1M actual actions/population, including fitness. Dreamer: about 1.1M pretraining + 1M adaptation callbacks.',
         'Different architectures, selection, compute and evaluation repetition. This is not a model-free vs world-model ablation.'], size=(15, 7.8))
    configs = [('clean', 'task_return', 1, 'Clean task return', (0, 4.8)),
               ('sparse', 'coverage', 100, 'Sparse coverage (%)', (0, 100)),
               ('sparse', 'events_per_10k', 1, 'Sparse events / 10K', (0, 21))]
    for j, (ax, (split, metric, scale, title, lim)) in enumerate(zip(panels(fig, 3, left=.25), configs)):
        hlabels(ax, names, title, lim, j == 0); ax.axhline(2.5, color='#a7afb5', lw=1)
        ax.set_xticks(([0, 1, 2, 3, 4], [0, 25, 50, 75, 100], [0, 5, 10, 15, 20])[j])
        for y, port in enumerate(ports):
            chosen = seeded(select(data[port], split=split))
            vals = [float(r[metric]) * scale for r in chosen]
            f.record(name, port, split + '/' + metric, vals, kind='independent_population_seed')
            dotrow(ax, y, vals, ['#888d92', '#b17e26', '#a45b86'][y], 's')
        for y, method in enumerate(dreamer, 3):
            vals = np.array(policy_values(policies, method, split, metric)) * scale
            f.record(name, method, split + '/' + metric, vals, kind=kind(method))
            dotrow(ax, y, vals, color(method), '^' if method.startswith('CleanAdapt') else 'o')
    f.save(fig, name, 'WUJI DDQN·A2C와 Dreamer 계열의 외부 비교',
        'WUJI 세 포트 모두 5 population 완료 결과다. Update8은 수행과 Sparse 평균 노출이 더 높았다. Dreamer의 높은 수행과 비슷한 노출 범위를 함께 보여준다.',
        [f'wuji_{p}_per_seed.csv' for p in ports] + ['p0_per_policy.csv'],
        'WUJI는 한 평가 라운드, Dreamer Sparse는 두 라운드의 정책별 평균이다. Population 대 단일 정책이므로 완전히 matched된 비교가 아니다.')


def opportunities(f, inputs):
    rows = inputs.csv('p0_sparse_repeat_opportunities.csv')
    name = '09_sparse_subtype_exposure'
    fig = f.page('Sparse subtypes differ greatly in reachability',
        'Common repeated-evaluation round | all 13 conditions | 5 slots each',
        ['Each cell counts evaluation slots with at least one recorded application, from 0 to 5. These are not event counts.',
         'The inventory subtype has zero eligible-context occurrences here; smoke-test executability does not imply policy reachability.'], size=(14, 9))
    ax = fig.add_axes([.25, .25, .65, .56])
    mat = np.zeros((len(METHODS), len(SUBTYPES)))
    for y, method in enumerate(METHODS):
        for x, (subtype, _) in enumerate(SUBTYPES):
            r = one(rows, method=method, subtype=subtype)
            require(int(r['n']) == 5, 'Subtype sample count mismatch')
            mat[y, x] = int(r['hit_seeds'])
            f.summary(name, method, subtype + '/hit_slots', mat[y, x], None, 5, 'observed_slot_count')
    im = ax.imshow(mat, cmap='YlGnBu', vmin=0, vmax=5, aspect='auto')
    for y in range(len(METHODS)):
        for x in range(len(SUBTYPES)):
            ax.text(x, y, str(int(mat[y, x])), ha='center', va='center', fontsize=11,
                    color='white' if mat[y, x] >= 3 else INK)
    ax.set_yticks(range(len(METHODS)), [label(m) for m in METHODS])
    ax.set_xticks(range(len(SUBTYPES)), [s[1] for s in SUBTYPES]); ax.tick_params(length=0)
    for at in (.5, 6.5): ax.axhline(at, color='white', lw=2)
    fig.colorbar(im, cax=fig.add_axes([.92, .25, .015, .56]), ticks=range(6), label='Slots exposed / 5')
    f.save(fig, name, 'Sparse 유형별 노출과 미도달 유형',
        'Inventory desync는 65평가 모두 미도달이다. Two-reward delay는 6/65평가에서 노출됐고, revisit delay는 65/65였다.',
        ['p0_sparse_repeat_opportunities.csv'], '발견하지 못한 유형을 분모에서 빼지 않는다. 집계는 공통 추가 Sparse 라운드에 한정한다.')


def drift(f, inputs):
    rows = inputs.csv('diagnostic_training_bins_summary.csv')
    name = '10_training_bonus_drift'
    fig = f.page('More shaping reward does not always mean better gameplay',
        'Earlier Clean-adaptation diagnostic cohort | FATE | 3 seeds per initialization',
        ['Lines: seed means; shading: sample SD across seeds. Training episodes are not fixed-policy evaluations.',
         'P1 bonus growth and task-return degradation co-occur, but these logs do not establish the causal direction.'], size=(14, 7.5))
    configs = [('mean_bonus', 1, 'Paid bonus / transition'),
               ('bonus_abs_task_ratio', 100, 'Bonus / absolute task reward (%)'),
               ('mean_task_episode_return', 1, 'Task-only episode return')]
    for ax, (key, scale, title) in zip(panels(fig, 3, left=.08, wspace=.33), configs):
        for prior, c in (('P0', '#008779'), ('P1', '#bf5360')):
            chosen = sorted(select(rows, prior=prior, method='FATE'), key=lambda r: int(r['bin']))
            require(len(chosen) == 10 and all(r['n'] == '3' for r in chosen), 'Diagnostic bins incomplete')
            x = np.arange(1, 11) / 10
            means = np.array([float(r[key + '_mean']) for r in chosen]) * scale
            sd = np.array([float(r[key + '_sd']) for r in chosen]) * scale
            ax.plot(x, means, color=c, marker='o', ms=4, lw=2, label=prior)
            ax.fill_between(x, means - sd, means + sd, color=c, alpha=.12)
            for r, mean, spread in zip(chosen, means, sd):
                f.summary(name, prior + '/bin' + r['bin'], key, mean, spread, 3, 'adaptation_seed')
        ax.set_title(title, loc='left', weight='bold'); ax.set_xlabel('Adaptation callbacks (millions)')
        ax.set_xlim(.08, 1.02); ax.set_xticks([.1, .3, .5, .7, 1.0]); ax.legend(frameon=False)
        ax.set_ylim(bottom=0); axis(ax, 'y')
    f.save(fig, name, 'P0/P1 학습 중 실제 보너스와 게임 점수',
        'P1 FATE에서 학습 후반 bonus가 증가했지만 순수 task return은 같이 증가하지 않았다. P0와 양상이 달랐다.',
        ['diagnostic_training_bins_summary.csv'], '교차 reference 실험과 별도의 3시드 진단이다. 보너스 포함 episode score를 평가 게임 점수처럼 해석하지 않는다.')


def finish(f, inputs):
    f.pdf.close()
    for name, rows in [('plotted_points.csv', f.points), ('plotted_summaries.csv', f.summaries)]:
        with (f.out / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    overview = Image.new('RGB', (1600, 570 * ((len(f.catalog) + 1) // 2)), 'white')
    draw = ImageDraw.Draw(overview)
    checks = []
    for i, item in enumerate(f.catalog):
        with Image.open(f.out / (item['name'] + '.png')) as im:
            ink = float(np.mean(np.asarray(im.convert('L')) < 230))
            require(.015 < ink < .7, 'Blank/overfilled figure')
            checks.append(dict(figure=item['name'], nonwhite_fraction=ink,
                               width=im.width, height=im.height, text_bounds='PASS'))
            im.thumbnail((780, 520))
            x, y = (i % 2) * 800, (i // 2) * 570
            overview.paste(im, (x, y + 32)); draw.text((x + 30, y + 8), item['name'], fill=INK)
    overview.save(f.out / 'overview.jpg', quality=94)
    for start in range(0, len(f.catalog), 4):
        overview.crop((0, (start // 2) * 570, 1600, min(((start + 4) // 2) * 570, overview.height))).save(
            f.out / f'preview_{start // 4 + 1:02d}.jpg', quality=95)
    lines = ['# FATE 최신 완료 결과 그래프', '',
        '2026-10-19 결과 스냅샷. CPU로 기존 CSV만 시각화했으며 새 학습·평가·큐 변경은 없다.', '',
        '[전체 PDF](fate_manuscript_all_figures.pdf) | [전체 미리보기](overview.jpg)', '',
        'PNG 240 dpi, 벡터 PDF, 편집 가능한 SVG를 각각 제공한다. 그림은 영문, 설명은 한글이다.', '',
        '## 먼저 볼 그림', '',
        '**01 수행 점수 → 02 결함 노출 → 03 반복 평가 → 04 행동 다양성 → 05 차분 제거** 순서가 주 결과다. '
        '06은 P0/P1 한계, 07은 복원 파일럿, 08은 WUJI 외부 비교, 09~10은 유형별 도달성과 보상 drift다.', '',
        '10월 15일의 첫 평가 중심 그림과 달리, Holdout/Sparse는 모든 비교군의 반복 결과를 반영했고 '
        'AbsoluteSameShape 및 WUJI A2C 완료 5시드를 포함했다.', '',
        '| 번호 | 그림 | 다운로드 |', '|---|---|---|']
    for i, item in enumerate(f.catalog, 1):
        name = item['name']
        lines.append(f'| {i:02d} | {item["title"]} | [PNG]({name}.png) / [PDF]({name}.pdf) / [SVG]({name}.svg) |')
    for i, item in enumerate(f.catalog, 1):
        lines += ['', f'## {i:02d}. {item["title"]}', '', f'![{item["title"]}]({item["name"]}.png)', '',
                  item['caption'], '', '**해석 주의:** ' + item['caution'], '',
                  '**원자료:** ' + ', '.join(f'[{s}](inputs/{s})' for s in item['sources'])]
    lines += ['', '## 집계 및 검증', '',
        '- 오차선/음영은 표본 SD이며 신뢰구간이나 SEM이 아니다. 유의성 표시는 없다.',
        '- P0 적응 5시드, 교차 3시드, 복원 1시드, WUJI 5 population을 서로 다른 단위로 표시했다.',
        '- Holdout/Sparse 반복은 정책 안에서 평균했다. 발견 집합의 union이나 독립 학습 10시드로 바꾸지 않았다.',
        '- Coverage는 wrapper-recorded operator exposure다. 자동 오류 진단 정확도나 독립적인 manifestation이 아니다.',
        '- `plotted_points.csv`와 `plotted_summaries.csv`에 표시 수치와 표본 단위를 저장했다.',
        '- 입력 snapshot SHA256 검증, 그림 텍스트 경계 및 빈 이미지 검사를 수행했다.', '',
        '재생성 코드: `dreamerv3/plot_fate_completed_results.py`. 기존 결과를 덮어쓰지 않도록 새 `--out` 경로를 사용한다.']
    (f.out / 'README.md').write_text('\n'.join(lines) + '\n')
    manifest = dict(sources=inputs.sources, figures=len(f.catalog), validation=checks,
        statistics='sample SD; nested evaluation means; cohorts not pooled',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        outputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in f.out.iterdir() if p.is_file()},
        no_training_or_rollout=True)
    (f.out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--materials', type=Path, default=REPO / 'analysis_outputs/fate_all_results_20261019')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), 'Choose a new output directory; existing results are immutable')
    (args.out / 'inputs').mkdir(parents=True)
    inputs = Inputs(args.materials, args.out)
    f = Figures(args.out)
    for plot in (main_results, behavior, temporal, cross, restore, wuji, opportunities, drift):
        plot(f, inputs)
    finish(f, inputs)
    print(f'Wrote {len(f.catalog)} figures to {args.out}', flush=True)


if __name__ == '__main__':
    main()
