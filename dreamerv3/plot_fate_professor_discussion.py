"""Plot completed FATE evidence without training, evaluation, or GPU imports."""

import argparse
import csv
import hashlib
import io
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/fate-professor-matplotlib')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.text import Text


METHODS = ['CleanEval', 'TaskOnly', 'RND-matched', 'DenseSurprise-matched',
           'FATE', 'Clean-adapt TaskOnly', 'Clean-adapt FATE']
WUJI = 'WUJI-derived replay8'
LABELS = ['CleanEval', 'TaskOnly | Fault', 'RND matched | Fault',
          'Dense matched | Fault', 'FATE | Fault', 'TaskOnly | Clean',
          'FATE | Clean', 'WUJI-derived | replay8']
COLORS = ['#74818c', '#a77635', '#377eb8', '#9367a4', '#008575',
          '#b19a69', '#007fa3', '#bf5964']
TEXT = '#25313b'
MUTED = '#5d6872'
OLD = 'fate_corrected_completed_20260929_5seed/all_completed_per_seed.csv'
BEHAVIOR = 'fate_behavior_diversity_20260929_5seed/clean_per_seed.csv'
CROSS = 'fate_reference_extension_20261012_v2/combined/primary_per_policy.csv'
PILOT = 'fate_joint_restore_20261014_v2/experiment/summary/paired_clean_returns.csv'
SYNTHESIS = 'fate_wuji_synthesis_20261014/inputs/evaluation_summary.csv'
WUJI_ROOTS = ['wuji_craftax_5seed_20261012_v2', 'wuji_replay8_5seed_20261014_v1']


def require(condition, message):
  if not condition:
    raise ValueError(message)


def stats(values):
  values = np.asarray(values, dtype=float)
  require(len(values) > 0 and np.isfinite(values).all(), 'Invalid observations')
  return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else None


def select(rows, **filters):
  return [r for r in rows if all(str(r.get(k)) == str(v) for k, v in filters.items())]


def one(rows, **filters):
  matches = select(rows, **filters)
  require(len(matches) == 1, f'Expected one row for {filters}, got {len(matches)}')
  return matches[0]


def seeded(rows, count=5, key='seed'):
  require(len(rows) == count, f'Expected {count} seed rows, got {len(rows)}')
  require({int(r[key]) for r in rows} == set(range(count)), 'Missing/duplicate seeds')
  return sorted(rows, key=lambda r: int(r[key]))


def subtype_sets(rows, key):
  sets = [set(r[key].split(';')) - {''} for r in rows]
  require(bool(sets), 'No subtype rows')
  for row, types in zip(rows, sets):
    require(len(types) == int(row['unique_types']), 'Subtype count mismatch')
    require(np.isclose(float(row['coverage']), len(types) / 7), 'Sparse denominator')
  return len(set.intersection(*sets)), len(set.union(*sets))


def completed_positions(data):
  values = []
  for rollout in data['rollouts']:
    count = len(rollout['completed_returns'])
    positions = rollout['segment_unique_positions']
    require(len(positions) == count + int(rollout['partial_return'] is not None),
            'Unexpected episode/position alignment')
    values.extend(positions[:count])
  require(bool(values), 'No completed episodes')
  return float(np.mean(values)), len(values)


class Snapshot:
  def __init__(self, base, out):
    self.base, self.out = base, out
    self.sources = {}

  def read(self, relative):
    path = self.base / relative
    raw = path.read_bytes()
    target = self.out / 'inputs' / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    self.sources[relative] = dict(source=str(path.resolve()),
                                 sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    return raw.decode('utf-8')

  def csv(self, relative):
    return list(csv.DictReader(io.StringIO(self.read(relative))))

  def json(self, relative):
    return json.loads(self.read(relative))


def load_data(snapshot):
  old = [r for r in snapshot.csv(OLD) if r['method'] in METHODS]
  behavior = snapshot.csv(BEHAVIOR)
  for method in METHODS:
    for split in ('clean', 'sparse'):
      rows = seeded(select(old, method=method, split=split))
      require(all(int(r['steps']) == (30000 if split == 'clean' else 60000)
                  for r in rows), 'Evaluation budget mismatch')
      if split == 'sparse':
        subtype_sets(rows, 'discovered_types')
    for b in seeded(select(behavior, method=method)):
      r = one(old, method=method, split='clean', seed=b['seed'])
      require(np.isclose(float(r['task_return']), float(b['task_return'])),
              'Behavior and outcome cohorts differ')
  wuji, spatial = [], []
  for root in WUJI_ROOTS:
    snapshot.json(f'{root}/COMPLETE.json')
    rows = snapshot.csv(f'{root}/results/per_seed.csv')
    snapshot.csv(f'{root}/results/summary.csv')
    wuji.append(rows)
    spatial_rows = []
    for split in ('clean', 'sparse'):
      subset = seeded(select(rows, split=split))
      if split == 'sparse':
        subtype_sets(subset, 'subtype_ids')
      for row in subset:
        budget = 30000 if split == 'clean' else 60000
        require(int(row['steps']) == budget, 'WUJI evaluation budget')
        require(np.isclose(float(row['events_per_10k']),
                           int(row['event_count']) * 10000 / budget), 'WUJI event rate')
        folder = 'eval_clean' if split == 'clean' else 'eval_fault'
        data = snapshot.json(f'{root}/seed_{row["seed"]}/{folder}/{split}.json')
        require(len(data['rollouts']) == 30, 'Expected WUJI population of 30')
        require(sum(r['steps'] for r in data['rollouts']) == budget, 'WUJI raw budget')
        mean, count = completed_positions(data)
        require(count == int(row['completed_episodes']), 'WUJI episode count')
        returns = [x for r in data['rollouts'] for x in r['completed_returns']]
        require(np.isclose(np.mean(returns), float(row['task_return'])), 'WUJI raw return')
        spatial_rows.append(dict(seed=row['seed'], split=split, positions=mean))
    spatial.append(spatial_rows)
  cross = snapshot.csv(CROSS)
  for prior in ('P0', 'P1'):
    for method in ('CleanEval', 'TaskOnly'):
      rows = [r for r in select(cross, prior=prior, method=method, split='clean')
              if int(r['adapt_seed']) < 3]
      seeded(rows, 3, 'adapt_seed')
      require(all(int(r['eval_repeats']) == 2 for r in rows), 'Cross repeat count')
  pilot = snapshot.csv(PILOT)
  for prior in ('P0', 'P1'):
    for cohort, method, evaluation_round in [
        ('existing', 'CleanEval', 'base'), ('existing', 'TaskOnly', 'final'),
        ('joint_restored', 'TaskOnly', 'final')]:
      row = one(pilot, prior=prior, cohort=cohort, method=method,
                evaluation_round=evaluation_round, adapt_seed=0, eval_seed=0)
      require(int(row['steps']) == 30000, 'Pilot evaluation budget')
  synthesis = snapshot.csv(SYNTHESIS)
  return old, behavior, wuji, spatial, cross, pilot, synthesis


class Figures:
  def __init__(self, out, stamp):
    self.out, self.stamp = out, stamp
    self.points, self.summaries, self.names = [], [], []
    self.pdf = PdfPages(out / 'fate_discussion_all_figures.pdf')

  def record(self, figure, group, metric, values):
    mean, sd = stats(values)
    for seed, value in enumerate(values):
      self.points.append(dict(figure=figure, group=group, metric=metric,
                              seed_slot=seed, value=float(value)))
    self.summaries.append(dict(figure=figure, group=group, metric=metric,
                              mean=mean, sample_sd=sd, n=len(values)))
    return mean, sd

  def aggregate(self, figure, group, metric, value, n):
    self.summaries.append(dict(figure=figure, group=group, metric=metric,
                              mean=value, sample_sd=None, n=n))

  def page(self, number, title, subtitle, footnotes):
    fig = plt.figure(figsize=(16, 9), facecolor='white')
    fig.text(.045, .95, title, fontsize=25, weight='bold', va='top', color=TEXT)
    fig.text(.045, .895, subtitle, fontsize=13.5, va='top', color=MUTED)
    fig.add_artist(plt.Line2D([.045, .955], [.852, .852], color='#d6dde1', lw=1))
    for i, line in enumerate(footnotes):
      fig.text(.045, .114 - i * .027, line, fontsize=11.5, color=MUTED, va='top')
    fig.text(.045, .025, f'FATE / Craftax  |  저장 결과 분석  |  {self.stamp}',
             fontsize=9.5, color=MUTED)
    fig.text(.955, .025, f'{number:02d}', ha='right', fontsize=10, color=MUTED)
    return fig

  def save(self, fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist in fig.findobj(match=Text):
      if not artist.get_visible() or not artist.get_text():
        continue
      box = artist.get_window_extent(renderer)
      require(box.x0 >= -1 and box.y0 >= -1 and
              box.x1 <= fig.bbox.x1 + 1 and box.y1 <= fig.bbox.y1 + 1,
              f'Text outside figure: {name}: {artist.get_text()}')
    fig.savefig(self.out / f'{name}.png', dpi=180)
    fig.savefig(self.out / f'{name}.pdf')
    self.pdf.savefig(fig)
    self.names.append(name)
    plt.close(fig)

  def finish(self):
    self.pdf.close()
    for filename, rows in [('plotted_seed_values.csv', self.points),
                           ('plotted_summaries.csv', self.summaries)]:
      with (self.out / filename).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def style_axis(ax, grid='x'):
  ax.set_axisbelow(True)
  ax.grid(axis=grid, color='#e7ebee', linewidth=.8)
  for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
  ax.spines['left'].set_color('#cbd2d8')
  ax.spines['bottom'].set_color('#cbd2d8')
  ax.tick_params(colors=TEXT, labelsize=12)


def horizontal(ax, groups, title, limit, labels=None, fmt='.2f', colors=COLORS):
  ax.set_title(title, loc='left', fontsize=16, pad=20, color=TEXT)
  for i, values in enumerate(groups):
    mean, sd = stats(values)
    ax.barh(i, mean, height=.53, color=colors[i], alpha=.19, edgecolor='none')
    if len(values) > 1:
      ax.scatter(values, i + np.linspace(-.13, .13, len(values)), s=21,
                 facecolor=colors[i], alpha=.65, edgecolor='white', linewidth=.4, zorder=3)
      ax.errorbar(mean, i, xerr=sd, fmt='D', color=colors[i], markersize=5,
                  capsize=4, lw=1.6, zorder=4)
    ax.text(limit * 1.025, i, format(mean, fmt), va='center', ha='left',
            fontsize=13.5, color=colors[i], weight='bold')
  ax.set_xlim(0, limit)
  ax.set_ylim(len(groups) - .45, -.65)
  ax.set_yticks(range(len(groups)), labels or [''] * len(groups))
  ax.tick_params(axis='y', length=0, pad=9)
  style_axis(ax)
  ax.spines['left'].set_visible(False)


def plot_main(figures, old, wuji):
  name = '01_gameplay_and_fault_exposure'
  fig = figures.page(1, '현재 결과: 게임 수행 능력과 결함 노출을 함께 비교',
      'P0 본 실험 5시드 + WUJI 외부 시스템 5시드 | 점: 개별 시드 슬롯 · 마름모: 평균 · 선: 표본 SD', [
      'Dreamer 적응은 같은 P0의 5개 학습 시드, CleanEval은 같은 정책의 평가 시드, WUJI는 5개 독립 집단이다.',
      'Fault/Clean은 추가학습 환경. 점수에는 내재적 보너스가 없다. WUJI 비교는 월드모델만 분리한 대조 실험이 아니다.',
      '발견율 = Sparse 7개 유형 중 oracle 발현 비율. 5시드 공통 유형은 교집합. 반복 평가 변동은 그림 06 참고.'])
  scores, coverages, common = [], [], []
  for method in METHODS + [WUJI]:
    source = wuji[1] if method == WUJI else select(old, method=method)
    clean = seeded(select(source, split='clean'))
    sparse = seeded(select(source, split='sparse'))
    scores.append([float(r['task_return']) for r in clean])
    coverages.append([100 * float(r['coverage']) for r in sparse])
    count, _ = subtype_sets(sparse, 'subtype_ids' if method == WUJI else 'discovered_types')
    common.append([count])
    for metric, values in [('clean_return', scores[-1]), ('sparse_coverage_percent', coverages[-1])]:
      figures.record(name, method, metric, values)
    figures.aggregate(name, method, 'common_subtypes_across_5', count, 5)
  axes = [fig.add_axes(rect) for rect in [( .225, .23, .205, .54),
                                          (.505, .23, .19, .54), (.775, .23, .14, .54)]]
  horizontal(axes[0], scores, 'Clean 게임 점수', 5, LABELS, '.3f')
  horizontal(axes[1], coverages, 'Sparse 발견율 (%)', 100, fmt='.1f')
  horizontal(axes[2], common, '5시드 공통 유형', 7, fmt='.0f')
  for ax in axes:
    ax.axhline(4.5, color='#cbd2d8', lw=.8, ls='--')
    ax.axhline(6.5, color='#cbd2d8', lw=.8, ls='--')
  axes[2].set_xticks([0, 3, 5, 7])
  figures.save(fig, name)


def plot_behavior(figures, behavior):
  name = '02_behavior_diversity'
  fig = figures.page(2, 'P0의 행동 다양성: 상황과 행동 조합이 얼마나 늘었나',
      '동일 Clean 평가 예산 30K callbacks | 5시드 평균, 표본 SD 및 개별 시드 | 공간 탐색 면적과는 다른 지표', [
      '상황 C = 인벤토리 구간 · 전방 타일 · 업적 단계 · 근처 몹 여부. 같은 P0에서 시작한 정책 비교다.',
      '다양성은 reset을 제외한 전체 관측 전이에서 계산. 빈도 균등화나 진행도 보정 전의 관측된 조합 수다.',
      '무효 행동도 조합 수를 늘릴 수 있으므로, 이 지표만으로 유의미한 탐색 또는 결함 탐지 우위를 확정하지 않는다.'])
  for j, (metric, label, limit) in enumerate([
      ('contexts', '관측한 상황 수', 150), ('context_actions', '상황별 행동 조합 수', 1300),
      ('transition_patterns', '전이 패턴 수', 3000)]):
    groups = []
    for method in METHODS:
      values = [float(r[metric]) for r in seeded(select(behavior, method=method))]
      figures.record(name, method, metric, values)
      groups.append(values)
    ax = fig.add_axes([.225 + j * .255, .23, .18, .54])
    horizontal(ax, groups, label, limit, LABELS[:7] if j == 0 else None, '.1f')
    ax.axhline(4.5, color='#cbd2d8', lw=.8, ls='--')
  figures.save(fig, name)


def paired(ax, groups, labels, title, limit, fmt='.2f', colors=('#74818c', '#bf5964')):
  ax.set_title(title, loc='left', fontsize=15, pad=18, color=TEXT)
  for values in zip(*groups):
    ax.plot(range(len(groups)), values, color='#b9c1c8', lw=.9, alpha=.8, zorder=1)
  for i, values in enumerate(groups):
    mean, sd = stats(values)
    ax.scatter(np.full(len(values), i), values, color=colors[i], s=32,
               edgecolor='white', linewidth=.5, zorder=3)
    ax.errorbar(i, mean, yerr=sd, fmt='D', color=colors[i], capsize=6,
                markersize=8, lw=2, zorder=4)
    ax.text(i, .95, format(mean, fmt), transform=ax.get_xaxis_transform(),
            ha='center', va='top', fontsize=15, color=colors[i], weight='bold')
  ax.set_xlim(-.5, len(groups) - .5)
  ax.set_ylim(0, limit)
  ax.set_xticks(range(len(groups)), labels)
  style_axis(ax, 'y')


def plot_wuji(figures, wuji, spatial):
  name = '03_wuji_update_comparison'
  fig = figures.page(3, 'WUJI-derived: 같은 상호작용에서 업데이트를 늘린 결과',
      '공개 DDQN 기반 포트, 각 5시드 | 2.1M actions 동일 · 버퍼 10K 동일 · 128 actions당 업데이트 1회 vs 8회', [
      '연결선은 같은 시드 번호. 마름모는 평균, 오차선은 표본 SD. 업데이트는 12,150회에서 97,200회로 증가했다.',
      '방문 위치 수는 완료 에피소드당 고유 위치의 평균이다. 에피소드 길이 보정값이나 전체 맵 커버리지 비율이 아니다.',
      '공통 유형은 5시드 교집합, 전체 유형은 합집합. A2C는 이 비교에 포함하지 않았다.'])
  specifications = [('clean', 'task_return', 'Clean 게임 점수', 2.8, '.3f'),
                    ('sparse', 'coverage', 'Sparse 발견율 (%)', 105, '.1f'),
                    ('sparse', 'events_per_10k', 'Sparse 결함 발생 / 1만 actions', 7, '.2f'),
                    ('clean', 'positions', 'Clean: 에피소드당 방문 위치', 8, '.2f'),
                    ('sparse', 'positions', 'Sparse: 에피소드당 방문 위치', 8, '.2f')]
  for j, (split, metric, title, limit, fmt) in enumerate(specifications):
    ax = fig.add_axes([.075 + (j % 3) * .31, .535 if j < 3 else .225, .245, .22])
    groups = []
    for i in range(2):
      rows = seeded(select(spatial[i] if metric == 'positions' else wuji[i], split=split))
      values = [float(r[metric]) * (100 if metric == 'coverage' else 1) for r in rows]
      figures.record(name, f'replay{1 if i == 0 else 8}', f'{split}_{metric}', values)
      groups.append(values)
    paired(ax, groups, ['replay1', 'replay8'], title, limit, fmt)
  ax = fig.add_axes([.695, .225, .245, .22])
  groups = [subtype_sets(seeded(select(rows, split='sparse')), 'subtype_ids') for rows in wuji]
  for i, counts in enumerate(groups):
    xpos = np.arange(2) + (i - .5) * .30
    ax.bar(xpos, counts, width=.27, color=['#74818c', '#bf5964'][i], label=f'replay{1 if i == 0 else 8}')
    for x, value, metric in zip(xpos, counts, ('common_subtypes_across_5', 'union_subtypes_across_5')):
      ax.text(x, value + .12, str(value), ha='center', fontsize=13, color=TEXT)
      figures.aggregate(name, f'replay{1 if i == 0 else 8}', metric, value, 5)
  ax.set_title('Sparse: 전체 발견 유형은 동일', fontsize=15, loc='left', pad=18)
  ax.set_xticks([0, 1], ['5시드 공통 유형', '5시드 전체 유형'])
  ax.set_ylim(0, 7)
  ax.legend(frameon=False, ncol=2, fontsize=10, loc='upper left')
  style_axis(ax, 'y')
  figures.save(fig, name)


def plot_priors(figures, cross):
  name = '04_p0_p1_taskonly_change'
  fig = figures.page(4, '확장 검증에서 나타난 문제: TaskOnly도 P0와 P1이 달랐다',
      '교차 실험의 추가학습 시드 0·1·2 | 각 정책의 평가 2회 평균 후 시드 평균 | 본 실험 5시드와 별도 집계', [
      '점은 시드 슬롯, 마름모는 평균, 선은 표본 SD. CleanEval의 슬롯은 같은 초기 정책에 대한 평가 반복이다.',
      'P0/P1은 같은 사전학습 설정의 서로 다른 초기 체크포인트. TaskOnly는 추가 보너스 없이 게임 보상으로 학습한다.',
      '하락은 관찰된 현상이며 원인 확정이 아니다. 초기 모델의 특성과 추가학습 전환 설정의 영향을 구분해야 한다.'])
  for j, prior in enumerate(('P0', 'P1')):
    groups = []
    for method in ('CleanEval', 'TaskOnly'):
      rows = seeded([r for r in select(cross, prior=prior, method=method, split='clean')
                     if int(r['adapt_seed']) < 3], 3, 'adapt_seed')
      values = [float(r['task_return']) for r in rows]
      figures.record(name, f'{prior}_{method}', 'clean_return', values)
      groups.append(values)
    change = np.mean(groups[1]) - np.mean(groups[0])
    ax = fig.add_axes([.09 + j * .47, .25, .35, .49])
    paired(ax, groups, ['추가학습 전', 'TaskOnly +1M'],
           f'{prior}  |  평균 변화 {change:+.3f}', 4.8, '.3f', ('#74818c', '#a77635'))
    ax.set_ylabel('Clean 게임 점수', fontsize=13)
  figures.save(fig, name)


def plot_restore(figures, pilot):
  name = '05_joint_restore_pilot'
  fig = figures.page(5, '원인 확인 중: 세 설정을 함께 복원한 TaskOnly 진단',
      '추가학습 시드 0, 평가 시드 0의 최종 1M 결과만 비교 | 각 조건 1회 평가 · 오차선 없음', [
      '기존 설정: ratio 128 / replay 100K / 빈 버퍼. 공동 복원: ratio 512 / replay 200K / 기존 clean 경험 승계.',
      '신규 환경 스텝은 모두 1M이지만, ratio 증가로 목표 학습량은 약 4배다. 저장 replay 복원은 완전한 무중단 재개와 다르다.',
      '두 초기 모델 모두 회복 방향인 초기 관찰. replay 하나의 인과 효과나 FATE의 개선 근거로 해석할 수 없다.'])
  for j, prior in enumerate(('P0', 'P1')):
    values = []
    for cohort, method, stage in [('existing', 'CleanEval', 'base'),
                                 ('existing', 'TaskOnly', 'final'),
                                 ('joint_restored', 'TaskOnly', 'final')]:
      row = one(pilot, prior=prior, cohort=cohort, method=method, evaluation_round=stage,
                adapt_seed=0, eval_seed=0)
      value = float(row['task_return'])
      values.append(value)
      figures.record(name, f'{prior}_{cohort}_{method}', 'clean_return', [value])
    ax = fig.add_axes([.09 + j * .47, .25, .35, .49])
    ax.bar(range(3), values, color=['#74818c', '#a77635', '#008575'], width=.58)
    ax.axhline(values[0], color='#74818c', lw=1, ls='--')
    for i, value in enumerate(values):
      ax.text(i, value + .10, f'{value:.3f}', ha='center', fontsize=17, color=TEXT, weight='bold')
    ax.set_title(f'{prior}  |  1시드 파일럿', loc='left', fontsize=18, pad=18)
    ax.set_xticks(range(3), ['추가학습 전', '기존 설정\nTaskOnly', '공동 복원\nTaskOnly'])
    ax.set_ylim(0, 5)
    ax.set_ylabel('Clean 게임 점수', fontsize=13)
    style_axis(ax, 'y')
  figures.save(fig, name)


def plot_repeats(figures, synthesis):
  name = '06_sparse_repeat_sensitivity'
  fig = figures.page(6, '함께 보여야 할 한계: 평가 묶음이 달라지면 발견율도 달라졌다',
      '같은 P0의 고정 정책, 추가학습 없이 Sparse 평가 시드 묶음 변경 | 각 묶음 5슬롯 평균 ± 표본 SD', [
      '기본·반복 평가가 모두 있는 세 조건만 표시. TaskOnly/RND/Dense의 반복 결과를 추정해서 채우지 않았다.',
      '평가 시드 묶음 변경에는 환경 궤적과 fault 실행의 확률성이 함께 포함된다. fault 추첨만의 효과를 분리한 결과가 아니다.',
      'FATE의 기본 평가 74.3%만으로 일관된 우위를 주장하지 않는다. 반복 평가는 새로운 학습 시드가 아니다.'])
  ax = fig.add_axes([.12, .25, .77, .49])
  methods = ['CleanEval', 'FaultAdaptFATE', 'CleanAdaptFATE']
  for i, round_ in enumerate(('original', 'repeat')):
    means, sds = [], []
    for method in methods:
      row = one(synthesis, prior='P0', evaluation_round=round_, method=method, split='sparse')
      require(int(row['n']) == 5, 'Repeat comparison count')
      mean, sd = float(row['coverage_mean']) * 100, float(row['coverage_sd']) * 100
      means.append(mean)
      sds.append(sd)
      figures.summaries.append(dict(figure=name, group=f'{method}_{round_}',
                                   metric='sparse_coverage_percent', mean=mean, sample_sd=sd, n=5))
    xpos = np.arange(3) + (i - .5) * .32
    ax.bar(xpos, means, yerr=sds, capsize=5, width=.28, color=['#007fa3', '#bf5964'][i],
           alpha=.85, label=['기본 평가', '반복 평가'][i], error_kw=dict(lw=1.3))
    for x, mean, sd in zip(xpos, means, sds):
      ax.text(x, mean + sd + 2, f'{mean:.1f}%', ha='center', fontsize=15, color=TEXT)
  ax.set_xticks(range(3), ['CleanEval', 'FATE | Fault-adapt', 'FATE | Clean-adapt'])
  ax.set_ylim(0, 105)
  ax.set_ylabel('Sparse 발견율 (%)', fontsize=14)
  ax.legend(frameon=False, loc='upper right', ncol=2, fontsize=13)
  style_axis(ax, 'y')
  figures.save(fig, name)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--base', type=Path, default=Path('analysis_outputs'))
  parser.add_argument('--out', type=Path, default=Path('analysis_outputs/fate_professor_figures_20261015'))
  args = parser.parse_args()
  args.out.mkdir(parents=True, exist_ok=True)
  stamp = datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M KST')
  plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False,
                       'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.size': 12})
  snapshot = Snapshot(args.base, args.out)
  old, behavior, wuji, spatial, cross, pilot, synthesis = load_data(snapshot)
  figures = Figures(args.out, stamp)
  plot_main(figures, old, wuji)
  plot_behavior(figures, behavior)
  plot_wuji(figures, wuji, spatial)
  plot_priors(figures, cross)
  plot_restore(figures, pilot)
  plot_repeats(figures, synthesis)
  figures.finish()
  manifest = dict(snapshot=stamp, figures=figures.names, sources=snapshot.sources,
                  training_launched=False, evaluation_launched=False,
                  dispersion='sample SD across seed slots, not CI',
                  pilot='adaptation seed 0 / evaluation seed 0 only',
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
  (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
  print(json.dumps(dict(output=str(args.out.resolve()), figures=len(figures.names),
                        sources=len(snapshot.sources), validation='PASS'), ensure_ascii=False))


if __name__ == '__main__':
  main()
