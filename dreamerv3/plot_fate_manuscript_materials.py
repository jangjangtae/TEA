"""Create manuscript figures from saved FATE evidence; no RL or GPU imports."""

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import textwrap

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/fate-manuscript-matplotlib')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image, ImageDraw

from dreamerv3.plot_fate_professor_discussion import one, require, seeded, select, stats


REPO = Path(__file__).resolve().parents[1]
MATERIALS = REPO / 'analysis_outputs/fate_manuscript_missing_materials_20261015'
METHODS = ['CleanEval'] + [f'{p}Adapt{m}' for p in ('Fault', 'Clean')
    for m in ('TaskOnly', 'RND', 'DenseSurprise', 'SparseSurprise', 'FATE')]
REPEATS = ['CleanEval', 'CleanAdaptTaskOnly', 'CleanAdaptFATE',
           'CleanAdaptSparseSurprise', 'FaultAdaptFATE', 'FaultAdaptSparseSurprise']
PALETTE = {'CleanEval': '#697681', 'TaskOnly': '#b17e26', 'RND': '#3972b4',
           'DenseSurprise': '#a45b86', 'SparseSurprise': '#cc6847', 'FATE': '#008779'}
INK, MUTED, GRID = '#24313a', '#596771', '#e4e9ed'


def core(method):
  for prefix in ('FaultAdapt', 'CleanAdapt'):
    if method.startswith(prefix):
      return method[len(prefix):]
  return method


def label(method):
  if method == 'CleanEval':
    return 'CleanEval'
  short = {'DenseSurprise': 'Dense', 'SparseSurprise': 'SparseSurprise'}.get(core(method), core(method))
  suffix = 'clean' if method.startswith('CleanAdapt') else 'fault'
  return f'{short} | {suffix}'


def color(method):
  return PALETTE[core(method)]


def marker(method):
  return 'D' if method == 'CleanEval' else '^' if method.startswith('CleanAdapt') else 'o'


def original_round(method):
  return 'basic' if method == 'FaultAdaptSparseSurprise' else 'original'


def repeat_average(rows):
  """Average fixed-budget evaluations inside each policy, never pool seed slots."""
  require(len(rows) == 2, 'Expected exactly two evaluations per policy')
  require(len({r['adapt_seed'] for r in rows}) == 1, 'Mixed adaptation seeds')
  require(len({r['eval_seed'] for r in rows}) == 2, 'Duplicate evaluation seed')
  return float(np.mean([float(r['task_return']) for r in rows]))


def common_count(rows):
  sets = [set(r['discovered_types'].split(';')) - {''} for r in rows]
  require(bool(sets), 'Empty subtype observations')
  return len(set.intersection(*sets))


class Inputs:
  def __init__(self, materials, out):
    self.materials, self.out = materials, out
    self.expected = json.loads((materials / 'manifest.json').read_text())['sources']
    self.sources, self.cache = {}, {}

  def csv(self, name, external=None):
    if name not in self.cache:
      path = external or self.materials / 'inputs' / name
      raw = path.read_bytes()
      digest = hashlib.sha256(raw).hexdigest()
      if external is None:
        require(digest == self.expected[name]['sha256'], f'Input changed: {name}')
      (self.out / 'inputs' / name).write_bytes(raw)
      self.sources[name] = dict(source=str(path.resolve()), sha256=digest)
      self.cache[name] = list(csv.DictReader(io.StringIO(raw.decode())))
    return self.cache[name]


class Figures:
  def __init__(self, out):
    self.out = out
    self.pdf = PdfPages(out / 'fate_manuscript_all_figures.pdf')
    self.catalog, self.points, self.summaries = [], [], []
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
        'axes.titlesize': 12, 'axes.labelsize': 11, 'axes.labelcolor': INK,
        'xtick.labelsize': 10, 'ytick.labelsize': 10, 'text.color': INK,
        'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'savefig.facecolor': 'white', 'axes.formatter.useoffset': False})

  def page(self, title, subtitle, notes, size=(10.8, 7.4)):
    fig = plt.figure(figsize=size, facecolor='white')
    fig.text(.045, .955, title, weight='bold', fontsize=17, va='top')
    fig.text(.045, .905, subtitle, fontsize=10.5, color=MUTED, va='top')
    lines = []
    for note in notes:
      lines.extend(textwrap.wrap(note, width=143))
    require(len(lines) <= 4, 'Too many footer lines')
    for i, line in enumerate(lines):
      fig.text(.045, .115 - i * .023, line, fontsize=8.4, color=MUTED, va='top')
    return fig

  def record(self, figure, group, metric, values, ids=None, kind='adaptation_seed'):
    mean, sd = stats(values)
    ids = list(range(len(values))) if ids is None else ids
    require(len(ids) == len(values), 'Point identity mismatch')
    for seed, value in zip(ids, values):
      self.points.append(dict(figure=figure, group=group, metric=metric,
          replicate=seed, replicate_kind=kind, value=float(value)))
    self.summary(figure, group, metric, mean, sd, len(values), kind)
    return mean, sd

  def summary(self, figure, group, metric, mean, sd, n, kind):
    self.summaries.append(dict(figure=figure, group=group, metric=metric,
        mean=mean, sample_sd=sd, n=n, replicate_kind=kind))

  def save(self, fig, name, korean, caption, sources, caution):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist in fig.findobj(match=Text):
      if artist.get_visible() and artist.get_text():
        box = artist.get_window_extent(renderer)
        require(box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.x1 + 1
                and box.y1 <= fig.bbox.y1 + 1, f'Clipped text in {name}: {artist.get_text()}')
    for extension in ('png', 'pdf', 'svg'):
      fig.savefig(self.out / f'{name}.{extension}', dpi=240)
    self.pdf.savefig(fig)
    plt.close(fig)
    self.catalog.append(dict(name=name, title=korean, caption=caption,
                             sources=sources, caution=caution))
    print('saved', name, flush=True)

  def finish(self, inputs):
    self.pdf.close()
    for name, values in [('plotted_points.csv', self.points), ('plotted_summaries.csv', self.summaries)]:
      with (self.out / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)
    canvas = Image.new('RGB', (1440, 510 * ((len(self.catalog) + 1) // 2)), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, item in enumerate(self.catalog):
      with Image.open(self.out / (item['name'] + '.png')) as im:
        im.thumbnail((710, 475))
        x, y = (i % 2) * 720, (i // 2) * 510
        canvas.paste(im, (x, y + 20))
        draw.text((x + 24, y + 4), item['name'], fill=INK)
    canvas.save(self.out / 'overview.jpg', quality=94)
    for start in range(0, len(self.catalog), 4):
      sheet = canvas.crop((0, (start // 2) * 510, 1440,
                           min(((start + 4) // 2) * 510, canvas.height)))
      sheet.save(self.out / f'preview_{start // 4 + 1:02d}.jpg', quality=95)
    lines = ['# FATE 논문 그림 초안', '',
        '2026-10-15에 고정한 기존 자료를 사용했다. 학습·평가·큐 변경 없이 CPU로 생성했다.', '',
        '[전체 PDF](fate_manuscript_all_figures.pdf) | [전체 미리보기](overview.jpg)', '',
        '개별 그림은 PNG(240 dpi), vector PDF, 편집 가능한 SVG로 제공한다. 그림 내 문구는 영문이며 아래 설명은 한글이다.', '',
        '## 읽는 순서', '',
        '01은 방법 구조, 02–09는 현재 결과와 대조 실험, 10–14는 세부 보상·원인 분석이다. '
        '이 그림 번호가 최종 논문 번호는 아니며 14장을 모두 본문에 넣을 필요는 없다.', '',
        '주장에 맞는 그림만 선택하되, 유리한 첫 평가만 남기고 반복 평가나 P1 결과를 생략하지 않는다. '
        'AbsoluteSameShape와 모든 baseline의 공통 반복 평가, A2C 5시드는 미완료여서 성능 그림을 만들지 않았다.', '',
        '## 그림 목록', '', '| 번호 | 그림 | 파일 |', '| --- | --- | --- |']
    for i, item in enumerate(self.catalog, 1):
      name = item['name']
      lines.append(f'| {i:02d} | {item["title"]} | [PNG]({name}.png) / [PDF]({name}.pdf) / [SVG]({name}.svg) |')
    for i, item in enumerate(self.catalog, 1):
      lines += ['', f'## {i:02d}. {item["title"]}', '', item['caption'], '',
          '**해석 주의:** ' + item['caution'], '',
          '**원자료:** ' + (', '.join(f'[{s}](inputs/{s})' for s in item['sources'])
                            or '기존 구현의 reward/replay/reference 경로를 요약한 구조도. 원고 보완 문서 3.2절 참고.')]
    lines += ['', '## 공통 지표 정의', '',
        '- Coverage는 wrapper의 operator-application exposure이며 독립 manifestation detector의 recall이 아니다.',
        '- Error bar와 음영은 표본 SD다. CI, SEM 또는 전이별 분산이 아니다. 유의성 표시는 하지 않았다.',
        '- CleanEval은 고정 정책의 평가 슬롯이다. Adaptation seed와 독립 clean pretraining seed를 구분한다.',
        '- 반복 평가는 정책 안에서 먼저 평균한다. 두 60K 평가의 합집합을 하나의 60K coverage로 쓰지 않는다.',
        '- 진단 학습 궤적의 return과 고정 정책 평가의 return을 섞지 않는다. 공동 복원은 모델당 1학습 seed다.',
        '- WUJI는 독립 population 5개다. Dreamer와 capacity·업데이트·실제 action/callback 예산이 완전히 matched된 비교가 아니다.', '',
        '## 재현과 검증', '',
        '`plotted_points.csv`는 그림에 사용한 seed/반복별 값, `plotted_summaries.csv`는 실제 표시한 집계값이다. '
        '그룹마다 통계 단위를 기록했다. `manifest.json`에 입력·출력 SHA256과 검사 결과를 저장했다.', '',
        '실행 스크립트: `dreamerv3/plot_fate_manuscript_materials.py`. '
        '재실행 시 기존 폴더를 덮어쓰지 않도록 새 `--out` 경로를 사용한다.']
    (self.out / 'README.md').write_text('\n'.join(lines) + '\n')
    validation = []
    for item in self.catalog:
      with Image.open(self.out / (item['name'] + '.png')) as im:
        gray = np.asarray(im.convert('L'))
        ink_fraction = float(np.mean(gray < 230))
        require(.015 < ink_fraction < .65, 'Unexpected blank/overfilled figure')
        validation.append(dict(figure=item['name'], width=im.width, height=im.height,
                               nonwhite_fraction=ink_fraction, text_bounds='PASS'))
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(self.out.iterdir()) if p.is_file()}
    manifest = dict(sources=inputs.sources, outputs=files, figures=len(self.catalog),
        validation=validation, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        status='Saved-result figures only; no learning or rollout',
        statistics='Sample SD; nested evaluation means; cohorts kept separate')
    (self.out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')


def axis(ax, grid='x'):
  ax.set_axisbelow(True)
  ax.grid(axis=grid, color=GRID, linewidth=.75)
  for side in ('left', 'bottom'):
    ax.spines[side].set_color('#c9d1d6')
  ax.tick_params(colors=INK)


def dotrow(ax, y, values, shade, symbol='o', annotate=False):
  mean, sd = stats(values)
  offsets = np.linspace(-.115, .115, len(values))
  ax.scatter(values, y + offsets, s=16, color=shade, alpha=.35, zorder=3)
  ax.errorbar(mean, y, xerr=sd, fmt=symbol, color=shade, ms=7,
              capsize=3, linewidth=1.8, zorder=4)
  if annotate:
    ax.text(.985, y, f'{mean:.3f}', transform=ax.get_yaxis_transform(),
            ha='right', va='center', fontsize=10, color=shade)


def hlabels(ax, labels, title, limit, show=True):
  ax.set_yticks(range(len(labels)), labels if show else [])
  ax.set_ylim(len(labels) - .5, -.5)
  ax.set_xlim(*limit)
  ax.set_title(title, loc='left', pad=13, weight='bold')
  axis(ax)


def p0_rows(data, method, split, repeat=False):
  return seeded(select(data, prior='P0', method=method, split=split,
      evaluation_round='repeat' if repeat else original_round(method)), 5, 'adapt_seed')


def workflow(f):
  name = '01_method_workflow'
  fig = f.page('Reusing a game-playing agent for testing',
      'Method schematic | the frozen reference supplies a shaping signal, not a bug verdict',
      ['Adaptation can use either a clean or fault-seeded environment; deployment uses a fixed policy.',
       'Fault labels are evaluation-only. Legacy exposure records denote operator applications, not independently verified failures.'])
  ax = fig.add_axes([.045, .18, .91, .65]); ax.axis('off')
  ax.set_xlim(0, 1); ax.set_ylim(0, 1)

  def box(x, y, w, h, title, body, shade):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=shade + '12', edgecolor=shade, lw=1.3))
    ax.text(x + w / 2, y + h - .045, title, ha='center', va='top', fontsize=11.5, weight='bold')
    ax.text(x + w / 2, y + h / 2 - .025, body, ha='center', va='center', fontsize=10.5, linespacing=1.5)

  def arrow(a, b, text=None):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=13, color=MUTED, lw=1.3))
    if text:
      vertical = np.isclose(a[0], b[0])
      left = vertical and text == 'score transitions'
      ax.text((a[0] + b[0]) / 2 + ((-.02 if left else .02) if vertical else 0),
              (a[1] + b[1]) / 2 + .025, text,
              ha=('right' if left else 'left') if vertical else 'center', va='bottom', fontsize=9, color=MUTED)

  box(.005, .58, .27, .35, '1  Clean pretraining',
      'Nominal gameplay\nDreamer world model + actor\nTask reward', PALETTE['TaskOnly'])
  box(.365, .58, .27, .35, '2  Policy adaptation',
      'Clean OR fault-seeded build\nCollect real transitions\nUpdate world model + actor', PALETTE['FATE'])
  box(.725, .58, .27, .35, '3  Fixed-policy evaluation',
      'Clean / Seen / Holdout / Sparse\nNo updates; no paid bonus\nReturn + behavior + exposure', PALETTE['RND'])
  arrow((.278, .77), (.362, .77), 'initialize')
  arrow((.638, .77), (.722, .77), 'deploy')
  box(.005, .04, .27, .33, 'Frozen clean reference',
      'Separate recurrent state\nObservation + requested action\nPosterior-prior KL / clean p95', PALETTE['DenseSurprise'])
  box(.365, .04, .27, .33, 'Temporal-excess bonus',
      '$d_t = \\max(z_t-z_{t-1}, 0)$\n$b_t = 0.2\\,\\mathrm{clip}((d_t-0.5)/1.5,0,1)$', PALETTE['FATE'])
  box(.725, .04, .27, .33, 'Learning reward path',
      '$r_t^{\\mathrm{delivered}} + b_t$ stored in replay\nReward head learns this target\nActor uses imagined reward', PALETTE['SparseSurprise'])
  arrow((.14, .58), (.14, .375), 'freeze')
  arrow((.278, .20), (.362, .20))
  arrow((.638, .20), (.722, .20))
  arrow((.50, .58), (.50, .375), 'score transitions')
  ax.plot([.86, .86, .62], [.375, .45, .45], color=MUTED, lw=1.3)
  arrow((.62, .45), (.62, .575))
  ax.text(.75, .46, 'learning target', ha='center', va='bottom', fontsize=9, color=MUTED)
  f.save(fig, name, '월드모델을 테스트 정책 적응에 재사용하는 구조',
      '게임 수행용 Dreamer에서 별도의 reference를 고정한다. 실제 수집 전이의 surprise 증가로 bonus를 계산해 replay reward에 더하고, 적응 모델의 reward head와 actor를 학습한다. 최종 정책은 보너스 없이 고정 평가한다.',
      [], '구조도는 실험 결과가 아니다. Frozen reference를 imagined rollout마다 다시 호출하는 구조나 자동 버그 판정기로 그리지 않았다.')


def main_p0(f, inputs):
  data = inputs.csv('all_evaluations.csv')
  aggregates = inputs.csv('p0_p1_evaluation_summary.csv')
  for method in METHODS:
    for split in ('clean', 'seen', 'holdout', 'sparse'):
      values = [float(r['task_return']) for r in p0_rows(data, method, split)]
      summary = one(aggregates, prior='P0', method=method, split=split,
                    evaluation_round=original_round(method))
      require(np.isclose(np.mean(values), float(summary['task_return_mean'])), 'Return cohort mismatch')
      require(np.isclose(np.std(values, ddof=1), float(summary['task_return_sd'])), 'SD mismatch')
  name = '02_p0_clean_return'
  fig = f.page('P0: task performance after adaptation',
      'Clean fixed-policy evaluation | five adaptation seeds per method; one shared clean initialization',
      ['Points: seed slots. Large markers and whiskers: mean +/- sample SD. CleanEval points are evaluation seeds of one fixed policy.',
       'Clean 30K callbacks; intrinsic bonuses are excluded. Circle: fault adaptation; triangle: clean adaptation.'])
  ax = fig.add_axes([.26, .21, .69, .61])
  for i, method in enumerate(METHODS):
    values = [float(r['task_return']) for r in p0_rows(data, method, 'clean')]
    f.record(name, method, 'clean_return', values, kind='evaluation_seed' if method == 'CleanEval' else 'adaptation_seed')
    dotrow(ax, i, values, color(method), marker(method), True)
  baseline = np.mean([float(r['task_return']) for r in p0_rows(data, 'CleanEval', 'clean')])
  ax.axvline(baseline, color=PALETTE['CleanEval'], lw=1, ls='--', alpha=.7)
  ax.axhline(.5, color=GRID); ax.axhline(5.5, color=GRID)
  hlabels(ax, [label(m) for m in METHODS], 'Task return', (0, 4.8))
  ax.set_xlabel('Mean completed-episode return')
  f.save(fig, name, 'P0 전체 11조건의 게임 수행 점수',
      '보상 경로 수정 후 P0의 전체 조건을 비교한다. 기존 원고에서 빠졌던 Fault SparseSurprise 및 Clean RND/Dense/SparseSurprise를 포함한다. FATE의 평균은 높지만 유의성 검정 결과를 표시한 그림은 아니다.',
      ['all_evaluations.csv', 'p0_p1_evaluation_summary.csv'], '하나의 clean 초기화에서 얻은 결과다. 높은 task return만으로 테스트 우위를 판정하지 않는다.')
  name = '03_p0_fault_coverage'
  fig = f.page('P0: fault-subtype exposure across profiles',
      'Original evaluation bundle | 60K callbacks per policy and profile',
      ['Exposure = wrapper-recorded operator applications. Points: seed slots; whiskers: sample SD.',
       'Seen has 8 eligible subtypes; Holdout/Sparse have 7. This is not an independent manifestation detector recall.'])
  grid = fig.add_gridspec(1, 3, left=.25, right=.97, bottom=.21, top=.82, wspace=.17)
  for col, split in enumerate(('seen', 'holdout', 'sparse')):
    ax = fig.add_subplot(grid[0, col])
    for i, method in enumerate(METHODS):
      values = [100 * float(r['coverage']) for r in p0_rows(data, method, split)]
      f.record(name, method, split + '_coverage_percent', values,
          kind='evaluation_seed' if method == 'CleanEval' else 'adaptation_seed')
      dotrow(ax, i, values, color(method), marker(method))
    hlabels(ax, [label(m) for m in METHODS], split.title(), (0, 104), col == 0)
    ax.set_xticks([0, 50, 100]); ax.set_xlabel('Coverage (%)')
    ax.axhline(.5, color=GRID); ax.axhline(5.5, color=GRID)
  f.save(fig, name, 'P0의 Seen·Holdout·Sparse 결함 연산 coverage',
      '같은 시작 모델의 전체 11조건을 각 fault profile에서 비교한다. Profile과 지표에 따라 우세 방법이 달라진다.',
      ['all_evaluations.csv'], '첫 평가 라운드다. Sparse에서 FATE가 보인 평균 우위는 반복 평가 그림 06과 함께 해석해야 한다.')
  name = '04_sparse_breadth_and_consistency'
  fig = f.page('Mean breadth and shared subtypes are different outcomes',
      'P0 Sparse, original evaluation bundle | five seed slots; seven eligible subtypes',
      ['Left: mean +/- sample SD of per-policy coverage. Right: intersection of discovered subtype sets across five slots.',
       'The intersection is a set statistic, not a mean or a maximum. No error bar is assigned to that single intersection.'])
  grid = fig.add_gridspec(1, 2, left=.25, right=.96, bottom=.21, top=.82, wspace=.22)
  left, right = [fig.add_subplot(grid[0, i]) for i in range(2)]
  for i, method in enumerate(METHODS):
    rows = p0_rows(data, method, 'sparse')
    values = [100 * float(r['coverage']) for r in rows]
    f.record(name, method, 'coverage_percent', values, kind='seed_slot')
    dotrow(left, i, values, color(method), marker(method))
    common = common_count(rows)
    summary = one(aggregates, prior='P0', method=method, split='sparse', evaluation_round=original_round(method))
    require(common == int(summary['consistent_types']), 'Intersection mismatch')
    f.summary(name, method, 'common_types', common, None, 5, 'intersection_across_seed_slots')
    right.barh(i, common, height=.53, color=color(method), alpha=.85)
    right.text(common + .14, i, str(common), va='center', fontsize=10)
  hlabels(left, [label(m) for m in METHODS], 'Mean subtype coverage', (0, 104))
  left.set_xlabel('Coverage (%)')
  hlabels(right, [label(m) for m in METHODS], 'Subtypes shared by all five slots', (0, 7.25), False)
  right.set_xticks(range(8)); right.set_xlabel('Intersection size / 7')
  f.save(fig, name, '평균 발견 범위와 공통 발견 유형의 차이',
      'Fault-adapt FATE의 첫 Sparse 평균 coverage는 74.3%지만 5시드 교집합은 3개다. RND와 Dense의 교집합은 각각 4개로, 평균 coverage와 재현 범위가 같은 순위를 주지 않는다.',
      ['all_evaluations.csv', 'p0_p1_evaluation_summary.csv'], '공통 유형은 평가 라운드마다 달라진다. 평균 coverage를 최고치로 해석하면 안 된다.')


def behavior(f, inputs):
  name = '05_behavior_diversity'
  rows = inputs.csv('behavior_per_seed.csv')
  methods = ['CleanEval', 'TaskOnly', 'RND-matched', 'DenseSurprise-matched',
             'FATE', 'Clean-adapt TaskOnly', 'Clean-adapt FATE']
  names = ['CleanEval', 'FaultAdaptTaskOnly', 'FaultAdaptRND', 'FaultAdaptDenseSurprise',
           'FaultAdaptFATE', 'CleanAdaptTaskOnly', 'CleanAdaptFATE']
  fig = f.page('P0: observed gameplay behavior',
      'Clean 30K evaluations | compressed context representation; five seed slots',
      ['Context = inventory bucket, front tile, achievement stage and nearby-mob indicator. Points: seed slots; whiskers: sample SD.',
       'These are observed context/transition counts, not spatial map coverage or code coverage. Progression and episode length can affect counts.'])
  grid = fig.add_gridspec(1, 3, left=.235, right=.97, bottom=.22, top=.81, wspace=.20)
  for col, (metric, title, limit) in enumerate([
      ('contexts', 'Distinct contexts', (0, 150)), ('context_actions', 'Context-action pairs', (0, 1250)),
      ('transition_patterns', 'Transition patterns', (0, 2900))]):
    ax = fig.add_subplot(grid[0, col])
    for i, (m, canonical) in enumerate(zip(methods, names)):
      group = seeded(select(rows, method=m))
      values = [float(r[metric]) for r in group]
      f.record(name, canonical, metric, values, kind='seed_slot')
      dotrow(ax, i, values, color(canonical), marker(canonical))
    hlabels(ax, [label(m) for m in names], title, limit, col == 0)
    ax.set_xlabel('Observed count')
  f.save(fig, name, '게임 상황·행동 조합·전이 패턴의 다양성',
      'P0 Clean 평가에서 FATE는 대조군보다 많은 상황별 행동 조합과 전이 패턴을 관측했다. Fault-adapt FATE context-action 평균은 970.8, TaskOnly 712.2, RND 847.2, Dense 767.2다.',
      ['behavior_per_seed.csv'], '압축 지표의 관측 다양성이다. 게임 진행 수준이 같다는 보장이나 결함 exposure 개선의 인과적 증거가 아니다.')


def repeats(f, inputs):
  name = '06_sparse_repeated_evaluation'
  data = inputs.csv('all_evaluations.csv')
  saved = inputs.csv('repeat_per_policy.csv')
  fig = f.page('Sparse coverage changes under repeated fixed-policy evaluation',
      'P0 | same learned policies, two evaluation bundles; 60K callbacks in each bundle',
      ['Left: mean coverage of each bundle. Right: average the two evaluations within each policy, then summarize five slots.',
       'Two evaluations are not ten training seeds. Their subtype union would use 120K callbacks and is not shown as 60K coverage.'])
  grid = fig.add_gridspec(1, 2, left=.26, right=.97, bottom=.23, top=.81, wspace=.28)
  a, b = [fig.add_subplot(grid[0, i]) for i in range(2)]
  for i, method in enumerate(REPEATS):
    first = p0_rows(data, method, 'sparse')
    repeat = p0_rows(data, method, 'sparse', True)
    values1 = [100 * float(r['coverage']) for r in first]
    values2 = [100 * float(r['coverage']) for r in repeat]
    f.record(name, method, 'first_coverage_percent', values1, kind='seed_slot')
    f.record(name, method, 'repeat_coverage_percent', values2, kind='seed_slot')
    x1, x2 = np.mean(values1), np.mean(values2)
    a.plot([x1, x2], [i, i], color=color(method), lw=2)
    a.scatter(x1, i, marker='o', color=color(method), s=55)
    a.scatter(x2, i, marker='s', facecolor='white', edgecolor=color(method), s=48, zorder=4)
    a.text(x1 + 1.2, i - .13, f'{x1:.1f}', color=color(method), fontsize=9, va='bottom')
    a.text(x2 - 1.2, i + .13, f'{x2:.1f}', color=color(method), fontsize=9, va='top', ha='right')
    average = [(u + v) / 2 for u, v in zip(values1, values2)]
    for seed, value in enumerate(average):
      require(np.isclose(value, 100 * float(one(saved, prior='P0', method=method, seed=seed)['coverage'])), 'Nested average mismatch')
    f.record(name, method, 'policy_average_coverage_percent', average, kind='policy_slot_two_eval_average')
    dotrow(b, i, average, color(method), marker(method))
  hlabels(a, [label(m) for m in REPEATS], 'First (filled) vs repeat (open)', (30, 90))
  hlabels(b, [label(m) for m in REPEATS], 'Within-policy average +/- SD', (30, 90), False)
  a.set_xlabel('Coverage (%)'); b.set_xlabel('Coverage (%)')
  f.save(fig, name, 'Sparse 반복 평가에 따른 결과 변동',
      'Fault-adapt FATE의 평균 coverage는 첫 74.3%에서 반복 54.3%로 바뀌었으며 정책별 두 평가 평균은 64.3%다. CleanEval의 동일 집계는 65.7%다. 완료된 여섯 조건을 모두 제시한다.',
      ['all_evaluations.csv', 'repeat_per_policy.csv'], 'RND/Dense 등 아직 공통 반복 묶음이 없는 조건의 최종 순위를 이 그림으로 결정하지 않는다.')


def cross(f, inputs):
  name = '07_initial_policy_reference_cross'
  rows = inputs.csv('cross_per_policy.csv')
  fig = f.page('Starting policy and reference dependence',
      'Balanced completed cross-design | three adaptation seeds per cell; two evaluations per policy',
      ['Points: policy-level evaluation averages; whiskers: sample SD across three slots. CleanEval is one fixed policy per prior.',
       'This is a separate cohort from the original P0 comparison. Reference labels include their respective calibration bundles.'])
  grid = fig.add_gridspec(2, 2, left=.11, right=.96, bottom=.23, top=.81, hspace=.55, wspace=.27)
  methods = ['CleanEval', 'TaskOnly', 'FATE_R0', 'FATE_R1']
  colors = [PALETTE['CleanEval'], PALETTE['TaskOnly'], PALETTE['FATE'], PALETTE['RND']]
  for rowid, prior in enumerate(('P0', 'P1')):
    for col, (split, metric, scale, ylim) in enumerate([
        ('clean', 'task_return', 1, (0, 4.5)), ('sparse', 'coverage', 100, (0, 100))]):
      ax = fig.add_subplot(grid[rowid, col])
      for i, (method, shade) in enumerate(zip(methods, colors)):
        subset = seeded(select(rows, prior=prior, method=method, split=split), 3, 'adapt_seed')
        require(all(int(r['eval_repeats']) == 2 for r in subset), 'Cross repeat count')
        values = [scale * float(r[metric]) for r in subset]
        mean, sd = f.record(name, f'{prior}/{method}', f'{split}_{metric}', values, kind='policy_slot_two_eval_average')
        ax.scatter(i + np.linspace(-.10, .10, 3), values, color=shade, alpha=.45, s=24)
        ax.errorbar(i, mean, yerr=sd, fmt='o', color=shade, capsize=4, lw=1.8, ms=7)
        ax.text(i, mean + sd + (0.12 if col == 0 else 3), f'{mean:.3f}' if col == 0 else f'{mean:.1f}', ha='center', fontsize=9)
      ax.set_xticks(range(4), ['CleanEval', 'TaskOnly', 'FATE + R0', 'FATE + R1'])
      ax.set_xlim(-.5, 3.5); ax.set_ylim(*ylim)
      ax.set_title(f'{prior} | ' + ('Clean task return' if col == 0 else 'Sparse coverage (%)'), loc='left', weight='bold')
      axis(ax, 'y')
  f.save(fig, name, 'P0/P1 시작 모델과 R0/R1 reference의 교차 비교',
      '각 셀의 완료 seed 0/1/2를 사용했다. TaskOnly는 P0에서 Clean 점수가 높아졌지만 P1에서는 낮아졌다. FATE의 TaskOnly 대비 이득도 두 시작 모델에서 달랐다.',
      ['cross_per_policy.csv'], '3시드 중간 결과다. R0/R1 차이를 reference 가중치만의 효과로 분리하지 않으며 P1 하락의 원인도 확정하지 않는다.')


def restoration(f, inputs):
  name = '08_joint_restore_pilot'
  rows = inputs.csv('joint_restore_returns.csv')
  fig = f.page('TaskOnly joint-restoration pilot',
      'One adaptation seed per starting policy | two final-policy evaluations, not two training seeds',
      ['Joint restoration: train ratio 128 -> 512, replay capacity 100K -> 200K, saved clean replay restored together.',
       'Open points: evaluation repeats; large points: their mean. No training-seed error bars. This is not a single-factor causal test.'])
  grid = fig.add_gridspec(1, 2, left=.09, right=.96, bottom=.25, top=.80, wspace=.23)
  conditions = [('existing', 'CleanEval', ('base', 'base_repeat')),
                ('existing', 'TaskOnly', ('final', 'repeat_final')),
                ('joint_restored', 'TaskOnly', ('final', 'repeat_final'))]
  colors = [PALETTE['CleanEval'], PALETTE['TaskOnly'], PALETTE['FATE']]
  for col, prior in enumerate(('P0', 'P1')):
    ax = fig.add_subplot(grid[0, col]); means = []
    for i, ((cohort, method, rounds), shade) in enumerate(zip(conditions, colors)):
      group = [r for r in select(rows, prior=prior, cohort=cohort, method=method, adapt_seed=0)
               if r['evaluation_round'] in rounds]
      group.sort(key=lambda r: int(r['eval_seed']))
      mean = repeat_average(group); means.append(mean)
      values = [float(r['task_return']) for r in group]
      f.record(name, f'{prior}/{cohort}/{method}', 'clean_return_eval_repeats', values,
          ids=[r['eval_seed'] for r in group], kind='evaluation_repeat_of_one_policy')
      ax.scatter(i + np.array([-.07, .07]), values, s=48, edgecolor=shade, facecolor='white', zorder=4)
      ax.scatter(i, mean, s=80, color=shade, zorder=5)
      ax.text(i, max(values) + .16, f'{mean:.3f}', ha='center', fontsize=12, color=shade)
    ax.plot(range(3), means, color='#bdc7cd', linewidth=1.2, zorder=1)
    ax.set_xticks(range(3), ['Initial\nCleanEval', 'TaskOnly\ncurrent protocol', 'TaskOnly\njoint restoration'])
    ax.set_xlim(-.45, 2.45); ax.set_ylim(0, 4.6); ax.set_title(prior, loc='left', weight='bold')
    ax.set_ylabel('Clean task return'); axis(ax, 'y')
  f.save(fig, name, 'TaskOnly 공동 복원 1시드 진단',
      '최종 1M 정책을 평가 seed 0/20000에서 평가한 평균을 사용한다. P0는 3.102→기존 TaskOnly 3.396→공동 복원 3.913, P1은 3.600→3.260→3.858이다. 이전 면담 그림의 첫 평가 값 대신 두 평가 평균을 썼다.',
      ['joint_restore_returns.csv'], '세 설정과 계산량을 함께 바꾼 1시드 파일럿이다. Replay만의 효과나 복원 TaskOnly 대 기존 FATE의 우위를 주장할 수 없다.')


def wuji(f, inputs):
  name = '09_wuji_update_budget'
  data = [inputs.csv(n) for n in ('wuji_replay1.csv', 'wuji_replay8.csv')]
  fig = f.page('WUJI-derived DDQN: increasing updates at fixed action budget',
      'Five independent populations per condition | 2.1M total actions per population; replay capacity stays at 10K',
      ['Updates per 128 actions: 1 vs 8. Total optimizer updates: 12,150 vs 97,200. Bars and whiskers: mean +/- sample SD.',
       'This is an external system baseline, not original WUJI reproduction or a matched world-model ablation.'])
  grid = fig.add_gridspec(1, 3, left=.085, right=.97, bottom=.25, top=.79, wspace=.36)
  shades = ['#71818e', PALETTE['SparseSurprise']]
  for col, (metric, splits, scale, title, limit) in enumerate([
      ('task_return', ['clean'], 1, 'Clean task return', (0, 2.0)),
      ('coverage', ['seen', 'holdout', 'sparse'], 100, 'Subtype coverage (%)', (0, 100)),
      ('events_per_10k', ['seen', 'holdout', 'sparse'], 1, 'Events / 10K actions', (0, 19))]):
    ax = fig.add_subplot(grid[0, col])
    for j, group in enumerate(data):
      means, sds = [], []
      for split in splits:
        r = one(group, split=split)
        mean, sd = float(r[metric + '_mean']) * scale, float(r[metric + '_sd']) * scale
        means.append(mean); sds.append(sd)
        f.summary(name, f'WUJI_replay{1 if j == 0 else 8}/{split}', metric, mean, sd, 5, 'independent_population')
      x = np.arange(len(splits)) + (j - .5) * .32
      ax.bar(x, means, yerr=sds, width=.30, color=shades[j], capsize=3,
             label=f'{1 if j == 0 else 8} update' + ('' if j == 0 else 's'))
    ax.set_xticks(range(len(splits)), [s.title() for s in splits], rotation=20 if col else 0)
    ax.set_ylim(*limit); ax.set_title(title, loc='left', weight='bold'); axis(ax, 'y')
    if col == 0:
      ax.legend(frameon=False, fontsize=10, loc='upper left')
  f.save(fig, name, 'WUJI 업데이트 보강 전후의 5시드 결과',
      'Replay 크기를 바꾸지 않고 업데이트 횟수를 늘린 실험이다. Clean return은 0.323에서 1.495, Sparse coverage는 51.4%에서 60.0%, Sparse events/10K는 1.87에서 4.13으로 증가했다.',
      ['wuji_replay1.csv', 'wuji_replay8.csv'], '5개 population의 SD다. WUJI A2C 결과는 포함하지 않으며, 낮은 점수만으로 model-free 전체를 평가하지 않는다.')


def reward_drift(f, inputs):
  name = '10_training_bonus_drift'
  external = REPO / 'analysis_outputs/fate_adaptation_diagnostics_20261006/training_bins_per_seed.csv'
  rows = inputs.csv('training_bins_per_seed.csv', external)
  fig = f.page('Realized bonuses change during adaptation',
      'Earlier P0/P1 Clean-adaptation diagnostic cohort | three seeds; non-reset transitions within 100K callback windows',
      ['Lines: means across three training seeds; shaded bands: sample SD. These are training-trajectory diagnostics, not fixed-policy scores.',
       'Probe matching does not enforce a constant on-policy bonus budget. The association does not identify the cause of return changes.'])
  grid = fig.add_gridspec(2, 2, left=.10, right=.96, bottom=.24, top=.79, hspace=.50, wspace=.27)
  for i, (metric, scale, title) in enumerate([
      ('mean_bonus', 1, 'Mean paid bonus'), ('bonus_positive_rate', 100, 'Positive-bonus transitions (%)'),
      ('cap_rate', 100, 'Cap-saturated transitions (%)'),
      ('bonus_abs_task_ratio', 100, 'Bonus / absolute task reward (%)')]):
    ax = fig.add_subplot(grid[i // 2, i % 2])
    for prior, shade in [('P0', PALETTE['RND']), ('P1', PALETTE['FATE'])]:
      means, sds = [], []
      for bin_id in range(10):
        group = seeded(select(rows, prior=prior, method='FATE', bin=bin_id), 3)
        values = [float(r[metric]) * scale for r in group]
        mean, sd = f.record(name, f'{prior}/FATE/bin{bin_id}', metric, values)
        means.append(mean); sds.append(sd)
      x = np.arange(1, 11) * 100
      ax.plot(x, means, marker='o', ms=3, color=shade, label=prior)
      # SD bounds may cross zero; the raw bands are kept, not silently clipped.
      ax.fill_between(x, np.array(means) - sds, np.array(means) + sds, color=shade, alpha=.13)
    ax.set_title(title, loc='left', weight='bold'); ax.set_xlabel('Adaptation callbacks (K)')
    ax.set_xticks([100, 400, 700, 1000]); axis(ax, 'y')
    if i == 0:
      ax.legend(frameon=False, ncol=2, fontsize=10)
  f.save(fig, name, '학습 중 실현 보너스의 변화',
      'P0/P1 각각의 기존 Clean-adapt FATE 3시드에서 100K 구간별 평균 지급 보너스, 양의 보너스 비율, cap 비율, 절댓값 task reward 대비 비율을 보여준다. P1의 후반 보너스 증가가 관측된다.',
      ['training_bins_per_seed.csv'], '교차 reference 실험과 별도의 3시드 진단이다. 시간에 따른 동반 변화가 성능 하락의 원인을 증명하지 않는다.')


def localization(f, inputs):
  name = '11_bonus_localization'
  data = inputs.csv('training_early_late.csv')
  fig = f.page('Bonus concentration on full-energy SLEEP requests',
      'Cross-reference diagnostic subset | last 100K callbacks; two adaptation seeds per cell',
      ['Bars: equal-weight means of two training seeds. Points: individual seeds. Visit share and bonus share use different denominators.',
       'These requests did not initiate sleep; time and other game state can still change. Concentration alone is not causal evidence of failure.'])
  ax = fig.add_axes([.10, .25, .86, .54])
  groups = [(p, r) for p in ('P0', 'P1') for r in ('FATE_R0', 'FATE_R1')]
  for j, (key, legend, shade) in enumerate([('visit_share', 'Share of transitions', PALETTE['RND']),
                                          ('bonus_share', 'Share of paid bonus', PALETTE['SparseSurprise'])]):
    for i, (prior, method) in enumerate(groups):
      rows = seeded(select(data, prior=prior, method=method, scope='late', group='full_energy_sleep'), 2)
      values = [100 * float(r[key]) for r in rows]
      mean, _ = f.record(name, f'{prior}/{method}', key + '_percent', values)
      x = i + (j - .5) * .32
      ax.bar(x, mean, color=shade, width=.30, label=legend if i == 0 else None, alpha=.85)
      ax.scatter(x + np.array([-.05, .05]), values, facecolors='white', edgecolors=shade, zorder=3, s=28)
      ax.text(x, max([mean] + values) + 1.5, f'{mean:.2f}', ha='center', fontsize=9)
  ax.set_xticks(range(4), [f'{p} / {m[-2:]}' for p, m in groups]); ax.set_ylim(0, 64)
  ax.set_ylabel('Share (%)'); ax.legend(frameon=False, ncol=2, loc='upper left'); axis(ax, 'y')
  f.save(fig, name, '특정 행동으로의 보너스 집중',
      'R1 조건에서 full-energy SLEEP 요청은 약 2–3%의 전이지만 약 34–40%의 보너스를 차지했다. 방문 비율과 보너스 비율을 나란히 표시하고 두 학습 seed의 점을 남겼다.',
      ['training_early_late.csv'], '이 분석은 완료된 2시드 subset이다. 전이가 완전히 무효라는 주장이나 reward hacking의 인과적 증명이 아니다.')


def offline(f, inputs):
  name = '12_identical_trajectory_rescoring'
  rows = select(inputs.csv('offline_reference_probe.csv'), group='full_energy_sleep')
  require(len(rows) == 3 and all(int(r['rows']) == 60 for r in rows), 'Offline probe sample')
  rows.sort(key=lambda r: int(r['rng_counter']))
  fig = f.page('Different reference bundles score the same requests differently',
      'Offline rescoring | 60 full-energy SLEEP requests from saved episode prefixes; three latent-RNG repeats',
      ['Each line connects the same latent-RNG repeat using R0 and R1. Reference recurrent states and clean p95 values are separate.',
       'This is a restricted saved sample, not three independently learned policies or an estimate of population-wide behavior frequency.'])
  ax = fig.add_axes([.17, .25, .66, .54])
  for i, r in enumerate(rows):
    values = [float(r['R0_g']), float(r['R1_g'])]
    ax.plot([0, 1], values, marker='o', lw=1.6, ms=7,
             color=['#3972b4', '#008779', '#cc6847'][i], label=f'Latent RNG {r["rng_counter"]}')
  for ref in ('R0', 'R1'):
    f.record(name, ref, 'pre_beta_g', [float(r[ref + '_g']) for r in rows],
             ids=[r['rng_counter'] for r in rows], kind='latent_rng_repeat_on_same_saved_sample')
  ax.set_xticks([0, 1], ['R0 + its calibration', 'R1 + its calibration'])
  ax.set_xlim(-.2, 1.2); ax.set_ylim(0, .30)
  ax.set_ylabel('Mean pre-beta shaping value g'); ax.legend(frameon=False, loc='upper left'); axis(ax, 'y')
  f.save(fig, name, '동일한 저장 궤적의 reference별 재채점',
      '같은 관측/action 요청을 사용해도 reference와 해당 calibration 묶음에 따라 SLEEP 요청의 pre-beta score가 달라진다. 세 선은 학습 seed가 아니라 latent RNG 반복이다.',
      ['offline_reference_probe.csv'], '가중치만의 차이를 분리한 실험이 아니며 선택된 60개 요청으로 전체 방문 빈도를 추정하지 않는다.')


def opportunities(f, inputs):
  name = '13_subtype_opportunities'
  rows = inputs.csv('subtype_opportunity_summary.csv')
  methods = ['CleanAdaptTaskOnly', 'CleanAdaptFATE']
  subtypes = ['remap_after_repeat_switch', 'reward_zero_after_repeat_switch',
      'reward_delay_after_two_rewards', 'revisit_action_delay', 'delayed_switch_failure',
      'achievement_unlock_missing_after_valid_progress', 'delayed_inventory_desync_after_station_use']
  labels = ['11  Remap after repeat', '14  Reward zero', '15  Reward delay',
            '18  Revisit action delay', '19  Delayed switch', '31  Achievement missing', '34  Inventory desync']
  fig = f.page('Separating fault opportunity from stochastic activation',
      'Earlier P1 Clean-adaptation cohort | Sparse; aggregate counts across adaptation seeds 0, 1, 2',
      ['Opportunities are counted only when that subtype is assigned. Left normalizes by assigned steps; middle/right show raw totals.',
       'No independent manifestation oracle is measured. Successful draws, injections and delayed releases are not interchangeable.'])
  grid = fig.add_gridspec(1, 3, left=.27, right=.96, bottom=.24, top=.79, wspace=.28)
  panels = [('draw_eligible', 'Eligible / 10K assigned', True),
            ('draw_success', 'Successful draws', False), ('new_injection', 'New injections', False)]
  for col, (metric, title, normalize) in enumerate(panels):
    ax = fig.add_subplot(grid[0, col])
    for j, method in enumerate(methods):
      shade = PALETTE['TaskOnly' if j == 0 else 'FATE']
      values = []
      for subtype in subtypes:
        r = one(rows, prior='P1', method=method, round='basic', split='sparse', subtype=subtype)
        require(int(r['n']) == 3, 'Opportunity seed scope')
        value = float(r[metric]) * (10000 / float(r['assigned_steps']) if normalize else 1)
        values.append(value)
        f.summary(name, method + '/' + subtype, title, value, None, 3, 'aggregate_counter_over_three_training_seeds')
      y = np.arange(7) + (j - .5) * .30
      ax.barh(y, values, color=shade, height=.27, label=core(method))
      for yi, value in zip(y, values):
        ax.text(value + .015 * max(max(values), 1), yi, f'{value:.0f}', va='center', fontsize=8)
    ax.set_yticks(range(7), labels if col == 0 else []); ax.invert_yaxis()
    ax.set_title(title, loc='left', fontsize=11, weight='bold'); ax.margins(x=.18); axis(ax)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True, prune='upper'))
    if col == 0:
      ax.legend(frameon=False, loc='lower right', fontsize=9)
  f.save(fig, name, 'Subtype별 기회와 추첨·주입의 분해',
      'P1 TaskOnly와 FATE의 Sparse 7개 subtype을 모두 표시한다. 기회는 해당 subtype 할당 step당 추첨 가능 횟수로 정규화했고, 추첨 성공과 새 주입은 3시드 합계다. 특정 유리하거나 불리한 subtype 하나만 고르지 않았다.',
      ['subtype_opportunity_summary.csv'], '원시 카운터 합계에는 통계 오차선을 붙이지 않는다. 동일 설정 확률이어도 실현 추첨 수는 다르며, eligibility와 coverage 동반 변화가 인과관계를 뜻하지 않는다.')


def beta(f, inputs):
  name = '14_beta_pilot'
  rows = inputs.csv('beta_pilot.csv')
  betas = ['0.05', '0.1', '0.2']
  fig = f.page('Corrected-protocol beta pilot',
      'P1 Clean adaptation | one training seed per setting; final 1M checkpoint only',
      ['Each point is a final fixed-policy evaluation. No training-seed uncertainty estimate is available.',
       'This pilot does not establish an optimal beta or replace percentile, dead-zone and clipping sensitivity studies.'])
  grid = fig.add_gridspec(1, 2, left=.10, right=.96, bottom=.25, top=.79, wspace=.27)
  a, b = [fig.add_subplot(grid[0, i]) for i in range(2)]
  for split, shade, symbol in [('clean', PALETTE['RND'], 'o'), ('sparse', PALETTE['FATE'], '^')]:
    values = []
    for beta_value in betas:
      r = one(rows, source='pilot', method=f'beta{beta_value}_final', split=split)
      require(int(r['adapt_seed']) == 0, 'Beta pilot seed')
      value = float(r['task_return']); values.append(value)
      f.record(name, 'beta' + beta_value, split + '_return', [value], kind='single_training_seed')
      if split == 'sparse':
        cov = 100 * float(r['coverage'])
        x = betas.index(beta_value)
        b.scatter(x, cov, color=shade, marker='^', s=75)
        b.text(x, cov + 3, f'{cov:.1f}', ha='center', fontsize=10)
        f.record(name, 'beta' + beta_value, 'sparse_coverage_percent', [cov], kind='single_training_seed')
    a.plot(range(3), values, marker=symbol, color=shade, label=split.title(), lw=1.5)
  for ax in (a, b):
    ax.set_xticks(range(3), betas); ax.set_xlabel('Bonus coefficient beta')
    ax.set_xlim(-.4, 2.4); axis(ax, 'y')
  a.set_ylim(0, 4); a.set_ylabel('Task return'); a.legend(frameon=False)
  b.set_ylim(0, 100); b.set_ylabel('Sparse subtype coverage (%)')
  f.save(fig, name, '수정 후 beta 1시드 파일럿',
      'P1 Clean-adapt FATE의 beta 0.05/0.10/0.20 최종 결과다. 이 실행에서는 보너스를 줄인 조건이 더 나은 게임 점수나 Sparse coverage를 보이지 않았다.',
      ['beta_pilot.csv'], 'n=1의 제한된 설정 비교다. 기존 보상 경로 수정 전 percentile sweep은 이 그림에 사용하지 않았다.')


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--materials', type=Path, default=MATERIALS)
  parser.add_argument('--out', type=Path,
      default=REPO / 'analysis_outputs/fate_manuscript_figures_20261015_v3')
  args = parser.parse_args()
  require(not args.out.exists(), f'Output already exists: {args.out}')
  args.out.mkdir(parents=True)
  (args.out / 'inputs').mkdir()
  inputs = Inputs(args.materials, args.out)
  f = Figures(args.out)
  workflow(f)
  main_p0(f, inputs)
  behavior(f, inputs)
  repeats(f, inputs)
  cross(f, inputs)
  restoration(f, inputs)
  wuji(f, inputs)
  reward_drift(f, inputs)
  localization(f, inputs)
  offline(f, inputs)
  opportunities(f, inputs)
  beta(f, inputs)
  require(len(f.catalog) == 14, 'Expected 14 figures')
  f.finish(inputs)
  print('Complete:', args.out, flush=True)


if __name__ == '__main__':
  main()
