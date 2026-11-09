"""Five publication figures from completed TEA cohorts; no rollout or learning."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st

import numpy as np
from PIL import Image
from matplotlib.ticker import MaxNLocator

from dreamerv3.plot_fate_tog_figures import (
    DOUBLE, STYLE, canvas, dot, key_handle, legend, plt, row_axis,
    setup_style, validate_layout, visible_text,
)


DATA = Path(__file__).resolve().parents[1]/'analysis_outputs'
T95 = 2.7764451051977987
BASE = ['TaskOnly', 'RND', 'DenseSurprise', 'SparseSurprise',
        'SparseMagnitude072', 'FATE', 'AbsoluteSameShape']
SHORT = ['Initial', 'TaskOnly', 'RND', 'Dense', 'Sparse',
         'Sparse x0.72', 'TEA', 'Abs. level']
METRICS = ['task_return', 'coverage', 'context_actions', 'contexts',
           'transition_patterns', 'events_per_10k', 'unique_types', 'mean_achievement_count']
EFFECT_PANELS = [('clean', 'task_return', 1, 'Clean task return'),
                 ('clean', 'context_actions', 1, 'Context-action count'),
                 ('sparse', 'coverage', 100, 'Sparse exposure (pp)')]


def require(ok, message):
  if not ok:
    raise ValueError(message)


def read_csv(path):
  with Path(path).open(encoding='utf-8-sig', newline='') as stream:
    return list(csv.DictReader(stream))


def sha(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numeric(value):
  value = float(value)
  require(math.isfinite(value), 'Nonfinite measurement')
  return value


def paired_interval(values):
  require(len(values) == 5 and all(math.isfinite(x) for x in values), 'Expected five finite seed pairs')
  mean, sd = st.mean(values), st.stdev(values)
  half = T95 * sd / math.sqrt(5)
  return mean, sd, mean-half, mean+half


def aggregate_evaluations(rows, methods, core=False):
  groups = defaultdict(list)
  for row in rows:
    groups[row['method'], row['split'], int(row['adapt_seed'])].append(row)
  expected = {(m, p, s) for m in methods for p in ('clean', 'holdout', 'sparse') for s in range(5)}
  require(set(groups) == expected, 'Missing or unexpected method/profile/seed')
  policies = {}
  for key, block in groups.items():
    method, split, seed = key
    repeats = 1 if core and split == 'clean' else 3
    require(len(block) == repeats, f'Incorrect repeat count: {key}')
    require(sorted(int(r['eval_seed']) for r in block) == [seed+10000*i for i in range(repeats)],
            f'Duplicate or mismatched evaluation seeds: {key}')
    require(all(int(r['steps']) == (30000 if split == 'clean' else 60000) for r in block),
            f'Budget mismatch: {key}')
    result = dict(method=method, split=split, adapt_seed=seed, repeats=repeats)
    for metric in METRICS:
      values = [numeric(r[metric]) for r in block if r.get(metric) not in ('', None)]
      if values:
        require(len(values) == repeats, f'Partial measurement: {key}/{metric}')
        result[metric] = st.mean(values)
    if split != 'clean':
      require(math.isclose(result['coverage'], result['unique_types']/7, abs_tol=1e-10),
              'Coverage denominator is not seven')
    policies[key] = result
  return policies


class Cohort:
  def __init__(self, name, root, core=False):
    self.name, self.root, self.core = name, root, core
    self.sources = []
    policies = self.read('per_policy.csv')
    methods = sorted({r['method'] for r in policies})
    require(len(methods) == {'core':15, 'replay':6, 'boundary':3}[name], 'Wrong cohort size')
    self.policies = aggregate_evaluations(self.read('evaluations.csv'), methods, core)
    require(len(policies) == len(self.policies), 'Wrong stored policy count')
    checks = 0
    for row in policies:
      key = row['method'], row['split'], int(row['adapt_seed'])
      actual = self.policies[key]
      require(int(row['repeats']) == actual['repeats'], 'Stored repeats mismatch')
      for metric in METRICS:
        if row.get(metric) not in ('', None):
          require(math.isclose(numeric(row[metric]), actual[metric], abs_tol=1e-9, rel_tol=1e-9),
                  f'Policy mean mismatch: {key}/{metric}')
          checks += 1
    for row in self.read('summary.csv'):
      for metric in METRICS:
        if row.get(metric+'_mean') not in ('', None):
          vals = self.values(row['method'], row['split'], metric)
          for suffix, actual in [('mean', st.mean(vals)), ('sd', st.stdev(vals))]:
            require(math.isclose(numeric(row[metric+'_'+suffix]), actual, abs_tol=1e-9, rel_tol=1e-9),
                    f'Summary mismatch: {row["method"]}/{metric}/{suffix}')
            checks += 1
    self.checks = checks
    if core:
      behavior = self.read('behavior_per_seed.csv')
      require(len(behavior) == 75, 'Incomplete core behavior cohort')
      require(len({(r['method'], r['seed']) for r in behavior}) == 75, 'Duplicate behavior row')
      for row in behavior:
        key = row['method'], 'clean', int(row['seed'])
        require(int(row['steps']) == 30000, 'Wrong behavior budget')
        require(math.isclose(numeric(row['task_return']), self.policies[key]['task_return'], abs_tol=2e-5),
                'Behavior and return policy mismatch')
        for metric in ('context_actions', 'contexts', 'transition_patterns'):
          self.policies[key][metric] = numeric(row[metric])
      self.bonuses = self.read('reward_per_seed.csv')
      require(len(self.bonuses) == 30, 'Incomplete reward statistics')
      require(len({(r['method'], r['seed']) for r in self.bonuses}) == 30, 'Duplicate bonus rows')

  def read(self, name):
    path = self.root/name
    self.sources.append(dict(path=str(path.resolve()), sha256=sha(path)))
    return read_csv(path)

  def values(self, method, split, metric):
    return [self.policies[method, split, seed][metric] for seed in range(5)]

  def bonus(self, method):
    rows = sorted((r for r in self.bonuses if r['method'] == method), key=lambda r:int(r['seed']))
    require([int(r['seed']) for r in rows] == list(range(5)), 'Missing bonus seed')
    require(all(0 < int(r['non_reset_steps']) < 1000000 for r in rows), 'Unexpected bonus denominator')
    return [numeric(r['bonus_mean'])*1000 for r in rows]


class Outputs:
  def __init__(self, out):
    self.out, self.points, self.summaries, self.catalog = out, [], [], []
    for directory in (out, out/'pdf', out/'data'):
      directory.mkdir(parents=True, exist_ok=True)
    setup_style()

  def record(self, figure, cohort, condition, split, metric, values, scale=1, effect=False):
    values = [float(v) for v in values]
    require(len(values) == 5, 'Expected five values per plotted series')
    for seed, value in enumerate(values):
      self.points.append(dict(figure=figure, cohort=cohort, condition=condition,
          split=split, metric=metric, seed=seed, value=value, scale=scale))
    mean, sd = st.mean(values), st.stdev(values)
    low, high = paired_interval(values)[2:] if effect else (mean-sd, mean+sd)
    self.summaries.append(dict(figure=figure, cohort=cohort, condition=condition,
        split=split, metric=metric, mean=mean, sample_sd=sd, lower=low, upper=high,
        interval='paired_t95_pointwise' if effect else 'sample_SD', n=5, scale=scale))

  def save(self, fig, name, caption):
    count = validate_layout(fig)
    require(not any(t.get_text().find('FATE') >= 0 for t in visible_text(fig)), 'Legacy display name')
    require(all(not ax.get_title() for ax in fig.axes) and fig._suptitle is None, 'Unwanted title')
    fig.savefig(self.out/f'{name}.png', dpi=600)
    fig.savefig(self.out/'pdf'/f'{name}.pdf')
    with Image.open(self.out/f'{name}.png') as image:
      ink = float(np.mean(np.asarray(image.convert('L')) < 230))
      require(.005 < ink < .45, 'Blank or overcrowded figure')
      pixels = list(image.size)
    self.catalog.append(dict(name=name, caption=caption, size_inches=list(fig.get_size_inches()),
        pixels=pixels, dpi=600, visible_texts=count, minimum_font_pt=9,
        layout_validation='passed', nonwhite_fraction=ink))
    plt.close(fig)
    print(f'SAVED {name}', flush=True)


def main_comparisons(out, core):
  for name, specs in [
      ('01_performance_behavior', [('clean','task_return',1,'Clean task return',(0,4.8),[0,1,2,3,4]),
                                   ('clean','context_actions',1,'Context-action count',(0,1450),[0,400,800,1200])]),
      ('02_operator_exposure', [('holdout','coverage',100,'Holdout subtype exposure (%)',(0,100),[0,25,50,75,100]),
                                ('sparse','coverage',100,'Sparse subtype exposure (%)',(0,100),[0,25,50,75,100])])]:
    fig, axes = canvas(2, height=3.9, left=1.08, top=.4, bottom=.58, gap=.36)
    legend(fig)
    for i, (ax, (split, metric, scale, xlabel, limits, ticks)) in enumerate(zip(axes,specs)):
      row_axis(ax, SHORT, limits, ticks, xlabel+f'\n({chr(97+i)})', i==0)
      ax.axhline(.5, color='#aaaaaa', lw=.6)
      entries = [('CleanEval',0,'Initial')]
      for y, method in enumerate(BASE,1):
        entries += [(p+'Adapt'+method, y+(-.16 if p=='Fault' else .16),p) for p in ('Fault','Clean')]
      for method,y,profile in entries:
        values = np.array(core.values(method,split,metric))*scale
        out.record(name,'core',method,split,metric,values,scale)
        dot(ax,y,values,STYLE[profile])
    common = ('Small points represent five adaptation seeds, and large symbols and bars show means and sample '
        'standard deviations. Initial denotes five evaluation slots of one fixed P0 policy, not five trained '
        'policies. Circles and open squares distinguish fault-seeded and clean adaptation. All adaptations '
        'start from P0. Dense, Sparse, Sparse x0.72, and Abs. level denote DenseSurprise, SparseSurprise, '
        'its coefficient-scaled variant, and AbsoluteSameShape. ')
    if name.startswith('01'):
      caption = ('Clean-evaluation task return and coarse context-action counts. Each policy has one 30K-callback '
          'Clean evaluation in this common all-method cohort. Return excludes intrinsic bonus; context-action '
          'counts are not spatial or code coverage. '+common)
    else:
      caption = ('Wrapper-recorded operator-subtype exposure in Holdout and Sparse profiles. Each point first '
          'averages three 60K-callback evaluations within one policy; coverage is the fraction of seven subtypes '
          'observed in each evaluation, not the union across repetitions. Exposure is not independently verified '
          'failure detection. '+common)
    out.save(fig,name,caption)


def reward_scale(out, core):
  name = '03_reward_scale'
  fig, axes = canvas(2,height=3.15,left=.62,right=.12,top=.4,bottom=.7,gap=.43)
  styles = [('SparseSurprise','#009E73','s','white','Sparse'),
            ('SparseMagnitude072','#CC79A7','^','white','Sparse x0.72'),
            ('FATE','#0072B2','o','#0072B2','TEA')]
  fig.legend(handles=[key_handle(c,m,f,l) for _,c,m,f,l in styles], loc='upper center',
      bbox_to_anchor=(.5,.995),ncol=3,frameon=False,handletextpad=.4,columnspacing=1.3)
  for i,(ax,profile) in enumerate(zip(axes,('Clean','Fault'))):
    ax.set_xlim(0,4.5)
    ax.set_ylim(2,4.7)
    ax.set_xticks([0,1,2,3,4])
    ax.set_yticks([2,3,4])
    ax.set_xlabel(r'Mean training bonus ($\times 10^{-3}$)'+f'\n({chr(97+i)}) {profile} adaptation')
    if i==0:
      ax.set_ylabel('Clean task return')
    ax.grid(color='#e3e3e3',lw=.5)
    ax.set_axisbelow(True)
    for method,color,marker,face,_ in styles:
      full = profile+'Adapt'+method
      x,y = core.bonus(full), core.values(full,'clean','task_return')
      require(min(x+[st.mean(x)-st.stdev(x)])>=0 and max(x+[st.mean(x)+st.stdev(x)])<=4.5,
              'Reward axis hides data')
      require(min(y+[st.mean(y)-st.stdev(y)])>=2 and max(y+[st.mean(y)+st.stdev(y)])<=4.7,
              'Return axis hides data')
      ax.scatter(x,y,s=9,color=color,marker=marker,alpha=.45,linewidths=.4)
      ax.errorbar(st.mean(x),st.mean(y),xerr=st.stdev(x),yerr=st.stdev(y),fmt=marker,
          color=color,mfc=face,ms=4.5,capsize=2,elinewidth=.8,mew=.8)
      out.record(name,'core',full,'train','bonus_mean',x,1000)
      out.record(name,'core',full,'clean','task_return',y)
  out.save(fig,name, 'Realized training bonus versus Clean-evaluation task return under clean and fault-seeded '
      'adaptation. Small symbols show paired measurements for each of five adaptation seeds; large symbols '
      'and horizontal/vertical bars show means and sample standard deviations. Bonus is averaged over non-reset '
      'training transitions within each run; return uses the common single 30K-callback Clean evaluation. '
      'The 0.72 coefficient was fixed using earlier observed bonus ratios, not an independent calibration probe. '
      'This is a reward-scale sensitivity comparison, not an exactly matched on-policy bonus-budget ablation.')


def effect_figure(out, cohort, name, comparisons, caption):
  fig, axes = canvas(3,height=2.65,left=1.38,right=.12,top=.14,bottom=.7,gap=.38)
  for i,(ax,(split,metric,scale,label)) in enumerate(zip(axes,EFFECT_PANELS)):
    all_values=[]
    blocks=[]
    for display,weights in comparisons:
      values = [sum(w*cohort.policies[m,split,seed][metric] for m,w in weights.items())*scale
                for seed in range(5)]
      mean,sd,low,high = paired_interval(values)
      blocks.append((values,mean,low,high))
      all_values.extend(values+[low,high,0])
      out.record(name,cohort.name,display,split,metric,values,scale,effect=True)
    low, high = min(all_values), max(all_values)
    pad = max((high-low)*.13, .01)
    row_axis(ax,[c[0] for c in comparisons],(low-pad,high+pad),[],
        r'$\Delta$ '+label+f'\n({chr(97+i)})',i==0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.axvline(0,color='#777777',ls='--',lw=.65,zorder=1)
    for y,(values,mean,lower,upper) in enumerate(blocks):
      ax.scatter(values,y+np.linspace(-.075,.075,5),s=9,color='#879396',alpha=.8,linewidths=0,zorder=2)
      ax.errorbar(mean,y,xerr=[[mean-lower],[upper-mean]],fmt='o',color='#0072B2',
          ms=4.3,capsize=2,elinewidth=.9,mew=.8,zorder=3)
  out.save(fig,name,caption+' Small gray points show the five paired seed differences; blue symbols and bars '
      'show mean differences and exploratory pointwise 95% paired t intervals (df=4), without multiplicity '
      'correction. Evaluations are averaged within each policy before pairing adaptation seeds. Each policy '
      'has three Clean (30K callbacks each), Holdout (60K), and Sparse (60K) evaluations; the panels display '
      'Clean task return, Clean context-action counts, and Sparse subtype exposure, respectively. Coverage '
      'differences are percentage points. All policies start from one P0 checkpoint. Intervals spanning zero '
      'do not establish equivalence. Seed pairing does not imply identical subsequent trajectories.')


def check_effects(out, replay, boundary):
  replay_path = replay.root.parent/'statistics/paired_effects.csv'
  previous = read_csv(replay_path)
  replay.sources.append(dict(path=str(replay_path.resolve()),sha256=sha(replay_path)))
  names = {'Empty replay':'fresh_TEA_minus_TaskOnly','Restored replay':'restored_TEA_minus_TaskOnly',
           'Interaction':'interaction_restored_gap_minus_fresh_gap'}
  boundary_rows = boundary.read('paired_effects.csv')
  pairs = {'TEA - TaskOnly':('FaultAdaptFATE','FaultAdaptTaskOnly'),
           'No-first - TaskOnly':('FaultAdaptTEANoFirstBonus','FaultAdaptTaskOnly'),
           'No-first - TEA':('FaultAdaptTEANoFirstBonus','FaultAdaptFATE')}
  checks=0
  for row in out.summaries:
    if row['cohort']=='replay':
      match=[r for r in previous if r['contrast']==names[row['condition']] and
             r['split']==row['split'] and r['metric']==row['metric']]
    elif row['cohort']=='boundary':
      a,b=pairs[row['condition']]
      match=[r for r in boundary_rows if r['method_a']==a and r['method_b']==b and
             r['split']==row['split'] and r['metric']==row['metric']]
    else:
      continue
    require(len(match)==1,'Missing reference effect')
    for field,source in [('mean','mean_difference'),('lower','t95_low'),('upper','t95_high')]:
      require(math.isclose(row[field],numeric(match[0][source]),abs_tol=1e-9,rel_tol=1e-9),
              f'Effect mismatch: {row["condition"]}/{row["metric"]}/{field}')
      checks+=1
  return checks


def finish(out, cohorts, checks):
  for filename,rows in [('plotted_seed_values.csv',out.points),('plotted_statistics.csv',out.summaries)]:
    with (out.out/'data'/filename).open('w',newline='',encoding='utf-8-sig') as stream:
      writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
      writer.writeheader()
      writer.writerows(rows)
  doc = ['TEA final paper figures', '',
      'Five figures only. PNG: 600 dpi; vector PDF: pdf/. No learning or evaluation was run.',
      'All figures are 7.16 inches wide; keep this two-column width for 9 pt labels.',
      '01-03 use the common all-method cohort: Clean 1 repeat; Holdout/Sparse 3 repeats.',
      '04-05 use matched 3-repeat cohorts. Do not substitute these return values into 01 selectively.',
      'Figures 01-03: sample SD. Figures 04-05: exploratory paired t 95% CI (n=5 pairs, df=4).',
      'No independent pretraining-seed generalization, no universal fault-detection superiority.',
      'P1, WUJI and opportunity/oracle diagnostics remain separate results, not pooled here.',
      'All three repetition-averaged exposure measures keep the denominator of seven subtypes.',
      'Source data are not changed. CSVs in data/ retain internal FATE IDs only as provenance.', '',
      'Suggested captions (English):', '']
  for item in out.catalog:
    doc.extend([item['name'],item['caption'],''])
  (out.out/'CAPTIONS.txt').write_text('\n'.join(doc)+'\n')
  sources=[r for cohort in cohorts for r in cohort.sources]
  source_file=Path(__file__)
  sources.append(dict(path=str(source_file.resolve()),sha256=sha(source_file)))
  manifest=dict(figures=out.catalog,sources=sources,cohort_checks={c.name:c.checks for c in cohorts},
      paired_effect_checks=checks,raw_transition_logs_rechecked=False,
      scope='Reaggregated completed per-evaluation CSVs, checked against policy and summary CSVs; no new RL.')
  manifest['outputs']={str(p.relative_to(out.out)):sha(p) for p in sorted(out.out.rglob('*'))
                       if p.is_file() and p.name!='verification.json'}
  (out.out/'data/verification.json').write_text(json.dumps(manifest,indent=2)+'\n')


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--out',type=Path,default=DATA/'tea_paper_figures_final_20261028')
  parser.add_argument('--core',type=Path,default=DATA/'tea_final_results_20261023')
  parser.add_argument('--replay',type=Path,default=DATA/'tea_replay_results_handoff_20261026/results')
  parser.add_argument('--boundary',type=Path,default=DATA/'tea_no_first_bonus_20261027_5seed_v1/results')
  args=parser.parse_args()
  core=Cohort('core',args.core,True)
  replay=Cohort('replay',args.replay)
  boundary=Cohort('boundary',args.boundary)
  for method in ('FaultAdaptFATE','FaultAdaptTaskOnly'):
    for split in ('clean','holdout','sparse'):
      for metric in METRICS:
        if metric in replay.policies[method,split,0]:
          require(np.allclose(replay.values(method,split,metric),boundary.values(method,split,metric),
                              atol=1e-10,rtol=1e-10),'Common baseline differs across ablations')
  out=Outputs(args.out)
  main_comparisons(out,core)
  reward_scale(out,core)
  effect_figure(out,replay,'04_replay_sensitivity',[
      ('Empty replay',{'FaultAdaptFATE':1,'FaultAdaptTaskOnly':-1}),
      ('Restored replay',{'FaultRestoredTEA':1,'FaultRestoredTaskOnly':-1}),
      ('Interaction',{'FaultRestoredTEA':1,'FaultRestoredTaskOnly':-1,'FaultAdaptFATE':-1,'FaultAdaptTaskOnly':1})],
      'Replay-initialization sensitivity in fault-seeded adaptation. The first two rows show TEA minus '
      'TaskOnly under empty and restored clean-experience replay, respectively. Interaction is the restored '
      'gap minus the empty-replay gap. Replay capacity (100K), train ratio (128), P0 agent/optimizer '
      'initialization and 1M new-callback budget are fixed. This is not a full pretraining resume and does '
      'not diagnose the earlier P1 effect.')
  effect_figure(out,boundary,'05_first_transition_ablation',[
      ('TEA - TaskOnly',{'FaultAdaptFATE':1,'FaultAdaptTaskOnly':-1}),
      ('No-first - TaskOnly',{'FaultAdaptTEANoFirstBonus':1,'FaultAdaptTaskOnly':-1}),
      ('No-first - TEA',{'FaultAdaptTEANoFirstBonus':1,'FaultAdaptFATE':-1})],
      'First-transition bonus ablation in fault-seeded adaptation. No-first suppresses only the bonus on '
      'the first non-reset transition of each episode; its normalized surprise score is retained for the '
      'next temporal difference. All other reward rules and the 1M-callback adaptation protocol are '
      'unchanged. All five adaptation seeds, including the final two completed runs, are included.')
  checks=check_effects(out,replay,boundary)
  finish(out,[core,replay,boundary],checks)
  print(f'VERIFIED {checks} interval fields; EXPORTED {args.out}',flush=True)


if __name__=='__main__':
  main()
