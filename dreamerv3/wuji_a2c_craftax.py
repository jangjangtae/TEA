"""WUJI-inspired A2C Craftax port, not an original-game reproduction.

SB3 supplies A2C/GAE/timeout handling; upstream WUJI supplies NSGA-II and
single-point crossover. Retains P, mutated P-prime, and trained Q for selection.
"""

import argparse
import configparser
import copy
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import random
import time
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import stable_baselines3 as sb3
from stable_baselines3.common.logger import configure
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
from torch import nn

from dreamerv3 import wuji_craftax as w


@dataclass(frozen=True)
class Config:
    seed: int = 0
    population: int = 30
    train_profile: str = 'train'
    max_env_steps: int = 2100000
    learn_steps: int = 7400
    fitness_steps: int = 3000
    fitness_rollouts: int = 3
    fitness_seed: int = 100000
    episode_length: int = 10000
    n_steps: int = 20
    lr: float = 0.0007
    discount: float = 0.99
    gae_lambda: float = 1.0
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    mutation_probability: float = 0.01
    mutation_std: float = 0.005
    device: str = 'cuda:0'
    min_free_gib: float = 20.0

    def validate(self):
        w.require(self.population >= 3, 'Population must be at least three.')
        w.require(self.train_profile in ('clean', 'train'), 'No held-out training/selection.')
        for key in ('max_env_steps', 'learn_steps', 'fitness_steps', 'fitness_rollouts',
                    'episode_length', 'n_steps'):
            w.require(getattr(self, key) > 0, f'{key} must be positive.')
        w.require(self.learn_steps % self.n_steps == 0, 'A2C rollout would overrun action budget.')
        w.require(self.fitness_steps >= self.fitness_rollouts, 'Empty fitness rollout.')
        w.require(min(self.seed, self.fitness_seed) >= 0, 'Negative seed.')
        w.require(all(math.isfinite(v) for v in (
            self.lr, self.discount, self.gae_lambda, self.ent_coef, self.vf_coef,
            self.max_grad_norm, self.mutation_probability, self.mutation_std, self.min_free_gib)),
            'Nonfinite hyperparameter.')
        w.require(self.lr > 0 and 0 <= self.discount <= 1 and 0 <= self.gae_lambda <= 1,
                  'Invalid A2C parameters.')
        w.require(self.ent_coef >= 0 and self.vf_coef >= 0 and self.max_grad_norm > 0,
                  'Invalid loss parameters.')
        w.require(0 <= self.mutation_probability <= 1 and self.mutation_std >= 0,
                  'Invalid mutation parameters.')
        w.require(self.min_free_gib >= 0, 'Invalid free-space margin.')
        accounting(self)


def accounting(c):
    initial = c.population * c.fitness_steps
    cost = c.population * (c.learn_steps + 2 * c.fitness_steps)
    generations, remainder = divmod(c.max_env_steps - initial, cost)
    w.require(generations >= 1 and remainder == 0, 'Budget must fit whole P/P-prime/Q generations.')
    return dict(generations=generations,
        train_steps=generations * c.population * c.learn_steps,
        fitness_steps=initial + 2 * generations * c.population * c.fitness_steps,
        updates=generations * c.population * (c.learn_steps // c.n_steps),
        candidates=c.population * (1 + 2 * generations))


class Features(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=256)
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3), nn.ReLU(), nn.Flatten(),
            nn.Linear(1024, 256), nn.ReLU())

    def forward(self, observations):
        return self.net(observations)


OBS_SPACE = gym.spaces.Box(0, 255, (3, 64, 64), dtype=np.uint8)
ACTION_SPACE = gym.spaces.Discrete(17)
POLICY_KWARGS = dict(features_extractor_class=Features, net_arch=[],
                     normalize_images=True, ortho_init=True)


class Actor(nn.Module):
    def __init__(self, config=None, policy=None):
        super().__init__()
        self.policy = policy if policy is not None else ActorCriticCnnPolicy(
            OBS_SPACE, ACTION_SPACE, lambda _: 0.0007, **POLICY_KWARGS)

    def forward(self, image):
        # Shared rollout utilities provide [0,1]; SB3 preprocesses [0,255].
        return self.policy.get_distribution(image * 255).distribution.logits

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        # SB3 registers the shared encoder three times. Evolve it only once.
        return {k: v for k, v in state.items() if not k.startswith((
            'policy.pi_features_extractor.', 'policy.vf_features_extractor.'))}

    def load_state_dict(self, state, strict=True, **kwargs):
        state = dict(state)
        for key, value in list(state.items()):
            if key.startswith('policy.features_extractor.'):
                suffix = key.removeprefix('policy.features_extractor.')
                for prefix in ('pi_features_extractor', 'vf_features_extractor'):
                    state[f'policy.{prefix}.{suffix}'] = value
        return super().load_state_dict(state, strict=strict, **kwargs)

    def group(self):
        groups = {}
        for key in self.state_dict():
            groups.setdefault(key.rsplit('.', 1)[0], []).append(key)
        return list(groups.values())


class GymAdapter(gym.Env):
    observation_space = OBS_SPACE
    action_space = ACTION_SPACE

    def __init__(self, c, seed, env_factory=w.CraftaxAdapter):
        self.adapter = env_factory(c.train_profile, seed, c.episode_length)
        self.actions = self.resets = self.terminations = self.truncations = 0
        self.reward_sum = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.resets += 1
        return self.adapter.reset()['image'].transpose(2, 0, 1).copy(), {}

    def step(self, action):
        obs = self.adapter.step(int(action))
        reward = w.delivered_reward(obs)
        terminal = bool(obs['is_terminal'])
        last = bool(obs['is_last'])
        w.require(not terminal or last, 'Terminal transition without episode end.')
        truncated = last and not terminal
        self.actions += 1
        self.reward_sum += reward
        self.terminations += int(terminal)
        self.truncations += int(truncated)
        # No position, raw reward, or fault oracle enters A2C's observation/info.
        return obs['image'].transpose(2, 0, 1).copy(), reward, terminal, truncated, {}

    def close(self):
        self.adapter.close()


class AuditedA2C(sb3.A2C):
    def train(self):
        super().train()
        w.require(all(p.grad is not None and torch.isfinite(p.grad).all()
                      and torch.isfinite(p).all() for p in self.policy.parameters()),
                  'Missing/nonfinite A2C gradient or weight.')
        for key in ('train/policy_loss', 'train/value_loss', 'train/entropy_loss'):
            w.require(math.isfinite(self.logger.name_to_value[key]), 'Nonfinite A2C loss.')


def train_individual(c, individual, env_factory=w.CraftaxAdapter):
    env = GymAdapter(c, individual['train_seed'], env_factory)
    before = w.blob_digest(individual['blob'])
    try:
        learner = AuditedA2C('CnnPolicy', env, learning_rate=c.lr, n_steps=c.n_steps,
            gamma=c.discount, gae_lambda=c.gae_lambda, ent_coef=c.ent_coef,
            vf_coef=c.vf_coef, max_grad_norm=c.max_grad_norm, use_rms_prop=True,
            rms_prop_eps=1e-5, normalize_advantage=False,
            policy_kwargs=POLICY_KWARGS, seed=individual['train_seed'], device=c.device)
        learner.set_logger(configure(folder=None, format_strings=[]))
        actor = Actor(policy=learner.policy)
        actor.load_state_dict(individual['blob'])
        learner.learn(c.learn_steps)
        w.require(env.actions == c.learn_steps == learner.num_timesteps, 'A2C action overrun.')
        w.require(learner._n_updates == c.learn_steps // c.n_steps, 'Missing A2C updates.')
        individual['blob'] = w.clone_blob(actor)
        individual['age'] += c.learn_steps
        individual['lineage_train_steps'] += c.learn_steps
        return dict(steps=env.actions, resets=env.resets, updates=learner._n_updates,
            task_reward_sum=env.reward_sum, terminations=env.terminations,
            truncations=env.truncations, weights_changed=before != w.blob_digest(individual['blob']),
            policy_loss=float(learner.logger.name_to_value['train/policy_loss']),
            value_loss=float(learner.logger.name_to_value['train/value_loss']),
            entropy_loss=float(learner.logger.name_to_value['train/entropy_loss']))
    finally:
        env.close()


def mutate(blob, rng, probability, std):
    result, changed = {}, 0
    for key, value in blob.items():
        value = value.detach().cpu().clone()
        mask = rng.random(tuple(value.shape)) < probability
        noise = rng.normal(0, std, tuple(value.shape)).astype(np.float32)
        value += torch.from_numpy(noise * mask)
        result[key] = value
        changed += int(mask.sum())
    return result, changed


class Evolution(w.Evolution):
    def __init__(self, c):
        from wuji.ea.crossover.blob.pth.single_point import Split
        from wuji.ea.selection.non_dominated.nsga_ii import Selection
        self.c = c
        self.config = configparser.ConfigParser()
        w.require(self.config.read(w.UPSTREAM / 'config.ini'), 'Missing upstream config.')
        context = {'encoding': {'blob': {'module': [Actor], 'init': [{'kwargs': {}}]}}}
        self.blob_cross = Split(self.config, SimpleNamespace(context=context, get_kind=lambda: 0))
        self.selection = Selection(self.config)

    def offspring(self, population, count, rng, new_individual):
        from wuji.ea.mating.non_dominated.nsga_ii import NSGA_II
        from wuji.ea.selection.non_dominated.nsga_ii import crowding_distance
        self.selection(self.candidates(population), len(population))
        candidates = []
        for front in self.selection.layers:
            crowding_distance(front, lambda result: result['point'])
            candidates.extend(front)
        mating = NSGA_II(self.config, candidates)
        for operator in (mating, self.blob_cross):
            operator.random.seed(int(rng.integers(2**31)))
        children = []
        while len(children) < count:
            parents = [p['source'] for p in mating(2)]
            for index, blob in enumerate(self.blob_cross(*(p['blob'] for p in parents))):
                if len(children) == count:
                    break
                blob, changed = mutate(blob, rng, self.c.mutation_probability, self.c.mutation_std)
                child = new_individual(blob, 'mutated')
                child.update(parents=[p['id'] for p in parents], mutated_parameters=changed,
                             age=parents[index]['age'])
                # Mixed-weight lineage has no single exact training age.
                child['lineage_train_steps'] = max(p['lineage_train_steps'] for p in parents)
                children.append(child)
        return children


def source_identity():
    identity = w.source_identity()
    for relative in ('dreamerv3/wuji_a2c_craftax.py', 'tools/run_wuji_a2c.py'):
        identity['hashes'][relative] = hashlib.sha256((w.ROOT / relative).read_bytes()).hexdigest()
    package = Path(sb3.__file__).parent
    identity['sb3_files'] = {str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(package.rglob('*.py'))}
    identity.update(method='WUJI-inspired A2C', stable_baselines3=sb3.__version__,
                    gymnasium=gym.__version__, torch=torch.__version__, numpy=np.__version__)
    return identity


class Experiment(w.Experiment):
    def __init__(self, config, outdir, resume=False, env_factory=w.CraftaxAdapter):
        config.validate()
        w.prepare_device(config.device)
        self.config, self.outdir, self.env_factory = config, Path(outdir), env_factory
        self.identity = source_identity()
        self.evolution = Evolution(config)
        self.rng = np.random.default_rng(config.seed)
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.checkpoint = self.outdir / 'checkpoint.pt'
        if resume:
            payload = torch.load(self.checkpoint, map_location='cpu', weights_only=False)
            w.require(payload['schema'] == w.SCHEMA and payload['config'] == asdict(config),
                      'Checkpoint configuration mismatch.')
            w.require(payload['identity'] == self.identity, 'Checkpoint code/dependency mismatch.')
            self.state = payload['state']
            self.rng.bit_generator.state = payload['rng']
            torch.set_rng_state(payload['torch_rng'])
        else:
            w.require(not any(self.outdir.iterdir()), 'Output must be empty; use --resume.')
            self.state = dict(stage='initial', generation=0, cursor=0, next_id=0,
                population=[], pending=[], history=[], train_steps=0, fitness_steps=0,
                resets=0, updates=0, pretrain=[], trained=[], selection_history=[])
            for _ in range(config.population):
                self.state['pending'].append(self.new_individual(w.clone_blob(Actor()), 'initial'))
            self.save('initialized')

    def new_individual(self, blob, origin):
        index = self.state['next_id']
        self.state['next_id'] += 1
        return dict(id=index, blob=blob, origin=origin, age=0, parents=[], lineage_train_steps=0,
                    train_seed=int(self.rng.integers(2**31)))

    def save(self, status):
        super().save(status)
        import json
        path = self.outdir / 'summary.json'
        value = json.loads(path.read_text())
        value['selection_history'] = self.state['selection_history']
        value['runtime']['stable_baselines3'] = sb3.__version__
        w.atomic_json(path, value)

    def run(self, stop_after_candidates=None):
        completed = 0
        while True:
            s, c = self.state, self.config
            if s['stage'] == 'ready':
                cost = c.population * (c.learn_steps + 2 * c.fitness_steps)
                if self.total_steps + cost > c.max_env_steps:
                    w.require(self.total_steps == c.max_env_steps, 'Incomplete final budget.')
                    self.save('budget_complete')
                    return
                s['pending'] = self.evolution.offspring(s['population'], c.population,
                                                       self.rng, self.new_individual)
                s['pretrain'], s['trained'], s['cursor'], s['stage'] = [], [], 0, 'mutated'
                self.save('running')
            while s['cursor'] < len(s['pending']):
                individual = s['pending'][s['cursor']]
                start = time.monotonic()
                train = None
                if s['stage'] == 'trained':
                    train = train_individual(c, individual, self.env_factory)
                    s['train_steps'] += train['steps']
                    s['resets'] += train['resets']
                    s['updates'] += train['updates']
                metrics = w.evaluate_individual(c, individual, self.env_factory, Actor)
                s['fitness_steps'] += metrics['steps']
                s['resets'] += metrics['resets']
                s['history'].append(dict(id=individual['id'], stage=s['stage'],
                    generation=s['generation'], parents=individual['parents'], train=train,
                    blob_sha256=w.blob_digest(individual['blob']), metrics=metrics,
                    seconds=time.monotonic() - start))
                s['cursor'] += 1
                w.require(self.total_steps <= c.max_env_steps, 'Budget overrun.')
                self.save('running')
                print(f"candidate={individual['id']} stage={s['stage']} actions={self.total_steps}", flush=True)
                completed += 1
                if stop_after_candidates is not None and completed >= stop_after_candidates:
                    self.save('paused_at_candidate_boundary')
                    return
            if s['stage'] == 'initial':
                s['population'] = self.evolution.select(s['pending'], c.population)
            elif s['stage'] == 'mutated':
                s['pretrain'] = s['pending']
                s['pending'] = []
                for p in s['pretrain']:
                    child = self.new_individual(copy.deepcopy(p['blob']), 'trained')
                    child.update(parents=[p['id']], age=p['age'],
                                 lineage_train_steps=p['lineage_train_steps'])
                    s['pending'].append(child)
                s['cursor'], s['stage'] = 0, 'trained'
                self.save('running')
                continue
            else:
                pool = s['population'] + s['pretrain'] + s['pending']
                w.require(len(pool) == 3 * c.population, 'Incomplete P/P-prime/Q pool.')
                selected = self.evolution.select(pool, c.population)
                s['selection_history'].append(dict(generation=s['generation'] + 1,
                    parent_ids=[p['id'] for p in s['population']],
                    mutated_ids=[p['id'] for p in s['pretrain']],
                    trained_ids=[p['id'] for p in s['pending']],
                    selected_ids=[p['id'] for p in selected]))
                s['population'] = selected
                s['generation'] += 1
            s['pending'], s['pretrain'], s['cursor'], s['stage'] = [], [], 0, 'ready'
            self.save('running')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    train = commands.add_parser('train')
    train.add_argument('--outdir', type=Path, required=True)
    train.add_argument('--resume', action='store_true')
    train.add_argument('--stop-after-candidates', type=int)
    defaults = Config()
    for key, value in asdict(defaults).items():
        train.add_argument('--' + key.replace('_', '-'), type=type(value), default=value)
    evaluate = commands.add_parser('eval')
    evaluate.add_argument('--checkpoint', type=Path, required=True)
    evaluate.add_argument('--outdir', type=Path, required=True)
    evaluate.add_argument('--profiles', nargs='+', default=['clean', 'seen', 'holdout', 'sparse'])
    evaluate.add_argument('--steps', type=int, default=60000)
    evaluate.add_argument('--seed', type=int, default=200000)
    evaluate.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.command == 'train':
        c = Config(**{key: getattr(args, key) for key in asdict(defaults)})
        w.require(args.stop_after_candidates is None or args.stop_after_candidates > 0,
                  'Invalid pause count.')
        with w.run_lock(args.outdir):
            Experiment(c, args.outdir, args.resume).run(args.stop_after_candidates)
    else:
        with w.run_lock(args.checkpoint.parent), w.run_lock(args.outdir):
            w.evaluate_population(args.checkpoint, args.outdir, args.profiles, args.steps,
                args.seed, args.device, model_factory=Actor, identity_factory=source_identity)


if __name__ == '__main__':
    main()
