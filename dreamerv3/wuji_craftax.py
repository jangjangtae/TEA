"""WUJI-derived Craftax port: upstream EA operators with a CUDA Double DQN.

Not a bitwise reproduction of the original asynchronous BlockMaze experiment.
See docs/wuji_craftax_port.md for objective mapping and explicit deviations.
This module uses GPL-3.0-or-later upstream WUJI code; preserve its attribution.
"""

import argparse
from collections import Counter
import configparser
from contextlib import contextmanager
import copy
from dataclasses import asdict, dataclass
import hashlib
import fcntl
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / '.deps/wuji_upstream'
UPSTREAM_COMMIT = 'de7161c29c59c4cc9ffb729d8c19dfb52e37a2c6'
SCHEMA = 1
PROFILES = ('clean', 'train', 'seen', 'holdout', 'sparse')
_CRAFTAX_ENGINE = None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


@contextmanager
def run_lock(outdir):
    outdir = Path(outdir).resolve()
    outdir.parent.mkdir(parents=True, exist_ok=True)
    with (outdir.parent / ('.' + outdir.name + '.lock')).open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Run is already in use: {outdir}') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


@dataclass(frozen=True)
class Config:
    seed: int = 0
    population: int = 30
    train_profile: str = 'train'
    max_env_steps: int = 1000000
    learn_steps: int = 10000
    fitness_steps: int = 3000
    fitness_rollouts: int = 3
    fitness_seed: int = 100000
    episode_length: int = 10000
    batch_size: int = 128
    replay_capacity: int = 10000
    learning_starts: int = 128
    train_every: int = 128
    updates_per_collect: int = 1
    target_update: int = 25
    lr: float = 0.00025
    discount: float = 0.99
    device: str = 'cuda:0'
    min_free_gib: float = 10.0

    def validate(self):
        require(self.population >= 3, 'NSGA-II mating requires population >= 3.')
        require(self.train_profile in ('clean', 'train'),
                'Selection/training cannot access held-out evaluation profiles.')
        for key in ('max_env_steps', 'learn_steps', 'fitness_steps', 'fitness_rollouts',
                    'episode_length', 'batch_size', 'replay_capacity', 'learning_starts',
                    'train_every', 'updates_per_collect', 'target_update'):
            require(getattr(self, key) > 0, f'{key} must be positive.')
        require(self.replay_capacity >= self.batch_size, 'Replay smaller than batch.')
        require(self.learning_starts >= self.batch_size, 'Warmup smaller than batch.')
        require(self.learn_steps >= max(self.learning_starts, self.train_every),
                'Each learning segment must contain at least one update.')
        require(self.fitness_steps >= self.fitness_rollouts, 'Empty fitness rollout.')
        require(self.max_env_steps >= self.population * (self.learn_steps + 2 * self.fitness_steps),
                'Budget must cover initial fitness and at least one learning generation.')
        require(0 <= self.discount <= 1 and self.lr > 0, 'Invalid DQN parameters.')
        require(self.min_free_gib >= 0, 'Invalid disk margin.')
        require(min(self.seed, self.fitness_seed) >= 0, 'Seeds must be nonnegative.')


class PixelQNetwork(nn.Module):
    """Spatial RGB CNN; unlike upstream DNN1, does not globally average position."""

    def __init__(self, config=None, inputs=3, outputs=17):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(inputs, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3), nn.ReLU(), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(),
                                  nn.Linear(256, outputs))

    def forward(self, image):
        return self.head(self.features(image))

    def group(self):
        groups = {}
        for key in self.state_dict():
            groups.setdefault(key.rsplit('.', 1)[0], []).append(key)
        return list(groups.values())


def clone_blob(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def blob_digest(blob):
    digest = hashlib.sha256()
    for key, value in blob.items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def image_tensor(images, device):
    array = np.asarray(images, dtype=np.uint8)
    return torch.as_tensor(array.copy(), device=device).permute(0, 3, 1, 2).float() / 255


def delivered_reward(obs):
    reward = float(obs['reward'])
    require(math.isfinite(reward), 'Nonfinite environment reward.')
    require(np.isclose(reward, float(obs.get('log/env_reward', reward)), atol=1e-6),
            'Environment reward contains an unexpected bonus or channel mismatch.')
    require(float(obs.get('log/rnd_reward_bonus', 0)) == 0, 'RND must be disabled.')
    return reward


def profile_environment(profile, seed):
    require(profile in PROFILES, f'Unknown profile: {profile}')
    enabled = profile != 'clean'
    return {
        'CRAFTAX_FAULT': '0',
        'CRAFTAX_FAULT_SAMPLER': str(int(enabled)),
        'CRAFTAX_FAULT_PROFILE': 'benchmark_' + ('train' if not enabled else profile),
        'CRAFTAX_FAULT_SEED': str(seed),
        # Match the corrected FATE queue, not the wrapper's 0.25 default.
        'CRAFTAX_FAULT_EP_PROB': '0.5' if enabled else '0.0',
        'CRAFTAX_FAULT_MANIFEST_PROB': '1.0' if enabled else '0.0',
        'CRAFTAX_USE_RND': '0', 'CRAFTAX_RND_UPDATE': '0',
        'CRAFTAX_SEMANTIC_FAULT': '0',
    }


class CraftaxAdapter:
    def __init__(self, profile, seed, length):
        from embodied.envs.craftax import Craftax
        keys = [key for key in os.environ if key.startswith('CRAFTAX_')]
        saved = {key: os.environ[key] for key in keys}
        overrides = profile_environment(profile, seed)
        try:
            for key in keys:
                os.environ.pop(key)
            os.environ.update(overrides)
            self.env = Craftax(task='classic_pixels', seed=seed, length=length,
                               logs=True, platform='cpu')
            # Episode sampling reads environment variables too; keep the
            # explicit profile active for this adapter's entire lifetime.
            self._saved = saved
            self._overrides = overrides
        except BaseException:
            for key in overrides:
                os.environ.pop(key, None)
            os.environ.update(saved)
            raise

    def _ensure_shared_engine(self):
        global _CRAFTAX_ENGINE
        require(self.env is not None, 'Craftax adapter is closed.')
        if self.env._env is not None:
            return
        # Craftax JITs reset/step with the engine as a static argument. Reuse
        # that stateless engine, never the wrapper's RNG, state, or fault data.
        pid = os.getpid()
        if _CRAFTAX_ENGINE is None or _CRAFTAX_ENGINE[0] != pid:
            self.env._ensure_env()
            _CRAFTAX_ENGINE = (pid, self.env._env, self.env._compute_score)
        else:
            import jax
            _, engine, compute_score = _CRAFTAX_ENGINE
            params = engine.default_params
            if self.env._length:
                params = params.replace(max_timesteps=self.env._length)
            with jax.transfer_guard('allow'):
                key = jax.random.PRNGKey(self.env._seed)
            self.env._env = engine
            self.env._compute_score = compute_score
            self.env._params = params
            self.env._key = key

    def reset(self):
        self._ensure_shared_engine()
        obs = self.env.step({'action': np.int32(0), 'reset': True})
        require(bool(obs['is_first']), 'Reset did not produce is_first.')
        return obs

    def step(self, action):
        require(self.env is not None, 'Craftax adapter is closed.')
        obs = self.env.step({'action': np.int32(action), 'reset': False})
        require(not bool(obs['is_first']), 'Reset was counted as an action transition.')
        delivered_reward(obs)
        return obs

    def position(self):
        # Privileged coordinates are only a selection metric, never policy input.
        return tuple(self.env._current_position())

    def close(self):
        if self.env is None:
            return
        env, self.env = self.env, None
        try:
            env.close()
        finally:
            for key in self._overrides:
                os.environ.pop(key, None)
            os.environ.update(self._saved)


class Replay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.size = self.cursor = 0
        self.images = np.empty((capacity, 64, 64, 3), np.uint8)
        self.next_images = np.empty_like(self.images)
        self.actions = np.empty(capacity, np.int64)
        self.rewards = np.empty(capacity, np.float32)
        self.terminals = np.empty(capacity, bool)

    def add(self, image, action, reward, next_image, terminal):
        index = self.cursor
        self.images[index], self.next_images[index] = image, next_image
        self.actions[index], self.rewards[index] = action, reward
        self.terminals[index] = terminal
        self.cursor = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, count, rng, device):
        require(self.size >= count, 'Insufficient replay.')
        indices = rng.choice(self.size, count, replace=False)
        return (image_tensor(self.images[indices], device),
                torch.as_tensor(self.actions[indices], device=device),
                torch.as_tensor(self.rewards[indices], device=device),
                image_tensor(self.next_images[indices], device),
                torch.as_tensor(self.terminals[indices], device=device))


def double_dqn_targets(online, target, next_images, rewards, terminals, discount):
    with torch.no_grad():
        actions = online(next_images).argmax(1, keepdim=True)
        values = target(next_images).gather(1, actions).squeeze(1)
        return rewards + discount * (~terminals).float() * values


class Learner:
    def __init__(self, config, blob):
        self.config = config
        self.model = PixelQNetwork().to(config.device)
        self.model.load_state_dict(blob)
        self.target = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.replay = Replay(config.replay_capacity)
        self.updates = 0

    def update(self, rng):
        c = self.config
        image, action, reward, nxt, terminal = self.replay.sample(c.batch_size, rng, c.device)
        label = double_dqn_targets(self.model, self.target, nxt, reward, terminal, c.discount)
        predicted = self.model(image).gather(1, action[:, None]).squeeze(1)
        loss = nn.functional.mse_loss(predicted, label)
        require(bool(torch.isfinite(loss)), 'Nonfinite Double-DQN loss.')
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                    for p in self.model.parameters()), 'Missing or nonfinite gradient.')
        self.optimizer.step()
        require(all(bool(torch.isfinite(p).all()) for p in self.model.parameters()),
                'Nonfinite model weight after update.')
        self.updates += 1
        if self.updates % c.target_update == 0:
            self.target.load_state_dict(self.model.state_dict())
        return float(loss.detach())


def policy_action(model, obs, device, rng=None, epsilon=0.0):
    if rng is not None and rng.random() < epsilon:
        return int(rng.integers(17))
    with torch.no_grad():
        q = model(image_tensor([obs['image']], device))
        require(bool(torch.isfinite(q).all()), 'Nonfinite action values.')
        return int(q.argmax(1).item())


def train_individual(config, individual, env_factory=CraftaxAdapter):
    c = config
    learner = Learner(c, individual['blob'])
    rng = np.random.default_rng(individual['train_seed'])
    env = env_factory(c.train_profile, individual['train_seed'], c.episode_length)
    epsilon = float(individual['epsilon'])
    total_reward = 0.0
    losses = []
    resets = 0
    initial_digest = blob_digest(individual['blob'])
    try:
        obs = env.reset()
        resets += 1
        for step in range(1, c.learn_steps + 1):
            action = policy_action(learner.model, obs, c.device, rng, epsilon)
            nxt = env.step(action)
            reward = delivered_reward(nxt)
            total_reward += reward
            learner.replay.add(obs['image'], action, reward, nxt['image'],
                               bool(nxt['is_terminal']))
            if step >= c.learning_starts and step % c.train_every == 0:
                for _ in range(c.updates_per_collect):
                    losses.append(learner.update(rng))
                # Exploration decays per collection block, not per replay pass.
                epsilon = max(epsilon - 0.001, 0.05)
            obs = nxt
            if nxt['is_last'] and step < c.learn_steps:
                obs = env.reset()
                resets += 1
        require(learner.updates > 0, 'No gradient update occurred.')
        individual['blob'] = clone_blob(learner.model)
        individual['epsilon'] = epsilon
        individual['age'] += c.learn_steps
        return dict(steps=c.learn_steps, resets=resets, updates=learner.updates,
                    target_syncs=learner.updates // c.target_update,
                    task_reward_sum=total_reward, mean_loss=float(np.mean(losses)),
                    weights_changed=initial_digest != blob_digest(individual['blob']))
    finally:
        env.close()


def allocate_steps(total, count):
    require(total >= count > 0, 'Every rollout must receive at least one step.')
    base, rest = divmod(total, count)
    return [base + int(i < rest) for i in range(count)]


def rollout_metrics(model, device, env, steps):
    obs = env.reset()
    positions = {env.position()}
    coverage = []
    completed_returns = []
    events = Counter()
    event_records = []
    reward_sum = episode_return = 0.0
    resets = 1
    episode_steps = 0
    for step in range(1, steps + 1):
        action = policy_action(model, obs, device)
        obs = env.step(action)
        reward = delivered_reward(obs)
        reward_sum += reward
        episode_return += reward
        episode_steps += 1
        positions.add(env.position())
        # Oracle labels enter reporting only, never fitness, policy, or replay.
        if int(obs.get('log/fault_manifested', 0)):
            subtype = int(obs.get('log/fault_type_id', 0))
            events[str(subtype)] += 1
            event_records.append(dict(step=step, subtype_id=subtype))
        if obs['is_last']:
            coverage.append(len(positions))
            completed_returns.append(episode_return)
            episode_return = 0.0
            episode_steps = 0
            if step < steps:
                obs = env.reset()
                resets += 1
                positions = {env.position()}
    if episode_steps:
        coverage.append(len(positions))
    return dict(steps=steps, resets=resets, task_reward_sum=reward_sum,
                segment_unique_positions=coverage, completed_returns=completed_returns,
                partial_return=episode_return if episode_steps else None,
                events=dict(events), event_records=event_records)


def aggregate_metrics(rollouts):
    steps = sum(row['steps'] for row in rollouts)
    reward = sum(row['task_reward_sum'] for row in rollouts)
    coverage = [n for row in rollouts for n in row['segment_unique_positions']]
    completed = [r for row in rollouts for r in row['completed_returns']]
    events = Counter()
    for row in rollouts:
        events.update(row['events'])
    return dict(steps=steps, resets=sum(row['resets'] for row in rollouts),
                task_reward_sum=reward, task_reward_per_1000_steps=1000 * reward / steps,
                mean_segment_unique_positions=float(np.mean(coverage)),
                mean_completed_episode_return=float(np.mean(completed)) if completed else None,
                completed_episodes=len(completed), events=dict(events),
                discovered_subtypes=sorted(map(int, events)),
                fault_events_per_10000_steps=10000 * sum(events.values()) / steps)


def fitness_objectives(metrics):
    objectives = (metrics['task_reward_per_1000_steps'],
                  metrics['mean_segment_unique_positions'])
    require(all(math.isfinite(x) for x in objectives), 'Nonfinite fitness.')
    return objectives


def evaluate_individual(config, individual, env_factory=CraftaxAdapter,
                        model_factory=PixelQNetwork):
    model = model_factory().to(config.device).eval()
    model.load_state_dict(individual['blob'])
    rollouts = []
    for index, steps in enumerate(allocate_steps(config.fitness_steps, config.fitness_rollouts)):
        seed = config.fitness_seed + index
        env = env_factory(config.train_profile, seed, config.episode_length)
        try:
            row = rollout_metrics(model, config.device, env, steps)
            rollouts.append(dict(row, seed=seed))
        finally:
            env.close()
    metrics = aggregate_metrics(rollouts)
    individual['result'] = dict(fitness_objectives=fitness_objectives(metrics), metrics=metrics)
    individual['age'] += metrics['steps']
    return metrics


class Evolution:
    def __init__(self):
        from wuji.ea.crossover.blob.pth.single_point import Split
        from wuji.ea.crossover.real.sbx import Crossover
        from wuji.ea.mutation.real.pm import Mutation
        from wuji.ea.selection.non_dominated.nsga_ii import Selection
        self.config = configparser.ConfigParser()
        require(bool(self.config.read(UPSTREAM / 'config.ini')), 'Missing upstream config.')
        context = {'encoding': {
            'blob': {'module': [PixelQNetwork], 'init': [{'kwargs': {}}]},
            'real': {'boundary': np.array([[0.0, 1.0]], np.float32)},
        }}
        evaluator = SimpleNamespace(context=context, get_kind=lambda: 0)
        self.blob_cross = Split(self.config, evaluator)
        self.real_cross = Crossover(self.config, evaluator)
        self.real_mutation = Mutation(self.config, evaluator)
        self.selection = Selection(self.config)

    @staticmethod
    def candidates(population):
        # Operators mutate selection metadata, not the underlying policy state.
        return [dict(id=p['id'], age=p['age'], source=p,
                     result=dict(objective=p['result']['fitness_objectives'],
                                 point=p['result']['fitness_objectives'],
                                 fitness=p['result']['fitness_objectives'][0])) for p in population]

    def select(self, population, size):
        selected = self.selection(self.candidates(population), size)
        return [p['source'] for p in selected]

    def offspring(self, population, count, rng, new_individual):
        from wuji.ea.mating.non_dominated.nsga_ii import NSGA_II
        from wuji.ea.selection.non_dominated.nsga_ii import crowding_distance
        # Populate crowding distances even when the entire last front survives.
        self.selection(self.candidates(population), len(population))
        candidates = []
        for front in self.selection.layers:
            crowding_distance(front, lambda result: result['point'])
            candidates.extend(front)
        mating = NSGA_II(self.config, candidates)
        for operator in (mating, self.blob_cross, self.real_cross, self.real_mutation):
            operator.random.seed(int(rng.integers(2**31)))
        children = []
        while len(children) < count:
            parents = [p['source'] for p in mating(2)]
            blobs = self.blob_cross(*(p['blob'] for p in parents))
            real = self.real_cross(*(np.array([p['epsilon']], np.float32) for p in parents))
            for index, (blob, genes) in enumerate(zip(blobs, real)):
                if len(children) == count:
                    break
                genes = self.real_mutation(genes)
                child = new_individual(copy.deepcopy(blob), float(genes[0]))
                child['parents'] = [p['id'] for p in parents]
                child['age'] = parents[index]['age']
                children.append(child)
        return children


def source_identity():
    commit = subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'],
                                     text=True).strip()
    require(commit == UPSTREAM_COMMIT, 'Upstream revision changed; re-audit before running.')
    dirty = subprocess.check_output(['git', '-C', str(UPSTREAM), 'status', '--porcelain'],
                                    text=True).strip()
    require(not dirty, 'Upstream checkout must remain unmodified.')
    paths = [Path(__file__), ROOT / 'embodied/envs/craftax.py',
             ROOT / 'tools/run_wuji_craftax.py']
    return dict(upstream_commit=commit, hashes={str(p.relative_to(ROOT)):
                hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def prepare_device(device):
    if device.startswith('cuda'):
        require(torch.cuda.is_available(), 'CUDA unavailable; refusing silent CPU fallback.')
        fraction = float(os.environ.get('WUJI_CUDA_MEMORY_FRACTION', '1.0'))
        require(math.isfinite(fraction) and 0 < fraction <= 1, 'Invalid CUDA memory fraction.')
        torch.cuda.set_per_process_memory_fraction(fraction, device)
    else:
        require(device == 'cpu', 'Expected cpu or cuda device.')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Experiment:
    def __init__(self, config, outdir, resume=False, env_factory=CraftaxAdapter):
        config.validate()
        prepare_device(config.device)
        self.config, self.outdir, self.env_factory = config, Path(outdir), env_factory
        self.identity = source_identity()
        self.evolution = Evolution()
        self.rng = np.random.default_rng(config.seed)
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.checkpoint = self.outdir / 'checkpoint.pt'
        if resume:
            require(self.checkpoint.exists(), 'No checkpoint to resume.')
            payload = torch.load(self.checkpoint, map_location='cpu', weights_only=False)
            require(payload['schema'] == SCHEMA and payload['config'] == asdict(config),
                    'Checkpoint configuration mismatch.')
            require(payload['identity'] == self.identity, 'Code changed since checkpoint.')
            self.state = payload['state']
            self.rng.bit_generator.state = payload['rng']
            torch.set_rng_state(payload['torch_rng'])
        else:
            require(not any(self.outdir.iterdir()), 'Output directory must be empty; use --resume.')
            self.state = dict(stage='initial', generation=0, cursor=0, next_id=0,
                              population=[], pending=[], history=[], train_steps=0,
                              fitness_steps=0, resets=0, updates=0)
            for _ in range(config.population):
                self.state['pending'].append(self.new_individual(
                    clone_blob(PixelQNetwork()), float(self.rng.uniform(0, 1))))
            self.save('initialized')

    @property
    def total_steps(self):
        return self.state['train_steps'] + self.state['fitness_steps']

    def new_individual(self, blob, epsilon):
        index = self.state['next_id']
        self.state['next_id'] += 1
        return dict(id=index, blob=blob, epsilon=epsilon, age=0, parents=[],
                    train_seed=int(self.rng.integers(2**31)))

    def save(self, status):
        require(shutil.disk_usage(self.outdir).free >= self.config.min_free_gib * 1024**3,
                'Disk free space below safety margin.')
        tmp = self.checkpoint.with_suffix('.pt.tmp')
        payload = dict(schema=SCHEMA, config=asdict(self.config), identity=self.identity,
                       state=self.state, rng=self.rng.bit_generator.state,
                       torch_rng=torch.get_rng_state())
        with tmp.open('wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, self.checkpoint)
        summary = dict(status=status, generation=self.state['generation'],
                       stage=self.state['stage'], cursor=self.state['cursor'],
                       train_steps=self.state['train_steps'], fitness_steps=self.state['fitness_steps'],
                       total_action_steps=self.total_steps, reset_calls=self.state['resets'],
                       gradient_updates=self.state['updates'], config=asdict(self.config),
                       identity=self.identity, population=[self.public(p) for p in self.state['population']],
                       candidate_history=self.state['history'])
        summary['runtime'] = dict(python=sys.version.split()[0], torch=torch.__version__,
                                  numpy=np.__version__, learner_device=self.config.device,
                                  environment_device='cpu')
        if self.config.device.startswith('cuda'):
            summary['runtime'].update(
                gpu=torch.cuda.get_device_name(self.config.device),
                cuda_allocator_fraction=float(os.environ.get('WUJI_CUDA_MEMORY_FRACTION', '1.0')),
                peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(self.config.device),
                peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(self.config.device))
        atomic_json(self.outdir / 'summary.json', summary)

    @staticmethod
    def public(individual):
        return {k: v for k, v in individual.items() if k != 'blob'}

    def run(self, stop_after_candidates=None):
        completed = 0
        while True:
            s, c = self.state, self.config
            if s['stage'] == 'ready':
                generation_cost = c.population * (c.learn_steps + c.fitness_steps)
                if self.total_steps + generation_cost > c.max_env_steps:
                    self.save('budget_complete')
                    return
                s['pending'] = self.evolution.offspring(
                    s['population'], c.population, self.rng, self.new_individual)
                s['cursor'], s['stage'] = 0, 'generation'
                self.save('running')
            while s['cursor'] < len(s['pending']):
                individual = s['pending'][s['cursor']]
                start = time.monotonic()
                train = None
                if s['stage'] == 'generation':
                    train = train_individual(c, individual, self.env_factory)
                    s['train_steps'] += train['steps']
                    s['resets'] += train['resets']
                    s['updates'] += train['updates']
                metrics = evaluate_individual(c, individual, self.env_factory)
                s['fitness_steps'] += metrics['steps']
                s['resets'] += metrics['resets']
                s['history'].append(dict(id=individual['id'], stage=s['stage'],
                    generation=s['generation'], parents=individual['parents'], train=train,
                    metrics=metrics, seconds=time.monotonic() - start))
                s['cursor'] += 1
                require(self.total_steps <= c.max_env_steps, 'Budget overrun.')
                self.save('running')
                print(json.dumps(dict(candidate=individual['id'], stage=s['stage'],
                      generation=s['generation'], total_steps=self.total_steps,
                      objective=individual['result']['fitness_objectives'])), flush=True)
                completed += 1
                if stop_after_candidates is not None and completed >= stop_after_candidates:
                    self.save('paused_at_candidate_boundary')
                    return
            if s['stage'] == 'initial':
                s['population'] = self.evolution.select(s['pending'], c.population)
            else:
                s['population'] = self.evolution.select(s['population'] + s['pending'], c.population)
                s['generation'] += 1
            s['pending'], s['cursor'], s['stage'] = [], 0, 'ready'
            self.save('running')


def evaluate_population(checkpoint, outdir, profiles, total_steps, seed, device,
                        env_factory=CraftaxAdapter, model_factory=PixelQNetwork,
                        identity_factory=source_identity):
    prepare_device(device)
    require(len(profiles) == len(set(profiles)) and set(profiles) <= set(PROFILES),
            'Invalid or duplicate evaluation profiles.')
    outdir = Path(outdir)
    require(not outdir.exists(), 'Evaluation output already exists.')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    require(payload['schema'] == SCHEMA and payload['identity'] == identity_factory(),
            'Checkpoint code/schema mismatch.')
    state = payload['state']
    require(state['stage'] == 'ready', 'Evaluate only a fully selected population.')
    population = sorted(state['population'], key=lambda p: p['id'])
    allocations = allocate_steps(total_steps, len(population))
    before = [blob_digest(p['blob']) for p in population]
    results = []
    outdir.mkdir(parents=True)
    for profile in profiles:
        rows = []
        for slot, (individual, steps) in enumerate(zip(population, allocations)):
            model = model_factory().to(device).eval()
            model.load_state_dict(individual['blob'])
            env = env_factory(profile, seed + slot, payload['config']['episode_length'])
            try:
                row = rollout_metrics(model, device, env, steps)
                require(blob_digest(clone_blob(model)) == before[slot], 'Evaluation changed weights.')
                rows.append(dict(row, policy_id=individual['id'], seed=seed + slot))
            finally:
                env.close()
        metrics = aggregate_metrics(rows)
        require(metrics['steps'] == total_steps, 'Population evaluation budget mismatch.')
        results.append(dict(profile=profile, metrics=metrics, rollouts=rows))
        atomic_json(outdir / f'{profile}.json', results[-1])
    require(before == [blob_digest(p['blob']) for p in population], 'Population mutated.')
    atomic_json(outdir / 'summary.json', dict(status='complete', results=results,
        total_action_steps=total_steps * len(profiles), population_size=len(population),
        allocation='equal fixed step blocks across the population; greedy; no updates',
        checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()))
    return results


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
    evaluate.add_argument('--steps', type=int, default=60000,
                          help='Total action steps PER PROFILE, not per population member.')
    evaluate.add_argument('--seed', type=int, default=200000)
    evaluate.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.command == 'train':
        config = Config(**{key: getattr(args, key) for key in asdict(defaults)})
        if args.stop_after_candidates is not None:
            require(args.stop_after_candidates > 0, 'Invalid candidate stop count.')
        with run_lock(args.outdir):
            experiment = Experiment(config, args.outdir, args.resume)
            try:
                experiment.run(args.stop_after_candidates)
            except BaseException as error:
                # Keep the last atomic checkpoint; do not commit partially trained offspring.
                atomic_json(args.outdir / 'failure.json', dict(type=type(error).__name__,
                            message=str(error), recovery='Resume the last committed candidate boundary.'))
                raise
    else:
        with run_lock(args.checkpoint.parent), run_lock(args.outdir):
            evaluate_population(args.checkpoint, args.outdir, args.profiles, args.steps,
                                args.seed, args.device)


if __name__ == '__main__':
    main()
