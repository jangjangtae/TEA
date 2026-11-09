#!/usr/bin/env python
import json
import os
import numpy as np

from embodied.envs.crafter import Crafter

STEPS = int(os.getenv("RANDOM_EVAL_STEPS", "100000"))
SEED = int(os.getenv("RANDOM_EVAL_SEED", "0"))
LOGDIR = os.path.expanduser(os.getenv("RANDOM_EVAL_LOGDIR", "~/logdir/random_bug_eval_100k"))

os.makedirs(LOGDIR, exist_ok=True)

env = Crafter(
    task='reward',
    size=(64, 64),
    logs=True,
    logdir=LOGDIR,
    seed=SEED,
)

num_actions = env._env.action_space.n
scores_path = os.path.join(LOGDIR, "random_scores.jsonl")

global_step = 0
episode = 0
ep_reward = 0.0
ep_task_reward = 0.0
ep_len = 0

obs = env.step({'action': 0, 'reset': True})

while global_step < STEPS:
    if bool(obs['is_last']):
        with open(scores_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "episode": episode,
                "global_step": global_step,
                "score_total": float(ep_reward),
                "score_task": float(ep_task_reward),
                "length": int(ep_len),
            }) + "\n")
        episode += 1
        ep_reward = 0.0
        ep_task_reward = 0.0
        ep_len = 0
        obs = env.step({'action': 0, 'reset': True})
        continue

    action = int(np.random.randint(num_actions))
    obs = env.step({'action': action, 'reset': False})

    ep_reward += float(obs['reward'])
    task_raw = float(obs.get('log/task_reward_raw', obs['reward']))
    ep_task_reward += task_raw
    ep_len += 1
    global_step += 1

    if global_step % 10000 == 0:
        print(f"[Random Eval] step={global_step}, episode={episode}, ep_len={ep_len}, ep_reward={ep_reward:.3f}", flush=True)

with open(scores_path, "a", encoding="utf-8") as f:
    f.write(json.dumps({
        "episode": episode,
        "global_step": global_step,
        "score_total": float(ep_reward),
        "score_task": float(ep_task_reward),
        "length": int(ep_len),
    }) + "\n")

print(f"[DONE] random bug eval finished: {LOGDIR}")