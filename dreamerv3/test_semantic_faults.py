import os
from collections import deque

# ------------------------------------------------------------------
# IMPORTANT:
# Run this script inside the same venv where modified crafter env.py lives.
# Example:
#   source ~/dreamerv3/dreamer_cuda/bin/activate
#   python /mnt/data/test_semantic_faults.py
# ------------------------------------------------------------------

import numpy as np
import crafter
from crafter import constants

MOVE_TO_DELTA = {
    'move_left': (-1, 0),
    'move_right': (1, 0),
    'move_up': (0, -1),
    'move_down': (0, 1),
}
DELTA_TO_MOVE = {v: k for k, v in MOVE_TO_DELTA.items()}
WALKABLE = set(constants.walkable)


def action_index(env, name):
    return env.action_names.index(name)


def semantic_ids(env):
    return env._world._mat_ids.copy(), env._sem_view._obj_ids.copy()


def is_free_for_player(env, pos):
    mat, obj = env._world[pos]
    return (obj is None) and (mat in WALKABLE)


def neighbors(env, pos):
    x, y = pos
    cand = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
    out = []
    for q in cand:
        mat, _ = env._world[q]
        if mat is None:
            continue
        out.append(q)
    return out


def bfs_path(env, start, goals):
    goals = set(tuple(g) for g in goals)
    q = deque([tuple(start)])
    prev = {tuple(start): None}
    while q:
        cur = q.popleft()
        if cur in goals:
            path = [cur]
            while prev[path[-1]] is not None:
                path.append(prev[path[-1]])
            path.reverse()
            return path
        for nxt in neighbors(env, cur):
            nxt = tuple(nxt)
            if nxt in prev:
                continue
            if not is_free_for_player(env, nxt):
                continue
            prev[nxt] = cur
            q.append(nxt)
    return None


def find_material_positions(env, material_name):
    mat_ids, _ = semantic_ids(env)
    mat_id = mat_ids[material_name]
    ys, xs = np.where(env._world._mat_map.T == mat_id)
    # transpose handling: easier to just scan world directly for correctness
    out = []
    area = env._world.area
    for x in range(area[0]):
        for y in range(area[1]):
            mat, _ = env._world[(x, y)]
            if mat == material_name:
                out.append((x, y))
    return out


def adjacent_walkable_goals(env, targets):
    goals = []
    for tx, ty in targets:
        for q in [(tx - 1, ty), (tx + 1, ty), (tx, ty - 1), (tx, ty + 1)]:
            mat, obj = env._world[q]
            if mat is None:
                continue
            if obj is None and mat in WALKABLE:
                goals.append(q)
    return list(set(goals))


def move_to(env, goal_positions, max_steps=500):
    start = tuple(env._player.pos)
    path = bfs_path(env, start, goal_positions)
    if not path:
        return False, None
    # skip start
    for nxt in path[1:]:
        dx = nxt[0] - env._player.pos[0]
        dy = nxt[1] - env._player.pos[1]
        act = DELTA_TO_MOVE[(int(dx), int(dy))]
        obs, reward, done, info = env.step(action_index(env, act))
        if done:
            return False, info
    return True, None


def step_action(env, action_name):
    obs, reward, done, info = env.step(action_index(env, action_name))
    return obs, reward, done, info


def set_facing_toward(env, target_pos):
    dx = int(target_pos[0] - env._player.pos[0])
    dy = int(target_pos[1] - env._player.pos[1])
    assert abs(dx) + abs(dy) == 1, (env._player.pos, target_pos)
    env._player.facing = np.array((dx, dy))


def do_on_material(env, material_name):
    targets = find_material_positions(env, material_name)
    if not targets:
        return False, None
    goals = adjacent_walkable_goals(env, targets)
    ok, info = move_to(env, goals)
    if not ok:
        return False, info
    player = tuple(env._player.pos)
    # nearest target adjacent to player
    adj_targets = [t for t in targets if abs(t[0] - player[0]) + abs(t[1] - player[1]) == 1]
    if not adj_targets:
        return False, None
    target = adj_targets[0]
    set_facing_toward(env, target)
    obs, reward, done, info = step_action(env, 'do')
    return True, info


def place_structure(env, name):
    player = tuple(env._player.pos)
    candidates = []
    for q in neighbors(env, player):
        mat, obj = env._world[q]
        if obj is None and mat in {'grass', 'path', 'sand'}:
            candidates.append(q)
    if not candidates:
        # move randomly a bit and retry
        for _ in range(10):
            obs, reward, done, info = step_action(env, np.random.choice(list(MOVE_TO_DELTA.keys())))
            if done:
                return False, info
        return place_structure(env, name)
    target = candidates[0]
    set_facing_toward(env, target)
    obs, reward, done, info = step_action(env, f'place_{name}')
    return True, info


def move_near_material(env, material_name):
    targets = find_material_positions(env, material_name)
    if not targets:
        return False, None
    goals = adjacent_walkable_goals(env, targets)
    return move_to(env, goals)


def ensure_table(env):
    # gather 2 wood and place table
    while env._player.inventory['wood'] < 2:
        ok, info = do_on_material(env, 'tree')
        if not ok:
            return False, info
    ok, info = place_structure(env, 'table')
    return ok, info


def craft_item(env, item_name):
    need_station = 'table'
    if 'iron_' in item_name:
        need_station = 'furnace'  # but table also needed, skip advanced recipes for smoke test
    ok, info = move_near_material(env, 'table')
    if not ok:
        return False, info
    obs, reward, done, info = step_action(env, f'make_{item_name}')
    return True, info


def print_fault(info, prefix=''):
    keys = [
        'semantic_fault_episode', 'semantic_fault_applied', 'semantic_fault_type',
        'semantic_fault_trigger', 'semantic_fault_count', 'inventory', 'achievements', 'reward'
    ]
    print(prefix)
    for k in keys:
        if k in info:
            print(f'  {k}: {info[k]}')


def configure_env_for(subtype, seed=0):
    os.environ['CRAFTER_SEMANTIC_FAULT_SAMPLER'] = '1'
    os.environ['CRAFTER_SEMANTIC_FAULT_PROFILE'] = 'eval_holdout'
    os.environ['CRAFTER_SEMANTIC_FAULT_EP_PROB'] = '1.0'
    os.environ['CRAFTER_SEMANTIC_SUBTYPES'] = subtype
    os.environ['CRAFTER_SEMANTIC_FAULT_VERBOSE'] = '1'
    return crafter.Env(reward=True, seed=seed)


def run_tool_collect_desync(seed=0):
    print('\n=== TEST: tool_collect_desync_on_upgrade ===')
    env = configure_env_for('tool_collect_desync_on_upgrade', seed=seed)
    env.reset()
    ensure_table(env)
    while env._player.inventory['wood'] < 1:
        do_on_material(env, 'tree')
    craft_item(env, 'wood_pickaxe')
    before = env._player.inventory.copy()
    ok, info = do_on_material(env, 'stone')
    after = env._player.inventory.copy()
    print('Before stone collect:', before)
    print('After stone collect :', after)
    if info:
        print_fault(info, prefix='Info after collect:')
    return info


def run_craft_result_missing(seed=1):
    print('\n=== TEST: craft_result_missing_on_retry ===')
    env = configure_env_for('craft_result_missing_on_retry', seed=seed)
    env.reset()
    ensure_table(env)
    while env._player.inventory['wood'] < 2:
        do_on_material(env, 'tree')
    craft_item(env, 'wood_pickaxe')
    before = env._player.inventory.copy()
    ok, info = craft_item(env, 'wood_pickaxe')
    after = env._player.inventory.copy()
    print('Before retry craft:', before)
    print('After retry craft :', after)
    if info:
        print_fault(info, prefix='Info after retry craft:')
    return info


def run_station_place_ghost(seed=2):
    print('\n=== TEST: station_place_ghost_on_relocate ===')
    env = configure_env_for('station_place_ghost_on_relocate', seed=seed)
    env.reset()
    while env._player.inventory['wood'] < 4:
        do_on_material(env, 'tree')
    ok, info = place_structure(env, 'table')
    print_fault(info, prefix='After first table placement:')
    # Move a bit so we can place somewhere else
    for act in ['move_left', 'move_right', 'move_up', 'move_down']:
        obs, reward, done, info = step_action(env, act)
        if not done:
            break
    ok, info = place_structure(env, 'table')
    print_fault(info, prefix='After second table placement:')
    return info


def main():
    run_tool_collect_desync()
    run_craft_result_missing()
    run_station_place_ghost()


if __name__ == '__main__':
    main()
