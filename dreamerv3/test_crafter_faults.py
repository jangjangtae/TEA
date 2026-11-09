import os
from collections import deque

import numpy as np

# import 경로는 네 프로젝트 구조에 맞게 바꿔야 할 수 있음
# 예: from embodied.envs.crafter import Crafter
from embodied.envs.crafter import (
    Crafter,
    FAULT_PROFILE_ALIASES,
    FAULT_PROFILE_DEFAULTS,
)


def make_env():
    # deterministic train profile
    os.environ["CRAFTER_FAULT"] = "0"
    os.environ["CRAFTER_FAULT_SAMPLER"] = "0"
    os.environ["CRAFTER_FAULT_PROFILE"] = "train"
    os.environ["CRAFTER_FAULT_COOLDOWN"] = "0"
    os.environ.setdefault("CRAFTER_OUTPUT_DIR", "/tmp/crafter_fault_unit_tests")

    env = Crafter(
        task="reward",
        size=(64, 64),
        logs=False,
        seed=0,
    )

    # reset 1회
    env.step({"action": np.int32(0), "reset": True})
    return env


def test_fault_profile_definitions():
    assert FAULT_PROFILE_ALIASES["train"] == "benchmark_train"
    assert FAULT_PROFILE_ALIASES["eval_seen"] == "benchmark_seen"
    assert FAULT_PROFILE_ALIASES["eval_holdout"] == "benchmark_holdout"
    assert FAULT_PROFILE_ALIASES["v2_train"] == "benchmark_v2_train"
    assert FAULT_PROFILE_ALIASES["v2_seen"] == "benchmark_v2_seen"
    assert FAULT_PROFILE_ALIASES["v2_holdout"] == "benchmark_v2_holdout"
    assert FAULT_PROFILE_ALIASES["v2_sparse"] == "benchmark_v2_sparse"

    train = FAULT_PROFILE_DEFAULTS["benchmark_train"]
    seen = FAULT_PROFILE_DEFAULTS["benchmark_seen"]
    holdout = FAULT_PROFILE_DEFAULTS["benchmark_holdout"]
    v2_train = FAULT_PROFILE_DEFAULTS["benchmark_v2_train"]
    v2_seen = FAULT_PROFILE_DEFAULTS["benchmark_v2_seen"]
    v2_holdout = FAULT_PROFILE_DEFAULTS["benchmark_v2_holdout"]
    v2_sparse = FAULT_PROFILE_DEFAULTS["benchmark_v2_sparse"]

    assert train["action"] == seen["action"]
    assert train["context"] == seen["context"]
    assert train["reward"] == seen["reward"]
    assert "revisit_action_delay" in holdout["action"]
    assert "revisit_action_ignore" in holdout["context"]
    assert "reward_delay_after_two_rewards" in holdout["reward"]
    assert train["stochastic_manifest"] is True
    assert holdout["stochastic_manifest"] is True
    assert v2_train["action"] == v2_seen["action"]
    assert v2_train["context"] == v2_seen["context"]
    assert v2_train["reward"] == v2_seen["reward"]
    assert "delay_after_success" in v2_train["action"]
    assert "ignore_nonzero_after_reward" in v2_train["context"]
    assert "reward_delay_on_positive" in v2_train["reward"]
    assert "revisit_action_delay" in v2_holdout["action"]
    assert "ignore_nonzero_after_two_rewards" in v2_holdout["context"]
    assert "reward_delay_after_two_rewards" in v2_holdout["reward"]
    assert int(v2_sparse["cooldown"]) > int(v2_holdout["cooldown"])
    assert v2_train["stochastic_manifest"] is True
    assert v2_holdout["stochastic_manifest"] is True
    print("PASS: fault_profile_definitions")


def prime_env(env, family, subtype):
    env._fault_spec = {
        "family": family,
        "type": subtype,
        "severity": 1.0,
        "trigger": env._trigger_name_from_subtype(subtype),
    }
    env._fault_episode = 1
    env._fault_cooldown = 0
    env._after_positive_window = 0
    env._last_reward = 0.0
    env._last_requested_action = 0
    env._prev_executed_action = 0
    env._length = 0
    env._pending_reward = 0.0
    env._sticky_action = None
    env._sticky_remaining = 0
    env._requested_hist = deque([0, 0], maxlen=2)
    env._executed_hist = deque([0, 0], maxlen=2)
    env._reward_hist = deque([0.0, 0.0, 0.0, 0.0], maxlen=4)


def set_recent_revisit(env):
    env._last_ctx_key = b"ctx"
    env._ctx_visit_steps[b"ctx"] = deque([0, 5], maxlen=4)
    env._episode_step = 5


def test_drop_to_fallback():
    env = make_env()
    prime_env(env, "legacy_action_exec", "drop_to_fallback")
    env._fault_spec["fallback_action"] = 0

    out_action, applied, fault_type = env._apply_action_fault(2)

    assert applied == 1
    assert fault_type == "drop_to_fallback"
    assert out_action == 0
    print("PASS: drop_to_fallback")


def test_remap_after_success_switch():
    env = make_env()
    prime_env(env, "action_exec", "remap_after_success_switch")
    env._after_positive_window = 3
    env._last_requested_action = 1
    out_action, applied, fault_type = env._apply_action_fault(2)

    assert applied == 1
    assert fault_type == "remap_after_success_switch"
    assert out_action != 2
    print("PASS: remap_after_success_switch")


def test_delay_after_success():
    env = make_env()
    prime_env(env, "action_exec", "delay_after_success")
    env._after_positive_window = 3
    env._last_requested_action = 1
    out_action, applied, fault_type = env._apply_action_fault(2)

    assert applied == 1
    assert fault_type == "delay_after_success"
    assert out_action == 1
    print("PASS: delay_after_success")


def test_sticky_after_repeat_switch():
    env = make_env()
    prime_env(env, "action_exec", "sticky_after_repeat_switch")
    env._requested_hist = deque([1, 1], maxlen=2)

    # 1차 호출: arm만 됨
    out_action, applied, fault_type = env._apply_action_fault(2)
    assert applied == 0
    assert env._sticky_action == 2
    assert env._sticky_remaining == 1

    # 2차 호출: sticky 실제 적용
    out_action2, applied2, fault_type2 = env._apply_action_fault(3)
    assert applied2 == 1
    assert fault_type2 == "sticky_after_repeat_switch"
    assert out_action2 == 2
    print("PASS: sticky_after_repeat_switch")


def test_remap_after_repeat_switch():
    env = make_env()
    prime_env(env, "action_exec", "remap_after_repeat_switch")
    env._requested_hist = deque([1, 1], maxlen=2)

    out_action, applied, fault_type = env._apply_action_fault(2)

    assert applied == 1
    assert fault_type == "remap_after_repeat_switch"
    assert out_action != 2
    print("PASS: remap_after_repeat_switch")


def test_delay_after_late_episode_switch():
    env = make_env()
    prime_env(env, "action_exec", "delay_after_late_episode_switch")
    env._length = 60
    env._last_requested_action = 1

    out_action, applied, fault_type = env._apply_action_fault(2)

    assert applied == 1
    assert fault_type == "delay_after_late_episode_switch"
    assert out_action == 1
    print("PASS: delay_after_late_episode_switch")


def test_ignore_nonzero_after_reward():
    env = make_env()
    prime_env(env, "context_exec", "ignore_nonzero_after_reward")
    env._after_positive_window = 3

    out_action, applied, fault_type = env._apply_action_fault(2)
    assert applied == 1
    assert fault_type == "ignore_nonzero_after_reward"
    assert out_action == 0
    print("PASS: ignore_nonzero_after_reward")


def test_ignore_switch_late_episode():
    env = make_env()
    prime_env(env, "context_exec", "ignore_switch_late_episode")
    env._length = 60
    env._last_requested_action = 1

    out_action, applied, fault_type = env._apply_action_fault(2)
    assert applied == 1
    assert fault_type == "ignore_switch_late_episode"
    assert out_action == 0
    print("PASS: ignore_switch_late_episode")


def test_ignore_nonzero_after_two_rewards():
    env = make_env()
    prime_env(env, "context_exec", "ignore_nonzero_after_two_rewards")
    env._reward_hist = deque([1.0, 0.0, 1.0, 0.0], maxlen=4)

    out_action, applied, fault_type = env._apply_action_fault(2)

    assert applied == 1
    assert fault_type == "ignore_nonzero_after_two_rewards"
    assert out_action == 0
    print("PASS: ignore_nonzero_after_two_rewards")


def test_revisit_action_ignore():
    env = make_env()
    prime_env(env, "context_exec", "revisit_action_ignore")
    set_recent_revisit(env)

    out_action, applied, fault_type = env._apply_action_fault(2)

    assert applied == 1
    assert fault_type == "revisit_action_ignore"
    assert out_action == 0
    print("PASS: revisit_action_ignore")


def test_revisit_action_delay():
    env = make_env()
    prime_env(env, "action_exec", "revisit_action_delay")
    set_recent_revisit(env)
    env._last_requested_action = 1

    out_action, applied, fault_type = env._apply_action_fault(2)

    assert applied == 1
    assert fault_type == "revisit_action_delay"
    assert out_action == 1
    print("PASS: revisit_action_delay")


def test_delayed_switch_failure():
    env = make_env()
    prime_env(env, "action_exec", "delayed_switch_failure")
    env._episode_step = 12
    env._last_positive_reward_step = 10
    env._last_requested_action = 1

    out_action, applied, fault_type = env._apply_action_fault(2)

    assert applied == 1
    assert fault_type == "delayed_switch_failure"
    assert out_action != 2
    print("PASS: delayed_switch_failure")


def test_reward_delay_on_positive():
    env = make_env()
    prime_env(env, "reward_timing", "reward_delay_on_positive")

    reward, applied, fault_type = env._apply_reward_fault(1.0, requested_action=1)
    assert applied == 1
    assert fault_type == "reward_delay_on_positive"
    assert reward == 0.0
    assert env._pending_reward > 0.0
    print("PASS: reward_delay_on_positive")


def test_reward_scale_half_on_positive_switch():
    env = make_env()
    prime_env(env, "reward_timing", "reward_scale_half_on_positive_switch")
    env._last_requested_action = 1

    reward, applied, fault_type = env._apply_reward_fault(2.0, requested_action=2)
    assert applied == 1
    assert fault_type == "reward_scale_half_on_positive_switch"
    assert reward == 1.0
    print("PASS: reward_scale_half_on_positive_switch")


def test_reward_zero_on_positive():
    env = make_env()
    prime_env(env, "reward_timing", "reward_zero_on_positive")

    reward, applied, fault_type = env._apply_reward_fault(2.0, requested_action=1)
    assert applied == 1
    assert fault_type == "reward_zero_on_positive"
    assert reward == 0.0
    print("PASS: reward_zero_on_positive")


def test_reward_zero_after_repeat_switch():
    env = make_env()
    prime_env(env, "reward_timing", "reward_zero_after_repeat_switch")
    env._requested_hist = deque([1, 1], maxlen=2)

    reward, applied, fault_type = env._apply_reward_fault(2.0, requested_action=2)

    assert applied == 1
    assert fault_type == "reward_zero_after_repeat_switch"
    assert reward == 0.0
    print("PASS: reward_zero_after_repeat_switch")


def test_reward_delay_after_two_rewards():
    env = make_env()
    prime_env(env, "reward_timing", "reward_delay_after_two_rewards")
    env._reward_hist = deque([1.0, 0.0, 1.0, 0.0], maxlen=4)

    reward, applied, fault_type = env._apply_reward_fault(2.0, requested_action=1)

    assert applied == 1
    assert fault_type == "reward_delay_after_two_rewards"
    assert reward == 0.0
    assert env._pending_reward > 0.0
    print("PASS: reward_delay_after_two_rewards")


def test_early_done_after_success_switch():
    env = make_env()
    prime_env(env, "termination_logic", "early_done_after_success_switch")
    env._after_positive_window = 3
    env._last_requested_action = 1

    done, info, applied, fault_type = env._apply_termination_fault(
        requested_action=2,
        done=False,
        info={"discount": 1.0},
    )
    assert applied == 1
    assert fault_type == "early_done_after_success_switch"
    assert done is True
    assert info["discount"] == 0.0
    print("PASS: early_done_after_success_switch")


def test_early_done_after_repeat_switch():
    env = make_env()
    prime_env(env, "termination_logic", "early_done_after_repeat_switch")
    env._requested_hist = deque([1, 1], maxlen=2)

    done, info, applied, fault_type = env._apply_termination_fault(
        requested_action=2,
        done=False,
        info={"discount": 1.0},
    )
    assert applied == 1
    assert fault_type == "early_done_after_repeat_switch"
    assert done is True
    assert info["discount"] == 0.0
    print("PASS: early_done_after_repeat_switch")


def run_all():
    test_fault_profile_definitions()
    test_drop_to_fallback()
    test_remap_after_success_switch()
    test_delay_after_success()
    test_sticky_after_repeat_switch()
    test_remap_after_repeat_switch()
    test_delay_after_late_episode_switch()
    test_ignore_nonzero_after_reward()
    test_ignore_switch_late_episode()
    test_ignore_nonzero_after_two_rewards()
    test_revisit_action_ignore()
    test_revisit_action_delay()
    test_delayed_switch_failure()
    test_reward_delay_on_positive()
    test_reward_scale_half_on_positive_switch()
    test_reward_zero_on_positive()
    test_reward_zero_after_repeat_switch()
    test_reward_delay_after_two_rewards()
    test_early_done_after_success_switch()
    test_early_done_after_repeat_switch()
    print("\nALL 20 LOW-LEVEL FAULT TESTS PASSED")


if __name__ == "__main__":
    run_all()
