# embodied/wrappers/action_prior.py
import numpy as np
import elements
import embodied

class DiscreteActionPrior(embodied.Env):
  """
  Expose a smaller discrete action space whose indices map to original actions.
  You can repeat actions in `mapping` to increase their sampling probability.
  """

  def __init__(self, env: embodied.Env, name="action", mapping=None):
    self._env = env
    self._name = name
    assert name in env.act_space, f"env.act_space has no key {name}"
    space = env.act_space[name]
    assert space.discrete, f"{name} must be discrete"

    self._orig_n = int(space.high)  # elements.Space(..., low=0, high=n)
    if mapping is None:
      mapping = list(range(self._orig_n))
    self._map = np.array(mapping, dtype=np.int32)
    assert self._map.ndim == 1 and len(self._map) > 0
    assert self._map.min() >= 0 and self._map.max() < self._orig_n

  @property
  def obs_space(self):
    return self._env.obs_space

  @property
  def act_space(self):
    # new action space size = len(mapping)
    spaces = dict(self._env.act_space)
    spaces[self._name] = elements.Space(np.int32, (), 0, len(self._map))
    return spaces

  def step(self, action):
    action = dict(action)
    a = int(action[self._name])
    action[self._name] = int(self._map[a])
    return self._env.step(action)

  def close(self):
    return self._env.close()