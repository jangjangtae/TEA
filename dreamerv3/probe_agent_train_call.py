import pathlib
import sys

folder = pathlib.Path(__file__).parent
sys.path.insert(0, str(folder.parent))
sys.path.insert(1, str(folder.parent.parent))
__package__ = folder.name

import elements
import ruamel.yaml as yaml

from embodied.jax import internal
from . import main as dreamer_main


def main(argv=None):
  configs = elements.Path(folder / 'configs.yaml').read()
  configs = yaml.YAML(typ='safe').load(configs)
  parsed, other = elements.Flags(configs=['defaults']).parse_known(argv)
  config = elements.Config(configs['defaults'])
  for name in parsed.configs:
    config = config.update(configs[name])
  config = elements.Flags(config).parse(other)
  config = config.update(logdir=(
      config.logdir.format(timestamp=elements.timestamp())))

  logdir = elements.Path(config.logdir)
  logdir.mkdir()
  config.save(logdir / 'config.yaml')

  agent = dreamer_main.make_agent(config)
  batch_size = config.batch_size
  length = config.batch_length + config.replay_context
  print('Preparing zero train batch:', batch_size, 'x', length)

  data = agent._zeros(agent.spaces, (batch_size, length))
  data = internal.device_put(data, agent.train_sharded)
  seed = agent._seeds(0, agent.train_mirrored)
  carry = agent.init_train(batch_size)
  data = {**data, 'seed': seed}

  print('Calling agent.train once...')
  carry, outs, mets = agent.train(carry, data)
  del carry
  print('agent.train returned.')
  print('outs:', sorted(outs.keys()))
  print('metrics sample:', sorted(mets.keys())[:20])


if __name__ == '__main__':
  main()
