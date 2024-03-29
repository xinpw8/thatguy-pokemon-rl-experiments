import functools

import pufferlib.emulation

from pokemonred_puffer.environment import RedGymEnv
from pokemonred_puffer.stream_wrapper import StreamWrapper


def env_creator(name="pokemon_red"):
    return functools.partial(make, name)


def make(name, **kwargs):
    """Pokemon Red"""
    env = RedGymEnv(kwargs)

    env = StreamWrapper(env, stream_metadata={"user": "\nBET_\n BET\n  _BET\n   _BET\n    _BET\n      BET\n"})
    # Looks like the following will optionally create the object for you
    # Or use theo ne you pass it. I'll just construct it here.
    return pufferlib.emulation.GymnasiumPufferEnv(
        env=env, postprocessor_cls=pufferlib.emulation.BasicPostprocessor
    )