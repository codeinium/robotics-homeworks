"""Run the course PPO trainer on Warp with a larger CCD buffer."""

import jax
import torch
from mujoco.mjx._src import io as mjx_io
from mujoco_playground import registry

import train


CCD_CAPACITY = 32768


def main():
    if not torch.cuda.is_available() or jax.default_backend() != 'gpu':
        raise RuntimeError('Panda Warp training requires a CUDA GPU in Torch and JAX')
    make_warp_data = mjx_io.mjwp.make_data
    get_default_config = registry.get_default_config

    def allocate_warp_data(*args, **kwargs):
        contact_capacity = kwargs.get('naconmax')
        if contact_capacity is not None:
            kwargs.setdefault('naccdmax', min(contact_capacity, CCD_CAPACITY))
        return make_warp_data(*args, **kwargs)

    def gpu_config(env_name):
        config = get_default_config(env_name)
        if env_name == train.ENV_NAME:
            config.impl = 'warp'
        return config

    mjx_io.mjwp.make_data = allocate_warp_data
    registry.get_default_config = gpu_config
    try:
        print(f'Warp CCD capacity: {CCD_CAPACITY}')
        train.main()
    finally:
        mjx_io.mjwp.make_data = make_warp_data
        registry.get_default_config = get_default_config


if __name__ == '__main__':
    main()
