"""Run a single controlled experiment with the course PPO trainer."""
import argparse
import json
from pathlib import Path
import sys

import torch
from mujoco_playground import registry

import train
import train_warp

EXPERIMENTS = {
    'baseline': {},
    'max_iters_500': {'max_iters': 500},
    'box_target_strong': {'box_target': 40.0},
    'qpos_strong': {'robot_target_qpos': 30.0},
    'learning_rate_3e_minus4': {'learning_rate': 0.0003},
}
SEED = 1
NUM_ENVS = 4096


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', choices=EXPERIMENTS)
    name = parser.parse_args().experiment
    change = EXPERIMENTS[name]
    iterations = change.get('max_iters', 1001)
    directory = Path('logs') / name
    directory.mkdir(parents=True, exist_ok=False)
    get_env_config = registry.get_default_config
    build_runner_config = train.build_runner_cfg
    metadata = {
        'experiment': name, 'num_envs': NUM_ENVS, 'max_iters': iterations, 'seed': SEED,
        'change': change, 'backend': 'warp', 'ccd_capacity': train_warp.CCD_CAPACITY,
    }

    def customized_environment(env_name):
        config = get_env_config(env_name)
        if env_name == train.ENV_NAME:
            scales = config.reward_config.scales
            for key in ('box_target', 'robot_target_qpos'):
                if key in change:
                    scales[key] = change[key]
            metadata['reward_scales'] = scales.to_dict()
        return config

    def customized_runner(observation_size):
        config = build_runner_config(observation_size)
        algorithm = config['algorithm']
        metadata['default_learning_rate'] = algorithm['learning_rate']
        if 'learning_rate' in change:
            algorithm['learning_rate'] = change['learning_rate']
        metadata['initial_learning_rate'] = algorithm['learning_rate']
        (directory / 'experiment.json').write_text(json.dumps(metadata, indent=2) + '\n')
        return config

    torch.manual_seed(SEED)
    registry.get_default_config = customized_environment
    train.build_runner_cfg = customized_runner
    sys.argv = ['train.py', '--exp_name', name, '--num_envs', str(NUM_ENVS), '--max_iters',
                str(iterations), '--seed', str(SEED)]
    train_warp.main()


if __name__ == '__main__':
    main()
