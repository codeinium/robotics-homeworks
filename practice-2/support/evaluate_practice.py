"""Record six Practice 1 videos and the underlying rollouts."""

import argparse
import csv
from pathlib import Path

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
import torch
from mujoco_playground import registry, wrapper_torch

from config import ENV_NAME
from device_utils import apply_mjx_overrides_to_playground_cfg
from inference import SUCCESS_THRESHOLD, build_runner

SEEDS = range(42, 52)
LIFT_METERS = 0.10
CUBE_POSITIONS = (
    (0.15, 0.10), (0.25, 0.15), (0.28, 0.15), (0.295, 0.15),
    (0.35, 0.22), (0.45, 0.25), (0.55, 0.30), (0.65, 0.35),
    (0.75, 0.40), (0.85, 0.45), (0.35, 0.35), (0.45, 0.40),
    (0.48, 0.12), (0.49, 0.16), (0.55, 0.45), (0.65, -0.22),
    (0.55, -0.25), (0.45, -0.30), (0.93, 0.10),
)
NOISE_LEVELS = (0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00)


def csv_append(path, rows):
    with path.open('a', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerows(rows)


class Evaluation:
    def __init__(self, directory, device):
        self.directory = directory
        self.device = device
        self.config = registry.get_default_config(ENV_NAME)
        apply_mjx_overrides_to_playground_cfg(self.config)
        self.config.impl = 'warp'
        self.config.nconmax = 1024
        self.env = registry.load(ENV_NAME, config=self.config)
        self.reset = jax.jit(jax.vmap(self.reset_episode, in_axes=(0, None, None)))
        self.step = jax.jit(jax.vmap(self.env.step))
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = (0.34, 0.06, 0.25)
        self.camera.distance = 1.42
        self.camera.azimuth = 148
        self.camera.elevation = -24
        self.env.mj_model.vis.quality.offsamples = 0
        self.render_data = mujoco.MjData(self.env.mj_model)
        self.renderer = mujoco.Renderer(self.env.mj_model, height=480, width=640)

    @staticmethod
    def render_pose(state):
        data = state.data
        return data.qpos[0], data.mocap_pos[0], data.mocap_quat[0]

    def record_frames(self, writer, trajectory):
        qpos, mocap_pos, mocap_quat = trajectory
        for position, target, orientation in zip(qpos, mocap_pos, mocap_quat):
            self.render_data.qpos[:] = position
            self.render_data.mocap_pos[:] = target
            self.render_data.mocap_quat[:] = orientation
            mujoco.mj_forward(self.env.mj_model, self.render_data)
            self.renderer.update_scene(self.render_data, camera=self.camera)
            self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
            writer.append_data(self.renderer.render())

    def reset_episode(self, rng, position, override):
        state = self.env.reset(rng)
        address = self.env._obj_qposadr
        qpos = state.data.qpos
        cube_xy = jnp.where(override, position, qpos[address:address + 2])
        qpos = qpos.at[address:address + 2].set(cube_xy)
        data = mjx.forward(self.env.mjx_model, state.data.replace(qpos=qpos))
        return state.replace(data=data, obs=self.env._get_obs(data, state.info))

    def rollout(self, policy, seeds, position, noise, capture):
        coordinates = jnp.asarray(position if position is not None else (0.0, 0.0), dtype=jnp.float32)
        keys = jnp.stack([jax.random.PRNGKey(seed) for seed in seeds])
        state = self.reset(keys, coordinates, position is not None)
        box = self.env._obj_qposadr
        palm = self.env._gripper_site
        targets = state.info['target_pos']
        poses = [state.data.qpos]
        grippers = [state.data.site_xpos[:, palm]]
        rewards, dones = [], []
        sample_indices = [0]
        if capture:
            target_positions = [state.data.mocap_pos]
            target_orientations = [state.data.mocap_quat]
        perturbations = None
        if noise:
            obs = state.obs['state'] if isinstance(state.obs, dict) else state.obs
            scales = []
            for seed in seeds:
                rng = torch.Generator(device=self.device).manual_seed(seed)
                scales.append(torch.randn((self.config.episode_length, obs.shape[-1]),
                                          generator=rng, device=self.device))
            perturbations = noise * torch.stack(scales, dim=1)
        with torch.inference_mode():
            for step in range(self.config.episode_length):
                obs = state.obs['state'] if isinstance(state.obs, dict) else state.obs
                obs = wrapper_torch._jax_to_torch(obs).to(self.device)
                if perturbations is not None:
                    obs = obs + perturbations[step]
                action = policy({'state': obs}).clip(-1.0, 1.0)
                state = self.step(state, wrapper_torch._torch_to_jax(action))
                poses.append(state.data.qpos)
                grippers.append(state.data.site_xpos[:, palm])
                rewards.append(state.reward)
                dones.append(state.done)
                if capture and (step + 1) % 2 == 0:
                    sample_indices.append(step + 1)
                    target_positions.append(state.data.mocap_pos)
                    target_orientations.append(state.data.mocap_quat)
        poses, grippers, rewards, dones, targets = jax.device_get((
            jnp.stack(poses), jnp.stack(grippers), jnp.stack(rewards), jnp.stack(dones), targets))
        if capture:
            target_positions, target_orientations = jax.device_get((
                jnp.stack(target_positions), jnp.stack(target_orientations)))
        result = []
        for index, seed in enumerate(seeds):
            done_steps = np.nonzero(dones[:, index])[0]
            steps = int(done_steps[0]) + 1 if len(done_steps) else len(rewards)
            cube = poses[:steps + 1, index, box:box + 3]
            gripper = grippers[1:steps + 1, index]
            rise = cube[:, 2] - cube[0, 2]
            distance = np.linalg.norm(cube[-1] - targets[index])
            metrics = {
                'seed': seed,
                'steps': steps,
                'reward': float(rewards[:steps, index].sum()),
                'lifted': int(rise.max() > LIFT_METERS),
                'returned_below_lift_threshold': int(rise.max() > LIFT_METERS >= rise[-1]),
                'delivered': int(distance < SUCCESS_THRESHOLD),
                'max_lift_m': float(rise.max()),
                'nearest_gripper_m': float(np.linalg.norm(gripper - cube[1:], axis=1).min()),
                'final_target_distance_m': float(distance),
            }
            trajectory = None
            if capture:
                selected = [step for step in sample_indices if step <= steps]
                end = len(selected)
                trajectory = (poses[selected, index], target_positions[:end, index], target_orientations[:end, index])
            result.append((trajectory, metrics))
        return result

    def measure(self, label, checkpoint, policy, episodes=10, position=None, noise=0.0, video=None):
        records = []
        writer = None
        if video:
            writer = imageio.get_writer(self.directory / video, fps=1 / (2 * self.env.dt), codec='libx264')
        try:
            seeds = list(SEEDS)[:episodes]
            for trajectory, metrics in self.rollout(policy, seeds, position, noise, writer is not None):
                records.append(metrics)
                if writer is not None:
                    self.record_frames(writer, trajectory)
        finally:
            if writer is not None:
                writer.close()
        x, y = position if position is not None else (None, None)
        params = {'run': label, 'checkpoint': checkpoint, 'cube_x': x, 'cube_y': y, 'obs_noise': noise}
        summary = dict(params, episodes=episodes, lifts=sum(row['lifted'] for row in records),
                       delivered=sum(row['delivered'] for row in records),
                       mean_reward=sum(row['reward'] for row in records) / episodes,
                       video=video or '')
        csv_append(self.directory / 'evaluation.csv', [summary])
        csv_append(self.directory / 'episodes.csv', [dict(params, **row) for row in records])
        print(summary, flush=True)
        return summary

    def select_ood(self, policy, baseline, label, candidates, parameter):
        target = baseline['lifts'] / 2
        attempts = []

        def measure_candidate(value):
            result = self.measure(f'{label}_{len(attempts) + 1}', 1000, policy, **{parameter: value})
            lifts = result['lifts']
            attempts.append((value, lifts))
            return lifts

        for value in candidates:
            lifts = measure_candidate(value)
            if target > 0 and abs(lifts - target) <= 0.5:
                return value

        brackets = [(0.0, baseline['lifts']), *attempts] if parameter == 'noise' else attempts
        for (first, first_lifts), (second, second_lifts) in zip(brackets, brackets[1:]):
            if parameter == 'position':
                x1, y1 = first
                x2, y2 = second
                outside_same_edge = (max(x1, x2) < 0.5 or min(x1, x2) > 0.9
                                     or max(y1, y2) < -0.2 or min(y1, y2) > 0.2)
                if not outside_same_edge:
                    continue
            if (first_lifts - target) * (second_lifts - target) >= 0:
                continue
            for _ in range(5):
                midpoint = (np.asarray(first) + np.asarray(second)) / 2
                value = float(midpoint) if midpoint.ndim == 0 else tuple(midpoint.tolist())
                lifts = measure_candidate(value)
                if abs(lifts - target) <= 0.5:
                    return value
                if (first_lifts - target) * (lifts - target) > 0:
                    first, first_lifts = value, lifts
                else:
                    second = value
            break
        return min(attempts, key=lambda attempt: abs(attempt[1] - target))[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    directory = args.run_directory.resolve()
    checkpoints = {step: directory / f'model_{step}.pt' for step in (50, 500, 1000)}
    for path in checkpoints.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    for name in ('evaluation.csv', 'episodes.csv'):
        if (directory / name).exists():
            raise FileExistsError(f'Archive {directory / name} before running again')
    evaluation = Evaluation(directory, args.device)
    for step, video in ((50, 'video_iter50.mp4'), (500, 'video_iter500.mp4'), (1000, 'video.mp4')):
        runner = build_runner(str(checkpoints[step]), args.device)
        policy = runner.get_inference_policy(device=args.device)
        evaluation.measure(f'checkpoint_{step}', step, policy, episodes=3, video=video)
        del runner
    baseline = evaluation.measure('in_domain', 1000, policy, video='exp1_indomain.mp4')
    position = (evaluation.select_ood(policy, baseline, 'cube_position', CUBE_POSITIONS, 'position')
                if baseline['lifts'] else CUBE_POSITIONS[1])
    evaluation.measure('selected_cube_position', 1000, policy, position=position, video='exp2_far.mp4')
    noise = (evaluation.select_ood(policy, baseline, 'sensor_noise', NOISE_LEVELS, 'noise')
             if baseline['lifts'] else 0.10)
    evaluation.measure('selected_sensor_noise', 1000, policy, noise=noise, video='exp3_noise.mp4')
    evaluation.renderer.close()


if __name__ == '__main__':
    main()
