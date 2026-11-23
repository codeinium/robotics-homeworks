"""Evaluate course success rate and package auditable experiment results."""
import csv
from importlib.metadata import version
import json
import math
from pathlib import Path
import re
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile

import imageio.v2 as imageio

from evaluate_practice import Evaluation, SEEDS
from inference import build_runner
from train_experiment import EXPERIMENTS


def write_csv(path, records):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def training_stats(path, experiment):
    result = []
    iteration = None
    elapsed = None
    with path.open(encoding='utf-8', errors='replace') as stream:
        for line in stream:
            if match := re.search(r'Learning iteration\s+(\d+)/', line):
                iteration = int(match.group(1))
            elif iteration is not None and (match := re.search(r'Mean reward:\s+([-+\d.eE]+)', line)):
                result.append({'experiment': experiment, 'iteration': iteration, 'mean_reward': float(match.group(1))})
            elif match := re.search(r'Обучение завершено за (\d+) с', line):
                elapsed = int(match.group(1))
    if not result or elapsed is None:
        raise ValueError(f'Incomplete training metrics for {experiment}')
    return result, elapsed


def main():
    device = 'cuda:0'
    logs = Path('logs')
    results = Path('results')
    results.mkdir(exist_ok=True)
    (results / 'clips').mkdir(exist_ok=True)
    evaluator = Evaluation(results, device)
    summary, episode_rows, training_curves, kept_checkpoints = [], [], [], []
    for experiment in EXPERIMENTS:
        directory = logs / experiment
        metadata = json.loads((directory / 'experiment.json').read_text())
        checkpoint = max(directory.glob('model_*.pt'), key=lambda path: int(path.stem.split('_')[-1]))
        iteration = int(checkpoint.stem.split('_')[-1])
        curve, elapsed = training_stats(Path('console') / f'{experiment}.log', experiment)
        expected_iteration = metadata['max_iters'] - 1
        if iteration != expected_iteration or curve[-1]['iteration'] != expected_iteration:
            raise ValueError(f'Unexpected last checkpoint or reward iteration for {experiment}')
        runner = build_runner(str(checkpoint), device)
        policy = runner.get_inference_policy(device=device)
        rollouts = evaluator.rollout(policy, list(SEEDS), None, 0.0, capture=True)
        clip = f'clips/{experiment}_seed42.mp4'
        with imageio.get_writer(results / clip, fps=1 / (2 * evaluator.env.dt), codec='libx264') as writer:
            evaluator.record_frames(writer, rollouts[0][0])
        episodes = [row for _, row in rollouts]
        if len(episodes) != 10 or any(not math.isfinite(row['final_target_distance_m']) for row in episodes):
            raise ValueError(f'Invalid inference results for {experiment}')
        metrics = dict(experiment=experiment, checkpoint=checkpoint.name,
                       final_mean_reward=curve[-1]['mean_reward'], training_seconds=elapsed,
                       episodes=len(episodes), successes=sum(row['delivered'] for row in episodes),
                       lifted=sum(row['lifted'] for row in episodes),
                       approach_within_3cm=sum(row['nearest_gripper_m'] < 0.03 for row in episodes),
                       lost_height=sum(row['returned_below_lift_threshold'] for row in episodes),
                       clip=clip)
        summary.append(metrics)
        training_curves.extend(curve)
        episode_rows.extend(dict(experiment=experiment, **row) for row in episodes)
        kept_checkpoints.append(checkpoint)
        write_csv(results / 'experiments.csv', summary)
        write_csv(results / 'episodes.csv', episode_rows)
        write_csv(results / 'training_curves.csv', training_curves)
        print(metrics, flush=True)
        del runner, policy
    evaluator.renderer.close()

    environment = {
        'gpu': subprocess.getoutput('nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader'),
        'course_commit': subprocess.check_output(['git', '-C', '/content/ap-physicalai-1', 'rev-parse', 'HEAD'], text=True).strip(),
        'packages': {item: version(item) for item in ('playground', 'jax', 'mujoco', 'mujoco-mjx', 'warp-lang', 'torch', 'rsl-rl-lib')},
        'evaluation_seeds': list(SEEDS), 'success_distance_m': 0.05,
    }
    (results / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n')
    files = [Path(name) for name in ('hypothesis.md', 'hypothesis-timestamp.txt', 'train_experiment.py', 'run_training.sh', 'evaluate_and_package.py', 'suite.log')]
    files += sorted(p for p in results.rglob('*') if p.is_file())
    files += sorted(Path('console').glob('*.log'))
    for checkpoint in kept_checkpoints:
        directory = checkpoint.parent
        files += [checkpoint, directory / 'experiment.json', directory / 'config.json']
        files += sorted(directory.glob('events.out.tfevents.*'))
    archive = Path('/content/physical-ai-practice-2-results.zip')
    temporary_archive = archive.with_suffix('.tmp')
    with ZipFile(temporary_archive, 'w', ZIP_DEFLATED, compresslevel=3) as bundle:
        for path in files:
            bundle.write(path, path)
    temporary_archive.replace(archive)
    print(f'Archive ready: {archive} ({archive.stat().st_size:,} bytes)', flush=True)


if __name__ == '__main__':
    main()
