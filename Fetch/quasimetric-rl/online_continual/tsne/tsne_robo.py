import argparse
import os
import sys

import numpy as np
import torch


ROBOT_STATE_INDICES = {
	"reach": np.arange(10, dtype=np.int64),
	"push": np.asarray([0, 1, 2, 9, 10, 20, 21, 22, 23, 24], dtype=np.int64),
	"pick-and-place": np.asarray([0, 1, 2, 9, 10, 20, 21, 22, 23, 24], dtype=np.int64),
	"slide": np.asarray([0, 1, 2, 9, 10, 20, 21, 22, 23, 24], dtype=np.int64),
}
ROBOT_STATE_FEATURE_NAMES = (
	"gripper_x",
	"gripper_y",
	"gripper_z",
	"left_finger_position",
	"right_finger_position",
	"gripper_velocity_x",
	"gripper_velocity_y",
	"gripper_velocity_z",
	"left_finger_velocity",
	"right_finger_velocity",
)

if __package__:
	from ..final_eval_cqrl import (
		build_eval_agent,
		checkpoint_plan,
		find_training_config,
		load_training_config,
		resolve_agent_kind,
		validate_checkpoints,
	)
	from ..main import (
		FetchGoalEnvSequence,
		add_common_args,
		add_quasimetric_args,
		add_sac_args,
		add_student_qrl_args,
		make_env_kwargs,
		set_seed_everywhere,
	)
else:
	sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	from final_eval_cqrl import (
		build_eval_agent,
		checkpoint_plan,
		find_training_config,
		load_training_config,
		resolve_agent_kind,
		validate_checkpoints,
	)
	from main import (
		FetchGoalEnvSequence,
		add_common_args,
		add_quasimetric_args,
		add_sac_args,
		add_student_qrl_args,
		make_env_kwargs,
		set_seed_everywhere,
	)


def parse_args():
	bootstrap_parser = argparse.ArgumentParser(add_help=False)
	bootstrap_parser.add_argument("--run_dir", type=str, default=None)
	bootstrap_parser.add_argument("--config", type=str, default=None)
	bootstrap_args, _ = bootstrap_parser.parse_known_args()

	run_dir = os.path.abspath(bootstrap_args.run_dir) if bootstrap_args.run_dir else None
	config_path = find_training_config(run_dir, bootstrap_args.config) if run_dir else None
	training_config = load_training_config(config_path)

	parser = argparse.ArgumentParser(
		description="Visualize robot-only rollout states from a continual Fetch agent with t-SNE"
	)
	parser.add_argument("--run_dir", type=str, required=True)
	parser.add_argument("--config", type=str, default=None)
	parser.add_argument("--model_dir", type=str, default=None)
	parser.add_argument("--output", type=str, default=None)
	parser.add_argument("--data_output", type=str, default=None)
	parser.add_argument("--checkpoint", choices=["all", "final"], default="final")
	parser.add_argument(
		"--agent_kind",
		choices=["auto", "meta", "student"],
		default="auto",
		help="Auto uses the meta agent for buffer methods and the student otherwise.",
	)
	parser.add_argument("--eval_seed", type=int, default=None)
	parser.add_argument("--rollout_episodes", type=int, default=20)
	parser.add_argument(
		"--success_only",
		action="store_true",
		help="Only visualize trajectories that reach is_success at least once.",
	)
	parser.add_argument(
		"--sample_every",
		type=int,
		default=1,
		help="Keep one state representation every N environment steps; 1 keeps every step.",
	)
	parser.add_argument(
		"--max_samples_per_task",
		type=int,
		default=0,
		help="Optionally subsample each task to this many states; <= 0 keeps all trajectories and steps.",
	)
	parser.add_argument("--perplexity", type=float, default=30.0)
	parser.add_argument("--tsne_iterations", type=int, default=1000)
	parser.add_argument(
		"--tsne_dim",
		type=int,
		choices=[2, 3],
		default=2,
		help="Number of t-SNE output dimensions; choose 3 for a 3D plot.",
	)
	parser.add_argument("--dpi", type=int, default=200)
	add_common_args(parser)
	add_sac_args(parser)
	add_student_qrl_args(parser)
	add_quasimetric_args(parser)
	parser.set_defaults(**training_config)
	args = parser.parse_args()

	args.run_dir = os.path.abspath(args.run_dir)
	args.config = config_path
	args.model_dir = os.path.abspath(args.model_dir or os.path.join(args.run_dir, "model"))
	run_name = os.path.basename(os.path.normpath(args.run_dir))
	checkpoint_tag = "all" if args.checkpoint == "all" else "final"
	selection_tag = "_success" if args.success_only else ""
	dimension_tag = "_3d" if args.tsne_dim == 3 else ""
	args.output = os.path.abspath(
		args.output
		or os.path.join(
			args.run_dir,
			f"{run_name}_{checkpoint_tag}{selection_tag}_robot_state_tsne{dimension_tag}.png",
		)
	)
	args.data_output = os.path.abspath(
		args.data_output
		or os.path.join(
			args.run_dir,
			f"{run_name}_{checkpoint_tag}{selection_tag}_robot_state_tsne{dimension_tag}.npz",
		)
	)
	args.eval_seed = args.seed if args.eval_seed is None else args.eval_seed

	if args.rollout_episodes <= 0:
		parser.error("--rollout_episodes must be positive")
	if args.sample_every <= 0:
		parser.error("--sample_every must be positive")
	if args.perplexity <= 0:
		parser.error("--perplexity must be positive")
	if args.tsne_iterations <= 0:
		parser.error("--tsne_iterations must be positive")
	return args


def robot_state_features(state_obs, task_name):
	state_obs = np.asarray(state_obs, dtype=np.float32).reshape(-1)
	indices = ROBOT_STATE_INDICES[task_name]
	if state_obs.shape[0] <= indices[-1]:
		raise ValueError(
			f"Task '{task_name}' needs state index {indices[-1]}, but the state has "
			f"only {state_obs.shape[0]} dimensions."
		)
	return state_obs[indices]


def rollout_task(env, agent, task_name, episodes, sample_every, seed, success_only):
	env.set_task(task_name)
	agent.eval()
	representations = []
	state_observations = []
	episode_indices = []
	step_indices = []
	episode_successes = []
	successful_episode_count = 0

	try:
		for episode_idx in range(episodes):
			obs, _ = env.reset(seed=seed + episode_idx)
			step_idx = 0
			episode_representations = []
			episode_state_observations = []
			episode_step_indices = []
			episode_success = False
			while True:
				state_obs = obs["observation"] if isinstance(obs, dict) else obs
				state_obs = np.asarray(state_obs, dtype=np.float32)
				if step_idx % sample_every == 0:
					episode_representations.append(robot_state_features(state_obs, task_name))
					episode_state_observations.append(state_obs.copy())
					episode_step_indices.append(step_idx)

				action = env._evaluate_action(agent, obs)
				obs, _, terminated, truncated, info = env.no_count_step(action)
				episode_success = episode_success or bool(info.get("is_success", False))
				step_idx += 1
				if terminated or truncated:
					break

			if episode_success:
				successful_episode_count += 1
			if success_only and not episode_success:
				continue
			representations.extend(episode_representations)
			state_observations.extend(episode_state_observations)
			episode_indices.extend([episode_idx] * len(episode_representations))
			step_indices.extend(episode_step_indices)
			episode_successes.extend([episode_success] * len(episode_representations))
	finally:
		agent.train()

	if not representations:
		if success_only:
			print(f"Skipped task {task_name}: 0/{episodes} successful episodes")
			return None
		raise RuntimeError(f"No state representations were collected for task '{task_name}'.")
	print(f"Task {task_name}: {successful_episode_count}/{episodes} successful episodes")
	return {
		"representations": np.asarray(representations, dtype=np.float32),
		"state_observations": np.asarray(state_observations, dtype=np.float32),
		"episode_indices": np.asarray(episode_indices, dtype=np.int32),
		"step_indices": np.asarray(step_indices, dtype=np.int32),
		"episode_successes": np.asarray(episode_successes, dtype=np.bool_),
	}


def subsample_task(data, max_samples, random_state):
	sample_count = len(data["representations"])
	if max_samples <= 0 or sample_count <= max_samples:
		return data
	indices = np.sort(random_state.choice(sample_count, size=max_samples, replace=False))
	return {key: values[indices] for key, values in data.items()}


def collect_representations(args, env, agent, plan):
	random_state = np.random.RandomState(args.eval_seed)
	collected = []
	for stage_idx, model_name, task_indices in plan:
		print(f"Loading checkpoint: {model_name}")
		agent.load(args.model_dir, model_name)
		for task_idx in task_indices:
			task_name = env.env_list[task_idx]
			data = rollout_task(
				env,
				agent,
				task_name,
				args.rollout_episodes,
				args.sample_every,
				args.eval_seed,
				args.success_only,
			)
			if data is None:
				continue
			data = subsample_task(data, args.max_samples_per_task, random_state)
			data.update(
				{
					"stage_indices": np.full(len(data["representations"]), stage_idx, dtype=np.int32),
					"task_indices": np.full(len(data["representations"]), task_idx, dtype=np.int32),
					"task_names": np.full(len(data["representations"]), task_name),
					"checkpoints": np.full(len(data["representations"]), model_name),
				}
			)
			collected.append(data)
			print(
				f"Collected stage {stage_idx}, task {task_name}: "
				f"{len(data['representations'])} representations"
			)

	if not collected:
		raise RuntimeError(
			"No robot states were collected. Increase --rollout_episodes "
			"or omit --success_only."
		)
	keys = collected[0].keys()
	return {key: np.concatenate([item[key] for item in collected], axis=0) for key in keys}


def compute_tsne(representations, perplexity, iterations, seed, dimensions):
	try:
		from sklearn.manifold import TSNE
	except ImportError as error:
		raise ImportError("Install scikit-learn to compute t-SNE embeddings.") from error

	sample_count = len(representations)
	if sample_count < 3:
		raise ValueError(f"t-SNE needs at least 3 samples, but only {sample_count} were collected.")
	effective_perplexity = min(float(perplexity), float(sample_count - 1))
	print(f"Running t-SNE on {sample_count} robot states (perplexity={effective_perplexity:g})")
	return TSNE(
		n_components=dimensions,
		perplexity=effective_perplexity,
		max_iter=iterations,
		init="pca",
		learning_rate="auto",
		random_state=seed,
	).fit_transform(representations)


def plot_tsne(embedding, task_names, stage_indices, episode_indices, step_indices, output, dpi):
	try:
		import matplotlib.pyplot as plt
	except ImportError as error:
		raise ImportError("Install matplotlib to save the t-SNE plot.") from error

	dimensions = embedding.shape[1]
	figure = plt.figure(figsize=(10, 8))
	axis = figure.add_subplot(111, projection="3d" if dimensions == 3 else None)
	unique_pairs = list(dict.fromkeys(zip(stage_indices.tolist(), task_names.tolist())))
	colors = plt.get_cmap("tab10", max(len(unique_pairs), 1))
	for color_idx, (stage_idx, task_name) in enumerate(unique_pairs):
		mask = (stage_indices == stage_idx) & (task_names == task_name)
		for episode_idx in np.unique(episode_indices[mask]):
			trajectory_mask = mask & (episode_indices == episode_idx)
			trajectory_indices = np.flatnonzero(trajectory_mask)
			trajectory_indices = trajectory_indices[np.argsort(step_indices[trajectory_indices])]
			trajectory = embedding[trajectory_indices]
			plot_coordinates = [trajectory[:, idx] for idx in range(dimensions)]
			axis.plot(
				*plot_coordinates,
				color=colors(color_idx),
				alpha=0.16,
				linewidth=0.7,
				zorder=1,
			)
		label = task_name if len(set(stage_indices.tolist())) == 1 else f"stage {stage_idx}: {task_name}"
		points = embedding[mask]
		scatter_coordinates = [points[:, idx] for idx in range(dimensions)]
		axis.scatter(
			*scatter_coordinates,
			s=14,
			alpha=0.65,
			color=colors(color_idx),
			label=label,
			edgecolors="none",
			zorder=2,
		)
	axis.set_title("Robot states from task trajectories")
	axis.set_xlabel("t-SNE 1")
	axis.set_ylabel("t-SNE 2")
	if dimensions == 3:
		axis.set_zlabel("t-SNE 3")
	axis.legend(frameon=False, markerscale=1.5)
	axis.grid(alpha=0.15)
	figure.tight_layout()
	os.makedirs(os.path.dirname(output), exist_ok=True)
	figure.savefig(output, dpi=dpi, bbox_inches="tight")
	plt.close(figure)


def main():
	args = parse_args()
	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
	args.seed = args.eval_seed
	set_seed_everywhere(args.seed)

	env = FetchGoalEnvSequence(**make_env_kwargs(args))
	try:
		agent_kind = resolve_agent_kind(args)
		plan = checkpoint_plan(args, len(env.env_list), agent_kind)
		validate_checkpoints(args.model_dir, plan, agent_kind)
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		agent = build_eval_agent(args, env, device, agent_kind)

		print("run_dir:", args.run_dir)
		print("model_dir:", args.model_dir)
		print("agent_kind:", agent_kind)
		print("tasks:", list(env.env_list))
		data = collect_representations(args, env, agent, plan)
	finally:
		env.close()

	embedding = compute_tsne(
		data["representations"],
		args.perplexity,
		args.tsne_iterations,
		args.eval_seed,
		args.tsne_dim,
	)
	os.makedirs(os.path.dirname(args.data_output), exist_ok=True)
	np.savez_compressed(
		args.data_output,
		tsne=embedding,
		robot_feature_names=np.asarray(ROBOT_STATE_FEATURE_NAMES),
		**data,
	)
	plot_tsne(
		embedding,
		data["task_names"],
		data["stage_indices"],
		data["episode_indices"],
		data["step_indices"],
		args.output,
		args.dpi,
	)
	print("plot:", args.output)
	print("data:", args.data_output)


if __name__ == "__main__":
	sys.exit(main())
'''
cd Fetch/quasimetric-rl
conda activate RLL3

python -m online_continual.tsne_robo \
  --run_dir online_continual/results/fetch_continual_push-pick-and-place_seed0_buffer_gc-sparse \
  --checkpoint final \
  --agent_kind meta \
  --eval_seed 0 \
  --gpu 0 \
  --rollout_episodes 100 \
	--success_only \
  --sample_every 1 \
	--max_samples_per_task 0 \
  --perplexity 30 \
	--tsne_iterations 1000 \
	--tsne_dim 3

	python -m online_continual.tsne_robo \
  --run_dir online_continual/results/fetch_continual_push-slide_seed0_buffer_gc-sparse_slide-scale0.8 \
  --checkpoint final \
  --agent_kind meta \
  --eval_seed 0 \
  --gpu 0 \
  --rollout_episodes 100 \
	--success_only \
  --sample_every 1 \
	--max_samples_per_task 0 \
  --perplexity 30 \
	--tsne_iterations 1000 \
	--tsne_dim 3

	
	python -m online_continual.tsne_robo \
  --run_dir online_continual/results/fetch_continual_push-pick-and-place_seed0_buffer_gc-sparse \
  --checkpoint final \
  --agent_kind meta \
  --eval_seed 0 \
  --gpu 0 \
  --rollout_episodes 50 \
  --sample_every 1 \
  --max_samples_per_task 0 \
  --perplexity 30 \
  --tsne_iterations 1000 \
  --tsne_dim 3
	
	python -m online_continual.tsne_robo \
    --run_dir online_continual/results/fetch_continual_push-slide_seed0_buffer_gc-sparse_slide-scale0.8 \
    --checkpoint final \
    --agent_kind meta \
    --eval_seed 0 \
    --gpu 0 \
    --rollout_episodes 50 \
    --sample_every 1 \
    --max_samples_per_task 0 \
    --perplexity 30 \
    --tsne_iterations 1000 \
    --tsne_dim 3
	
	python -m online_continual.tsne_robo \
    --run_dir online_continual/results/fetch_continual_push-slide_seed0_buffer_gc-sparse_slide-scale0.8 \
    --checkpoint final \
    --agent_kind student \
    --eval_seed 0 \
    --gpu 0 \
    --rollout_episodes 50 \
    --sample_every 1 \
    --max_samples_per_task 0 \
    --perplexity 30 \
    --tsne_iterations 1000 \
    --tsne_dim 3
	
	cd Fetch/quasimetric-rl
conda activate RLL3

python -m online_continual.tsne_robo \
  --run_dir online_continual/results/fetch_continual_push-pick-and-place_seed0_buffer_gc-sparse \
  --checkpoint final \
  --agent_kind student \
  --eval_seed 0 \
  --gpu 0 \
  --rollout_episodes 50 \
  --sample_every 1 \
  --perplexity 30 \
  --tsne_iterations 1000 \
  --tsne_dim 3
	
  python -m online_continual.tsne.tsne_robo \
	  --run_dir online_continual/results/cqrl_wd/fetch_cqrl_push-slide-pick-and-place_seed0_gc-sparse_slide-scale0.795 \
	  --checkpoint final \
	  --agent_kind meta \
	  --eval_seed 0 \
	  --gpu 0 \
	  --rollout_episodes 100 \
	  --sample_every 1 \
	  --max_samples_per_task 0 \
	  --perplexity 30 \
	  --tsne_iterations 2000 \
	  --tsne_dim 3 \
	  --success_only


'''