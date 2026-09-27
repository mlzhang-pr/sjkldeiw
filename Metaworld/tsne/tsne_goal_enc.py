"""Visualize the last CQRL meta agent's goal-encoder latent trajectories."""

import json
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
METAWORLD_DIR = os.path.dirname(SCRIPT_DIR)
if METAWORLD_DIR not in sys.path:
	sys.path.insert(0, METAWORLD_DIR)

from final_eval_main3 import (  # noqa: E402
	build_eval_agent,
	build_parser as build_eval_parser,
	build_rollout_goal_buffers,
	make_env_kwargs,
	make_log_name,
	obs_vector,
	resolve_eval_model_dir,
	select_meta_goal,
	set_seed_everywhere,
	task_model_name,
	vector_observation_space,
)
from envs.metaworld_env import MetaWorldSingleEnvSequence  # noqa: E402


ROLLOUT_GOAL_SOURCES = {
	"rollout_task",
	"rollout_any",
	"rollout_task_buffer",
	"rollout_buffer",
}


def positive_float(value):
	value = float(value)
	if value <= 0:
		raise ValueError("value must be positive")
	return value


def parse_args():
	parser = build_eval_parser(
		"t-SNE visualization of CQRL goal-encoder latents from the last meta agent"
	)
	parser.set_defaults(
		eval_agent_type="meta",
		num_eval_runs=1,
		rollout_goal_episodes=1,
		meta_goal_source="rollout_task",
	)
	parser.add_argument(
		"--tsne_output_dir",
		type=str,
		default=os.path.join(SCRIPT_DIR, "results"),
		help="Directory for plots, metadata, and raw latent arrays",
	)
	parser.add_argument(
		"--tsne_prefix",
		type=str,
		default=None,
		help="Output filename prefix; defaults to the checkpoint name plus goal_encoder_tsne",
	)
	parser.add_argument("--tsne_perplexity", type=positive_float, default=30.0)
	parser.add_argument("--tsne_max_iter", type=int, default=1000)
	parser.add_argument("--tsne_point_size", type=positive_float, default=14.0)
	parser.add_argument("--tsne_dpi", type=int, default=220)
	parser.add_argument(
		"--connect_trajectories",
		type=int,
		default=1,
		choices=[0, 1],
		help="Connect consecutive latent points within each episode",
	)
	return parser.parse_args()


def encode_observation(agent, observation):
	observation = np.asarray(observation, dtype=np.float32)
	with torch.inference_mode():
		latent = agent.encode_goal(observation)
	latent = latent.detach().cpu().numpy()
	if latent.ndim != 2 or latent.shape[0] != 1:
		raise ValueError(f"Expected one encoded observation, got latent shape {latent.shape}")
	return latent[0]


def collect_task_rollouts(env, agent, task_idx, task_name, num_episodes):
	"""Collect one goal-encoder latent for every action step in evaluation."""
	test_env = env._wrap_env(env._make_base_env(), eval_mode=True)
	initial_reset_kwargs = {"seed": env.current_seed} if env.current_seed is not None else {}
	latents = []
	metadata = defaultdict(list)
	episode_summaries = []
	agent.eval()

	try:
		for episode_idx in range(num_episodes):
			reset_kwargs = initial_reset_kwargs if episode_idx == 0 else {}
			obs, _ = test_env.reset(**reset_kwargs)
			if env._uses_obs_normalization():
				obs = env._normalize_obs(obs)

			episode_return = 0.0
			episode_success = False
			step_idx = 0
			while True:
				current_obs = np.asarray(obs_vector(obs), dtype=np.float32)
				with torch.inference_mode():
					latent = encode_observation(agent, current_obs)
					action = env._evaluate_action(agent, obs)

				next_obs, reward, terminated, truncated, info = test_env.step(action)
				done = bool(terminated or truncated)
				episode_return += float(reward)
				step_success = bool(info.get("success", info.get("is_success", False)))
				episode_success = episode_success or step_success

				latents.append(latent)
				metadata["task_idx"].append(task_idx)
				metadata["task_name"].append(task_name)
				metadata["episode"].append(episode_idx)
				metadata["step"].append(step_idx)
				metadata["reward"].append(float(reward))
				metadata["success"].append(step_success)
				metadata["done"].append(done)

				if env._uses_obs_normalization():
					next_obs = env._normalize_obs(next_obs)
				obs = next_obs
				step_idx += 1

				if done:
					recorded_return = info.get("episode", {}).get("r", episode_return)
					episode_summaries.append(
						{
							"task_idx": task_idx + 1,
							"task_name": task_name,
							"episode": episode_idx,
							"steps": step_idx,
							"return": float(recorded_return),
							"success": episode_success,
						}
					)
					break
	finally:
		agent.train()
		test_env.close()

	if not latents:
		raise RuntimeError(f"No rollout steps collected for task {task_name}")
	return np.stack(latents), dict(metadata), episode_summaries


def concatenate_metadata(metadata_parts):
	keys = metadata_parts[0].keys()
	return {
		key: np.concatenate([np.asarray(part[key]) for part in metadata_parts])
		for key in keys
	}


def fit_tsne(trajectory_latents, goal_latents, args):
	all_latents = np.concatenate([trajectory_latents, goal_latents], axis=0)
	if all_latents.shape[0] < 3:
		raise ValueError("t-SNE requires at least three latent points")
	if not np.isfinite(all_latents).all():
		raise ValueError("Goal-encoder latents contain NaN or infinite values")
	if args.tsne_max_iter < 250:
		raise ValueError("--tsne_max_iter must be at least 250")

	effective_perplexity = min(float(args.tsne_perplexity), float(all_latents.shape[0] - 1))
	print(
		f"fitting t-SNE: samples={all_latents.shape[0]} latent_dim={all_latents.shape[1]} "
		f"perplexity={effective_perplexity}"
	)
	embedding = TSNE(
		n_components=2,
		perplexity=effective_perplexity,
		learning_rate="auto",
		max_iter=args.tsne_max_iter,
		init="pca",
		random_state=args.seed,
	).fit_transform(all_latents)
	n_trajectory_points = trajectory_latents.shape[0]
	return (
		embedding[:n_trajectory_points],
		embedding[n_trajectory_points:],
		effective_perplexity,
	)


def plot_tsne(trajectory_embedding, goal_embedding, metadata, task_names, checkpoint, args, path):
	fig, ax = plt.subplots(figsize=(12, 8.5), constrained_layout=True)
	cmap = plt.get_cmap("tab20", max(len(task_names), 1))

	for task_idx, task_name in enumerate(task_names):
		color = cmap(task_idx)
		task_mask = metadata["task_idx"] == task_idx
		task_points = trajectory_embedding[task_mask]
		if bool(args.connect_trajectories):
			for episode_idx in np.unique(metadata["episode"][task_mask]):
				episode_mask = task_mask & (metadata["episode"] == episode_idx)
				episode_points = trajectory_embedding[episode_mask]
				ax.plot(
					episode_points[:, 0],
					episode_points[:, 1],
					color=color,
					alpha=0.24,
					linewidth=0.8,
					zorder=1,
				)
		ax.scatter(
			task_points[:, 0],
			task_points[:, 1],
			s=args.tsne_point_size,
			color=color,
			alpha=0.72,
			linewidths=0,
			label=f"{task_idx + 1}: {task_name}",
			zorder=2,
		)
		ax.scatter(
			goal_embedding[task_idx, 0],
			goal_embedding[task_idx, 1],
			s=180,
			color=color,
			marker="*",
			edgecolors="black",
			linewidths=0.9,
			zorder=4,
		)

	ax.set_title(f"CQRL goal-encoder latent trajectories\n{checkpoint}", fontsize=16, pad=12)
	ax.set_xlabel("t-SNE dimension 1")
	ax.set_ylabel("t-SNE dimension 2")
	ax.grid(color="#d9d9d9", linewidth=0.6, alpha=0.55)
	ax.spines[["top", "right"]].set_visible(False)
	legend_handles, legend_labels = ax.get_legend_handles_labels()
	legend_handles.append(
		Line2D(
			[0],
			[0],
			marker="*",
			color="none",
			markerfacecolor="#bdbdbd",
			markeredgecolor="black",
			markersize=13,
			label="Fast-agent rollout goal",
		)
	)
	legend_labels.append("Fast-agent rollout goal")
	ax.legend(
		legend_handles,
		legend_labels,
		loc="center left",
		bbox_to_anchor=(1.01, 0.5),
		frameon=False,
		fontsize=9,
	)
	fig.savefig(path, dpi=args.tsne_dpi, bbox_inches="tight", facecolor="white")
	plt.close(fig)


def save_outputs(
	trajectory_latents,
	goal_latents,
	goal_observations,
	trajectory_embedding,
	goal_embedding,
	metadata,
	goal_sources,
	episode_summaries,
	task_names,
	checkpoint,
	effective_perplexity,
	args,
):
	os.makedirs(args.tsne_output_dir, exist_ok=True)
	prefix = args.tsne_prefix or f"{checkpoint}_goal_encoder_tsne"
	base_path = os.path.join(args.tsne_output_dir, prefix)

	plot_tsne(
		trajectory_embedding,
		goal_embedding,
		metadata,
		task_names,
		checkpoint,
		args,
		f"{base_path}.png",
	)
	plot_tsne(
		trajectory_embedding,
		goal_embedding,
		metadata,
		task_names,
		checkpoint,
		args,
		f"{base_path}.pdf",
	)

	trajectory_frame = pd.DataFrame(
		{
			"point_type": "rollout_step",
			"task_idx": metadata["task_idx"] + 1,
			"task_name": metadata["task_name"],
			"episode": metadata["episode"],
			"step": metadata["step"],
			"reward": metadata["reward"],
			"success": metadata["success"],
			"done": metadata["done"],
			"tsne_x": trajectory_embedding[:, 0],
			"tsne_y": trajectory_embedding[:, 1],
		}
	)
	goal_frame = pd.DataFrame(
		{
			"point_type": "behavior_goal",
			"task_idx": np.arange(1, len(task_names) + 1),
			"task_name": task_names,
			"episode": -1,
			"step": -1,
			"reward": np.nan,
			"success": np.nan,
			"done": np.nan,
			"tsne_x": goal_embedding[:, 0],
			"tsne_y": goal_embedding[:, 1],
			"goal_source": goal_sources,
		}
	)
	pd.concat([trajectory_frame, goal_frame], ignore_index=True).to_csv(
		f"{base_path}_points.csv",
		index=False,
	)
	pd.DataFrame(episode_summaries).to_csv(f"{base_path}_episodes.csv", index=False)
	np.savez_compressed(
		f"{base_path}_latents.npz",
		trajectory_latents=trajectory_latents,
		goal_latents=goal_latents,
		goal_observations=goal_observations,
		trajectory_tsne=trajectory_embedding,
		goal_tsne=goal_embedding,
		task_idx=metadata["task_idx"],
		task_name=metadata["task_name"],
		episode=metadata["episode"],
		step=metadata["step"],
		goal_source=np.asarray(goal_sources),
	)

	config = dict(vars(args))
	config.update(
		{
			"checkpoint": checkpoint,
			"effective_tsne_perplexity": effective_perplexity,
			"trajectory_points": int(trajectory_latents.shape[0]),
			"goal_points": int(goal_latents.shape[0]),
			"latent_dim": int(trajectory_latents.shape[1]),
		}
	)
	with open(f"{base_path}_config.json", "w", encoding="utf-8") as config_file:
		json.dump(config, config_file, indent=2, default=str)

	print(f"saved t-SNE plot: {base_path}.png")
	print(f"saved raw latents: {base_path}_latents.npz")
	return base_path


def main():
	args = parse_args()
	if args.eval_agent_type != "meta":
		raise ValueError("This script requires --eval_agent_type meta")

	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
	set_seed_everywhere(args.seed)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	env_kwargs = make_env_kwargs(args)
	env = MetaWorldSingleEnvSequence(**env_kwargs)
	if not hasattr(env, "env_list"):
		env.env_list = [env.base_task_name]

	num_agents = args.num_agents if args.num_agents is not None else len(env.env_list)
	if not 1 <= num_agents <= len(env.env_list):
		raise ValueError(f"num_agents={num_agents} must be in [1, {len(env.env_list)}]")
	agent_idx = num_agents - 1 if args.agent_idx is None else args.agent_idx
	if not 0 <= agent_idx < num_agents:
		raise ValueError(f"agent_idx={agent_idx} must be in [0, {num_agents - 1}]")

	model_dir = resolve_eval_model_dir(args)
	log_name = make_log_name(args)
	checkpoint = task_model_name(log_name, agent_idx, use_meta=True)
	obs_space = vector_observation_space(env.env.observation_space)
	action_shape = env.env.action_space.shape

	print(f"device: {device}")
	print(f"last meta checkpoint: {checkpoint}")
	print(f"tasks: {agent_idx + 1}")
	rollout_goal_buffers = None
	if args.meta_goal_source in ROLLOUT_GOAL_SOURCES or args.meta_goal_fallback in ROLLOUT_GOAL_SOURCES:
		rollout_goal_buffers = build_rollout_goal_buffers(
			log_name,
			model_dir,
			obs_space,
			action_shape,
			device,
			env_kwargs,
			env.env_list,
			agent_idx + 1,
			args,
		)

	agent = build_eval_agent(
		obs_space.shape[0],
		action_shape[0],
		model_dir,
		checkpoint,
		device,
		args,
	)
	trajectory_parts = []
	metadata_parts = []
	goal_latents = []
	goal_observations = []
	goal_sources = []
	episode_summaries = []
	task_names = list(env.env_list[: agent_idx + 1])

	for task_idx, task_name in enumerate(task_names):
		env.set_task(task_name)
		goal_info = select_meta_goal(
			agent,
			env,
			agent_idx,
			task_idx,
			args,
			rollout_goal_buffers,
		)
		behavior_goal = getattr(agent, "behavior_goal", None)
		if behavior_goal is None:
			raise ValueError(
				f"Task {task_name} has no behavior goal; use --meta_goal_source rollout_task"
			)
		goal_observations.append(np.asarray(behavior_goal, dtype=np.float32))
		goal_latents.append(encode_observation(agent, behavior_goal))
		goal_sources.append(goal_info["source"])

		task_latents, task_metadata, task_summaries = collect_task_rollouts(
			env,
			agent,
			task_idx,
			task_name,
			args.num_eval_runs,
		)
		trajectory_parts.append(task_latents)
		metadata_parts.append(task_metadata)
		episode_summaries.extend(task_summaries)
		print(
			f"collected task {task_idx + 1}/{len(task_names)} {task_name}: "
			f"steps={task_latents.shape[0]} goal_source={goal_info['source']}"
		)

	trajectory_latents = np.concatenate(trajectory_parts, axis=0)
	goal_latents = np.stack(goal_latents)
	goal_observations = np.stack(goal_observations)
	metadata = concatenate_metadata(metadata_parts)
	trajectory_embedding, goal_embedding, effective_perplexity = fit_tsne(
		trajectory_latents,
		goal_latents,
		args,
	)
	save_outputs(
		trajectory_latents,
		goal_latents,
		goal_observations,
		trajectory_embedding,
		goal_embedding,
		metadata,
		goal_sources,
		episode_summaries,
		task_names,
		checkpoint,
		effective_perplexity,
		args,
	)


if __name__ == "__main__":
	main()
'''
python Metaworld/tsne/tsne_goal_enc.py \
  --env metaworld_sequence_set12 \
  --method buffer \
  --seed 0 \
  --gpu 0 \
  --freeze_rand_vec 0 \
  --model_dir Metaworld/results/main3-2/set12_trj20/model \
  --num_eval_runs 10 \
  --rollout_goal_episodes 5 \
  --tsne_output_dir Metaworld/tsne/results/main3-2/set12_trj20

conta_abl
python Metaworld/tsne/tsne_goal_enc.py \
  --env metaworld_sequence_set12 \
  --method buffer \
  --seed 0 \
  --gpu 0 \
  --freeze_rand_vec 0 \
  --model_dir Metaworld/results/main3-2/set12_trj20_contra_abl/model \
  --num_eval_runs 10 \
  --rollout_goal_episodes 5 \
  --tsne_output_dir Metaworld/tsne/results/main3-2/set12_trj20_contra_abl


  Metaworld/results/main3/set12/model
  --rollout_goal_episodes for student sampling goals
'''