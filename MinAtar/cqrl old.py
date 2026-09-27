import argparse
import copy
import importlib
import math
import os
import pickle
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from CL_envs import CL_envs_func_replacement


def parse_args():
	parser = argparse.ArgumentParser(
		description="Continual quasimetric Q-learning for MinAtar"
	)
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--seq", type=int, default=0, help="Task sequence index")
	parser.add_argument("--t-steps", type=int, default=3500000)
	parser.add_argument("--switch", type=int, default=500000)
	parser.add_argument("--batch-size", type=int, default=64)
	parser.add_argument("--buffer-size", type=int, default=100000)
	parser.add_argument("--lr", type=float, default=1e-5, help="DQN learning rate")
	parser.add_argument("--gamma", type=float, default=0.99)
	parser.add_argument("--epsilon", type=float, default=0.1)
	parser.add_argument("--target-update", type=int, default=1000)
	parser.add_argument("--random-steps", type=int, default=10000)
	parser.add_argument("--distill-updates", type=int, default=1000)
	parser.add_argument("--distill-alpha", type=float, default=0.1)
	parser.add_argument("--distill-temperature", type=float, default=1.0)
	parser.add_argument("--qm-latent-dim", type=int, default=256)
	parser.add_argument("--qm-components", type=int, default=8)
	parser.add_argument("--qm-hidden-dim", type=int, default=256)
	parser.add_argument("--qm-lr", type=float, default=1e-4)
	parser.add_argument("--qm-discount", type=float, default=0.995)
	parser.add_argument("--qm-lambda", type=float, default=0.95)
	parser.add_argument("--qm-next-state-sample", type=float, default=0.2)
	parser.add_argument("--qm-backup-clip", type=float, default=5.0)
	parser.add_argument("--qm-action-invariance-coef", type=float, default=1.0)
	parser.add_argument("--qm-transition-consistency-coef", type=float, default=1.0)
	parser.add_argument("--qm-contrastive-coef", type=float, default=0.05)
	parser.add_argument(
		"--qm-nce-mode",
		choices=("forward_nce", "backward_nce"),
		default="forward_nce",
	)
	parser.add_argument("--qm-target-tau", type=float, default=0.01)
	parser.add_argument("--qm-current-batch-ratio", type=float, default=0.5)
	parser.add_argument("--qm-min-buffer-size", type=int, default=256)
	parser.add_argument("--structure-updates", type=int, default=1000)
	parser.add_argument("--memory-max-tasks", type=int, default=None)
	parser.add_argument("--memory-max-transitions-per-task", type=int, default=10000)
	parser.add_argument("--behavior-goal-success-only", type=int, choices=[0, 1], default=1)
	parser.add_argument("--save", action="store_true")
	parser.add_argument("--save-model", action="store_true")
	parser.add_argument("--output-dir", type=str, default="results")
	parser.add_argument("--gpu", type=int, default=0)
	parser.add_argument("--log-interval", type=int, default=1000)
	parser.add_argument("--wandb-project", type=str, default="minatar-cqrl")
	parser.add_argument("--wandb-entity", type=str, default=None)
	parser.add_argument("--wandb-group", type=str, default=None)
	parser.add_argument("--wandb-name", type=str, default=None)
	parser.add_argument(
		"--wandb-mode",
		type=str,
		default="online",
		choices=["online", "offline", "disabled"],
	)
	return parser.parse_args()


class WandbLogger:
	def __init__(self, args, run_name):
		self.run = None
		self.structure_step = 0
		self.distill_step = 0
		if args.wandb_mode == "disabled":
			return
		try:
			wandb = importlib.import_module("wandb")
		except ImportError:
			print("W&B is unavailable; install wandb or use --wandb-mode disabled.")
			return

		os.makedirs(args.output_dir, exist_ok=True)
		try:
			self.run = wandb.init(
				project=args.wandb_project,
				entity=args.wandb_entity,
				group=args.wandb_group,
				name=args.wandb_name or run_name,
				config=vars(args),
				dir=args.output_dir,
				mode=args.wandb_mode,
			)
			self.run.define_metric("global_step")
			self.run.define_metric("train/*", step_metric="global_step")
			self.run.define_metric("episode/*", step_metric="global_step")
			self.run.define_metric("task/*", step_metric="global_step")
			self.run.define_metric("structure_step")
			self.run.define_metric("structure/*", step_metric="structure_step")
			self.run.define_metric("distill_step")
			self.run.define_metric("distill/*", step_metric="distill_step")
			print(f"W&B run: {self.run.url or self.run.path}")
		except Exception as error:
			print(f"W&B initialization failed: {error}")
			self.run = None

	def log(self, metrics):
		if self.run is not None:
			self.run.log(metrics)

	def log_structure(self, metrics, task_id, update_index):
		self.structure_step += 1
		self.log(
			{
				"structure_step": self.structure_step,
				"structure/task_id": task_id,
				"structure/task_update": update_index,
				**{f"structure/{key}": value for key, value in metrics.items()},
			}
		)

	def log_distill(self, loss, task_id, update_index):
		self.distill_step += 1
		self.log(
			{
				"distill_step": self.distill_step,
				"distill/task_id": task_id,
				"distill/task_update": update_index,
				"distill/loss": loss,
			}
		)

	def finish(self, games):
		if self.run is not None:
			self.run.summary["task_sequence"] = games
			self.run.finish()
			self.run = None


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def observation_tensor(observation, device):
	observation = np.asarray(observation, dtype=np.float32)
	return torch.as_tensor(
		np.moveaxis(observation, -1, 0), device=device, dtype=torch.float32
	)


def update_average_return(average_return, episode_return):
	if average_return is None:
		return float(episode_return)
	return 0.99 * average_return + 0.01 * episode_return


@dataclass
class Transition:
	observation: torch.Tensor
	action: int
	next_observation: torch.Tensor
	reward: float
	done: float
	episode_id: int


class TrajectoryReplayBuffer:
	def __init__(self, capacity):
		self.transitions = deque(maxlen=capacity)
		self.episode_id = 0

	def __len__(self):
		return len(self.transitions)

	def add(self, observation, action, next_observation, reward, done):
		self.transitions.append(
			Transition(
				observation=torch.as_tensor(
					np.moveaxis(np.asarray(observation, dtype=np.float32), -1, 0)
				).clone(),
				action=int(action),
				next_observation=torch.as_tensor(
					np.moveaxis(np.asarray(next_observation, dtype=np.float32), -1, 0)
				).clone(),
				reward=float(reward),
				done=float(done),
				episode_id=self.episode_id,
			)
		)
		if done:
			self.episode_id += 1

	def sample_transitions(self, batch_size, device):
		if not self.transitions:
			raise ValueError("Cannot sample an empty replay buffer.")
		batch = random.sample(list(self.transitions), min(batch_size, len(self)))
		return {
			"observations": torch.stack([item.observation for item in batch]).to(device),
			"actions": torch.tensor(
				[item.action for item in batch], device=device, dtype=torch.long
			),
			"next_observations": torch.stack(
				[item.next_observation for item in batch]
			).to(device),
			"rewards": torch.tensor(
				[item.reward for item in batch], device=device, dtype=torch.float32
			),
			"dones": torch.tensor(
				[item.done for item in batch], device=device, dtype=torch.float32
			),
		}

	def sample_quasimetric_batch(
		self,
		batch_size,
		device,
		discount,
		intermediate_lambda,
		next_state_sample,
	):
		if not self.transitions:
			raise ValueError("Cannot sample an empty replay buffer.")

		transitions = list(self.transitions)
		indices_by_episode = {}
		for index, transition in enumerate(transitions):
			indices_by_episode.setdefault(transition.episode_id, []).append(index)

		sample_size = min(batch_size, len(transitions))
		indices = random.sample(range(len(transitions)), sample_size)
		goals = []
		intermediate_goals = []
		offsets = []
		selected = []
		for index in indices:
			transition = transitions[index]
			episode_indices = indices_by_episode[transition.episode_id]
			position = episode_indices.index(index)
			future_indices = episode_indices[position:]

			goal_offset = self._discounted_offset(len(future_indices), discount)
			goal_index = future_indices[goal_offset]
			if random.random() < next_state_sample or goal_offset == 0:
				intermediate_offset = 0
			else:
				intermediate_offset = self._discounted_offset(
					goal_offset + 1, intermediate_lambda
				)
			intermediate_index = future_indices[intermediate_offset]

			selected.append(transition)
			goals.append(transitions[goal_index].next_observation)
			intermediate_goals.append(
				transitions[intermediate_index].next_observation
			)
			offsets.append(float(intermediate_offset))

		return {
			"observations": torch.stack([item.observation for item in selected]).to(device),
			"actions": torch.tensor(
				[item.action for item in selected], device=device, dtype=torch.long
			),
			"next_observations": torch.stack(
				[item.next_observation for item in selected]
			).to(device),
			"dones": torch.tensor(
				[item.done for item in selected], device=device, dtype=torch.float32
			),
			"goals": torch.stack(goals).to(device),
			"intermediate_goals": torch.stack(intermediate_goals).to(device),
			"offsets": torch.tensor(offsets, device=device, dtype=torch.float32),
		}

	def sample_behavior_goal(self, success_only=True):
		if not self.transitions:
			return None
		candidates = [
			item.next_observation for item in self.transitions if item.reward > 0
		]
		if not candidates and not success_only:
			candidates = [item.next_observation for item in self.transitions]
		if not candidates:
			candidates = [self.transitions[-1].next_observation]
		return random.choice(candidates).clone()

	def snapshot(self, max_transitions=None):
		capacity = len(self) if max_transitions is None else min(len(self), max_transitions)
		result = TrajectoryReplayBuffer(max(1, capacity))
		items = list(self.transitions)
		if max_transitions is not None:
			items = items[-max_transitions:]
		result.transitions.extend(copy.deepcopy(items))
		result.episode_id = self.episode_id
		return result

	@staticmethod
	def _discounted_offset(length, discount):
		if length <= 1:
			return 0
		weights = np.power(discount, np.arange(length, dtype=np.float64))
		weights /= weights.sum()
		return int(np.random.choice(length, p=weights))


class TaskAwareReplayMemory:
	def __init__(self, max_tasks=None, max_transitions_per_task=None):
		self.max_tasks = max_tasks
		self.max_transitions_per_task = max_transitions_per_task
		self.task_buffers = deque(maxlen=max_tasks)

	def __len__(self):
		return sum(len(buffer) for _, buffer in self.task_buffers)

	@property
	def num_tasks(self):
		return len(self.task_buffers)

	def add_task(self, task_id, replay_buffer):
		self.task_buffers.append(
			(
				task_id,
				replay_buffer.snapshot(self.max_transitions_per_task),
			)
		)

	def sample_quasimetric_batch(self, batch_size, device, **sample_kwargs):
		if not self.task_buffers:
			raise ValueError("Cannot sample empty task memory.")
		per_task = np.random.multinomial(
			batch_size, np.full(self.num_tasks, 1.0 / self.num_tasks)
		)
		batches = []
		for count, (_, buffer) in zip(per_task, self.task_buffers):
			if count > 0:
				batches.append(
					buffer.sample_quasimetric_batch(count, device, **sample_kwargs)
				)
		return {
			key: torch.cat([batch[key] for batch in batches], dim=0)
			for key in batches[0]
		}

	def sample_observations(self, batch_size, device):
		if not self.task_buffers:
			raise ValueError("Cannot sample empty task memory.")
		observations = []
		for _ in range(batch_size):
			_, buffer = random.choice(list(self.task_buffers))
			transition = random.choice(list(buffer.transitions))
			observations.append(transition.observation)
		return torch.stack(observations).to(device)


class StateEncoderCNN(nn.Module):
	def __init__(self, in_channels, latent_dim):
		super().__init__()
		self.conv = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1)
		self.hidden = nn.Linear(32 * 8 * 8, 256)
		self.output = nn.Linear(256, latent_dim)
		self.apply(self._initialize)

	@staticmethod
	def _initialize(module):
		if isinstance(module, (nn.Conv2d, nn.Linear)):
			nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
			if module.bias is not None:
				nn.init.zeros_(module.bias)

	def forward(self, observations):
		features = F.relu(self.conv(observations))
		features = F.relu(self.hidden(features.flatten(start_dim=1)))
		return self.output(features)


class QNetwork(nn.Module):
	def __init__(self, in_channels, num_actions):
		super().__init__()
		self.encoder = StateEncoderCNN(in_channels, 256)
		self.output = nn.Linear(256, num_actions)
		nn.init.kaiming_normal_(self.output.weight, nonlinearity="linear")
		nn.init.zeros_(self.output.bias)

	def forward(self, observations):
		return self.output(F.relu(self.encoder(observations)))


def mrn_distance(first, second, components):
	first, second = torch.broadcast_tensors(first, second)
	total_dim = first.shape[-1]
	if total_dim % components != 0:
		raise ValueError(
			f"Embedding dimension {total_dim} must be divisible by {components}."
		)
	component_dim = total_dim // components
	difference = (first - second).reshape(
		*first.shape[:-1], components, component_dim
	)
	asymmetric_dim = component_dim // 2
	if asymmetric_dim:
		max_component = F.relu(
			difference[..., :asymmetric_dim].amax(dim=-1)
		)
	else:
		max_component = torch.zeros_like(difference[..., 0])
	l2_component = torch.linalg.vector_norm(
		difference[..., asymmetric_dim:], dim=-1
	)
	return (max_component + l2_component).mean(dim=-1) / math.sqrt(total_dim)


class DiscreteQuasimetricLearner(nn.Module):
	def __init__(self, in_channels, num_actions, args, device):
		super().__init__()
		if args.qm_latent_dim % args.qm_components != 0:
			raise ValueError("qm-latent-dim must be divisible by qm-components.")
		self.num_actions = num_actions
		self.latent_dim = args.qm_latent_dim
		self.components = args.qm_components
		self.discount = args.qm_discount
		self.intermediate_lambda = args.qm_lambda
		self.next_state_sample = args.qm_next_state_sample
		self.backup_clip = args.qm_backup_clip
		self.action_invariance_coef = args.qm_action_invariance_coef
		self.transition_consistency_coef = args.qm_transition_consistency_coef
		self.contrastive_coef = args.qm_contrastive_coef
		self.nce_mode = args.qm_nce_mode
		if self.nce_mode not in ("forward_nce", "backward_nce"):
			raise ValueError(f"Unsupported NCE mode: {self.nce_mode}")
		self.target_tau = args.qm_target_tau
		self.current_batch_ratio = args.qm_current_batch_ratio

		self.state_encoder = StateEncoderCNN(in_channels, self.latent_dim)
		self.transition_encoder = nn.Sequential(
			nn.Linear(self.latent_dim + num_actions, args.qm_hidden_dim),
			nn.ReLU(),
			nn.Linear(args.qm_hidden_dim, args.qm_hidden_dim),
			nn.ReLU(),
			nn.Linear(args.qm_hidden_dim, self.latent_dim),
		)
		self.target_state_encoder = copy.deepcopy(self.state_encoder)
		for parameter in self.target_state_encoder.parameters():
			parameter.requires_grad_(False)
		self.to(device)
		self.optimizer = torch.optim.Adam(
			list(self.state_encoder.parameters())
			+ list(self.transition_encoder.parameters()),
			lr=args.qm_lr,
		)

	def transition_representation(self, state_representation, actions):
		action_one_hot = F.one_hot(actions, self.num_actions).float()
		return self.transition_encoder(
			torch.cat([state_representation, action_one_hot], dim=-1)
		)

	def action_distances(self, observations, goals):
		state_representation = self.state_encoder(observations)
		goal_representation = self.state_encoder(goals)
		batch_size = observations.shape[0]
		actions = torch.arange(
			self.num_actions, device=observations.device
		).repeat(batch_size)
		repeated_states = state_representation.repeat_interleave(
			self.num_actions, dim=0
		)
		transitions = self.transition_representation(repeated_states, actions)
		repeated_goals = goal_representation.repeat_interleave(
			self.num_actions, dim=0
		)
		return mrn_distance(
			transitions, repeated_goals, self.components
		).reshape(batch_size, self.num_actions)

	def sample_training_batch(self, current_buffer, memory, batch_size, device):
		sample_kwargs = {
			"discount": self.discount,
			"intermediate_lambda": self.intermediate_lambda,
			"next_state_sample": self.next_state_sample,
		}
		if memory.num_tasks == 0:
			return current_buffer.sample_quasimetric_batch(
				batch_size, device, **sample_kwargs
			)

		current_size = max(1, int(batch_size * self.current_batch_ratio))
		memory_size = max(0, batch_size - current_size)
		batches = [
			current_buffer.sample_quasimetric_batch(
				current_size, device, **sample_kwargs
			)
		]
		if memory_size:
			batches.append(
				memory.sample_quasimetric_batch(
					memory_size, device, **sample_kwargs
				)
			)
		return {
			key: torch.cat([batch[key] for batch in batches], dim=0)
			for key in batches[0]
		}

	def update(self, batch):
		state_representation = self.state_encoder(batch["observations"])
		goal_representation = self.state_encoder(batch["goals"])
		transition_representation = self.transition_representation(
			state_representation, batch["actions"]
		)
		with torch.no_grad():
			next_representation = self.target_state_encoder(
				batch["next_observations"]
			)
			intermediate_representation = self.target_state_encoder(
				batch["intermediate_goals"]
			)
			target_goal_representation = self.target_state_encoder(batch["goals"])

		distance = mrn_distance(
			transition_representation, goal_representation, self.components
		)
		next_distance = mrn_distance(
			intermediate_representation,
			target_goal_representation,
			self.components,
		)
		bootstrap = torch.pow(
			torch.full_like(batch["offsets"], self.discount), batch["offsets"]
		) * (1.0 - batch["dones"])
		delta = distance - next_distance
		delta_clipped = torch.minimum(
			delta, torch.full_like(delta, self.backup_clip)
		)
		backup = torch.where(
			delta > self.backup_clip,
			delta,
			bootstrap * torch.exp(delta_clipped) - distance,
		)
		backup_loss = backup.mean()
		transition_consistency_distance = mrn_distance(
			transition_representation,
			next_representation,
			self.components,
		)
		transition_consistency_loss = transition_consistency_distance.mean()

		action_distance = mrn_distance(
			transition_representation, state_representation, self.components
		)
		action_invariance_loss = (
			(torch.exp(-action_distance) - 1.0).square().mean()
		)
		pairwise_distance = mrn_distance(
			transition_representation[:, None, :],
			goal_representation[None, :, :],
			self.components,
		)
		logits = -pairwise_distance
		if self.nce_mode == "backward_nce":
			logits = logits.transpose(0, 1)
		labels = torch.arange(logits.shape[0], device=distance.device)
		contrastive_loss = F.cross_entropy(logits, labels)
		loss = (
			backup_loss
			+ self.action_invariance_coef * action_invariance_loss
			+ self.transition_consistency_coef * transition_consistency_loss
			+ self.contrastive_coef * contrastive_loss
		)

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		with torch.no_grad():
			for source, target in zip(
				self.state_encoder.parameters(),
				self.target_state_encoder.parameters(),
			):
				target.data.lerp_(source.data, self.target_tau)
		return {
			"loss": float(loss.item()),
			"backup_loss": float(backup_loss.item()),
			"action_invariance_loss": float(action_invariance_loss.item()),
			"transition_consistency_loss": float(
				transition_consistency_loss.item()
			),
			"contrastive_loss": float(contrastive_loss.item()),
			"distance": float(distance.mean().item()),
			"structure_alignment": float(
				torch.exp(-transition_consistency_distance).mean().item()
			),
		}


class ContinualQuasimetricDQN:
	def __init__(self, in_channels, num_actions, args, device):
		self.device = device
		self.num_actions = num_actions
		self.batch_size = args.batch_size
		self.gamma = args.gamma
		self.distill_alpha = args.distill_alpha
		self.distill_temperature = args.distill_temperature
		self.quasimetric = DiscreteQuasimetricLearner(
			in_channels, num_actions, args, device
		)
		self.memory = TaskAwareReplayMemory(
			max_tasks=args.memory_max_tasks,
			max_transitions_per_task=args.memory_max_transitions_per_task,
		)
		self.behavior_goal = None
		self.behavior_goals = {}
		self.in_channels = in_channels
		self.lr = args.lr
		self.reset_student()

	def reset_student(self):
		self.q_network = QNetwork(self.in_channels, self.num_actions).to(self.device)
		self.target_network = copy.deepcopy(self.q_network).to(self.device)
		self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)

	def act(self, observation, epsilon, action_space):
		if random.random() < epsilon:
			return action_space.sample()
		with torch.no_grad():
			q_values = self.q_network(
				observation_tensor(observation, self.device).unsqueeze(0)
			)
		return int(q_values.argmax(dim=1).item())

	def update_q(self, replay_buffer, use_distillation):
		batch = replay_buffer.sample_transitions(self.batch_size, self.device)
		with torch.no_grad():
			next_q = self.target_network(batch["next_observations"]).max(dim=1).values
			target = batch["rewards"] + (
				1.0 - batch["dones"]
			) * self.gamma * next_q
		prediction = self.q_network(batch["observations"]).gather(
			1, batch["actions"].unsqueeze(1)
		).squeeze(1)
		q_loss = F.mse_loss(prediction, target)
		loss = q_loss
		distill_loss = None

		if use_distillation and self.behavior_goal is not None:
			teacher_logits = self.teacher_logits(batch["observations"])
			student_log_probs = F.log_softmax(
				self.q_network(batch["observations"])
				/ self.distill_temperature,
				dim=-1,
			)
			teacher_probs = F.softmax(
				teacher_logits / self.distill_temperature, dim=-1
			)
			distill_loss = F.kl_div(
				student_log_probs, teacher_probs, reduction="batchmean"
			) * self.distill_temperature**2
			loss = loss + self.distill_alpha * distill_loss

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		return {
			"loss": float(loss.item()),
			"q_loss": float(q_loss.item()),
			"distill_loss": (
				float(distill_loss.item()) if distill_loss is not None else 0.0
			),
			"q_value": float(prediction.mean().item()),
			"target_q_value": float(target.mean().item()),
		}

	@torch.no_grad()
	def teacher_logits(self, observations):
		goal = self.behavior_goal.to(self.device).unsqueeze(0)
		goals = goal.expand(observations.shape[0], -1, -1, -1)
		return -self.quasimetric.action_distances(observations, goals)

	def distill(self, replay_buffer, updates, on_update=None):
		if self.behavior_goal is None or not len(replay_buffer):
			return None
		last_loss = None
		for update_index in range(updates):
			batch = replay_buffer.sample_transitions(self.batch_size, self.device)
			with torch.no_grad():
				teacher_probs = F.softmax(
					self.teacher_logits(batch["observations"])
					/ self.distill_temperature,
					dim=-1,
				)
			student_log_probs = F.log_softmax(
				self.q_network(batch["observations"])
				/ self.distill_temperature,
				dim=-1,
			)
			loss = F.kl_div(
				student_log_probs, teacher_probs, reduction="batchmean"
			) * self.distill_temperature**2
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
			last_loss = float(loss.item())
			if on_update is not None:
				on_update(update_index + 1, last_loss)
		return last_loss

	def finish_task(self, task_id, replay_buffer, args, on_structure_update=None):
		if len(replay_buffer) < args.qm_min_buffer_size:
			return {}
		last_metrics = {}
		for update_index in range(args.structure_updates):
			batch = self.quasimetric.sample_training_batch(
				replay_buffer, self.memory, args.batch_size, self.device
			)
			last_metrics = self.quasimetric.update(batch)
			if on_structure_update is not None:
				on_structure_update(update_index + 1, last_metrics)
		self.behavior_goal = replay_buffer.sample_behavior_goal(
			success_only=bool(args.behavior_goal_success_only)
		)
		self.behavior_goals[task_id] = self.behavior_goal.clone()
		self.memory.add_task(task_id, replay_buffer)
		last_metrics.update(
			{
				"memory_tasks": self.memory.num_tasks,
				"memory_transitions": len(self.memory),
			}
		)
		return last_metrics

	def update_target(self):
		self.target_network.load_state_dict(self.q_network.state_dict())

	def checkpoint(self):
		return {
			"q_network": self.q_network.state_dict(),
			"target_network": self.target_network.state_dict(),
			"quasimetric": self.quasimetric.state_dict(),
			"behavior_goal": self.behavior_goal,
			"behavior_goals": self.behavior_goals,
		}


def collect_random_warmup(env, replay_buffer, steps):
	observation = env.reset()
	episode_return = 0.0
	completed_returns = []
	for _ in range(steps):
		action = env.action_space.sample()
		next_observation, reward, done, _ = env.step(action)
		replay_buffer.add(observation, action, next_observation, reward, done)
		episode_return += reward
		observation = next_observation
		if done:
			completed_returns.append(episode_return)
			episode_return = 0.0
			observation = env.reset()
	return observation, completed_returns, episode_return


def main():
	args = parse_args()
	if args.switch <= 0 or args.t_steps <= 0:
		raise ValueError("t-steps and switch must be positive.")
	if args.random_steps < 0:
		raise ValueError("random-steps cannot be negative.")
	if args.log_interval <= 0:
		raise ValueError("log-interval must be positive.")
	num_tasks = math.ceil(args.t_steps / args.switch)
	if num_tasks > 7:
		raise ValueError("MinAtar task sequences contain at most seven tasks.")

	set_seed(args.seed)
	if torch.cuda.is_available():
		device = torch.device(f"cuda:{args.gpu}")
		torch.cuda.set_device(device)
	else:
		device = torch.device("cpu")

	run_name = (
		f"CQRL_steps_{args.t_steps}_switch_{args.switch}_seq_{args.seq}"
		f"_seed_{args.seed}"
	)
	logger = WandbLogger(args, run_name)
	env = CL_envs_func_replacement(args.seq, 0, args.seed)
	in_channels = env.observation_space.shape[2]
	num_actions = env.action_space.n
	agent = ContinualQuasimetricDQN(
		in_channels, num_actions, args, device
	)
	replay_buffer = TrajectoryReplayBuffer(args.buffer_size)
	returns = np.zeros(args.t_steps, dtype=np.float32)
	average_return = None
	observation, warmup_returns, episode_return = collect_random_warmup(
		env, replay_buffer, args.random_steps
	)
	for value in warmup_returns:
		average_return = update_average_return(average_return, value)
	print(f"Started task 0 ({env.game_name}); warmup={args.random_steps}")
	task_id = 0
	games = [env.game_name]
	last_metrics = {}
	last_q_metrics = {}
	logger.log(
		{
			"global_step": 0,
			"task/id": task_id,
			"task/game": env.game_name,
			"task/start": 1,
			"task/warmup_steps": args.random_steps,
			"task/warmup_episodes": len(warmup_returns),
			"task/warmup_return_mean": (
				float(np.mean(warmup_returns)) if warmup_returns else 0.0
			),
			"train/replay_size": len(replay_buffer),
		}
	)

	progress = tqdm(range(args.t_steps))
	for step in progress:
		if step > 0 and step % args.switch == 0:
			last_metrics = agent.finish_task(
				task_id,
				replay_buffer,
				args,
				on_structure_update=lambda update_index, metrics: logger.log_structure(
					metrics, task_id, update_index
				),
			)
			print(f"Finished task {task_id} ({env.game_name}): {last_metrics}")
			logger.log(
				{
					"global_step": step,
					"task/id": task_id,
					"task/end": 1,
					**{f"task/final_{key}": value for key, value in last_metrics.items()},
				}
			)

			task_id += 1
			env = CL_envs_func_replacement(args.seq, task_id, args.seed)
			games.append(env.game_name)
			if (
				env.observation_space.shape[2] != in_channels
				or env.action_space.n != num_actions
			):
				raise ValueError("All tasks must share observation and action dimensions.")

			agent.reset_student()   # new individual Q
			last_q_metrics = {}
			replay_buffer = TrajectoryReplayBuffer(args.buffer_size)
			warmup_steps = args.random_steps
			average_return = None
			observation, warmup_returns, episode_return = collect_random_warmup(
				env, replay_buffer, warmup_steps
			)
			for value in warmup_returns:
				average_return = update_average_return(average_return, value)
			distill_loss = agent.distill(
				replay_buffer,
				args.distill_updates,
				on_update=lambda update_index, loss: logger.log_distill(
					loss, task_id, update_index
				),
			)
			print(
				f"Started task {task_id} ({env.game_name}); "
				f"warmup={warmup_steps}, distill_loss={distill_loss}"
			)
			logger.log(
				{
					"global_step": step,
					"task/id": task_id,
					"task/game": env.game_name,
					"task/start": 1,
					"task/warmup_steps": warmup_steps,
					"task/warmup_episodes": len(warmup_returns),
					"task/warmup_return_mean": (
						float(np.mean(warmup_returns)) if warmup_returns else 0.0
					),
					"task/initial_distill_loss": distill_loss or 0.0,
					"train/replay_size": len(replay_buffer),
				}
			)

		action = agent.act(observation, args.epsilon, env.action_space)
		next_observation, reward, done, _ = env.step(action)
		replay_buffer.add(observation, action, next_observation, reward, done)
		episode_return += reward

		if len(replay_buffer) >= args.batch_size:
			last_q_metrics = agent.update_q(
				replay_buffer,
				use_distillation=(
					agent.behavior_goal is not None
					and step % args.switch < args.random_steps
				),
			)
		if step > 0 and step % args.target_update == 0:
			agent.update_target()

		observation = next_observation
		if done:
			completed_return = episode_return
			average_return = update_average_return(
				average_return, completed_return
			)
			episode_return = 0.0
			observation = env.reset()
			logger.log(
				{
					"global_step": step + 1,
					"episode/return": completed_return,
					"episode/average_return": average_return,
					"episode/task_id": task_id,
				}
			)
		logged_average_return = average_return if average_return is not None else 0.0
		returns[step] = logged_average_return
		if step % args.log_interval == 0:
			progress.set_postfix(
				task=task_id,
				game=env.game_name,
				avg_return=f"{logged_average_return:.3f}",
			)
			logger.log(
				{
					"global_step": step + 1,
					"train/task_id": task_id,
					"train/task_step": step % args.switch,
					"train/reward": reward,
					"train/average_return": logged_average_return,
					"train/replay_size": len(replay_buffer),
					"train/epsilon": args.epsilon,
					**{f"train/{key}": value for key, value in last_q_metrics.items()},
				}
			)

	last_metrics = agent.finish_task(
		task_id,
		replay_buffer,
		args,
		on_structure_update=lambda update_index, metrics: logger.log_structure(
			metrics, task_id, update_index
		),
	)
	print(f"Finished task {task_id} ({env.game_name}): {last_metrics}")
	print("Games:", games)
	logger.log(
		{
			"global_step": args.t_steps,
			"task/id": task_id,
			"task/end": 1,
			**{f"task/final_{key}": value for key, value in last_metrics.items()},
		}
	)
	if args.save or args.save_model:
		os.makedirs(args.output_dir, exist_ok=True)
	if args.save:
		with open(
			os.path.join(args.output_dir, run_name + "_returns.pkl"), "wb"
		) as output_file:
			pickle.dump(returns, output_file)
	if args.save_model:
		torch.save(
			agent.checkpoint(),
			os.path.join(args.output_dir, run_name + "_checkpoint.pt"),
		)
	logger.finish(games)


if __name__ == "__main__":
	main()

'''
python cqrl.py \
  --seed 0 \
  --seq 0 \
  --gpu 0 \
  --wandb-project minatar-cqrl \
  --wandb-name cqrl-seq0-seed0forward2 \
  --save \
  --save-model \
  --qm-nce-mode forward_nce

'''


