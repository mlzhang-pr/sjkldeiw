import argparse
import csv
import importlib
import os
import random
import time

import numpy as np
import torch

import fetch_env
from agent.td3_her import TD3HERAgent
from replay_buffer_td3_her import HerReplayBuffer


def str2none(value):
    if value is None:
        return None
    if str(value).lower() in {"none", ""}:
        return None
    return value


def str_choice(value):
    return str(value).strip().strip("'\"").lower()


def set_seed_everywhere(seed_value):
    seed_value = int(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)


def str2log_backend(value):
    value = str(value).strip().lower()
    if value not in {"none", "wandb"}:
        raise argparse.ArgumentTypeError(f"Unsupported log backend: {value}")
    return value


def normalize_log_backends(backends):
    normalized = []
    for backend in backends:
        if backend == "none":
            if len(backends) != 1:
                raise ValueError("'none' cannot be combined with other logging backends")
            return []
        if backend not in normalized:
            normalized.append(backend)
    return normalized


def load_wandb_module():
    try:
        return importlib.import_module("wandb")
    except ImportError:
        return None


def create_wandb_run(args, log_name):
    if "wandb" not in args.log_backends:
        return None

    wandb_module = load_wandb_module()
    if wandb_module is None:
        print("W&B is unavailable. Install 'wandb' to enable W&B monitoring.")
        return None

    os.makedirs(args.save_path, exist_ok=True)
    try:
        return wandb_module.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=log_name,
            config=vars(args),
            dir=args.save_path,
            mode=args.wandb_mode,
            save_code=True,
        )
    except Exception as error:
        print(f"W&B initialization failed: {error}")
        return None


def wandb_log(wandb_run, values, step):
    if wandb_run is None:
        return
    clean_values = {}
    for key, value in values.items():
        if value is None or isinstance(value, str):
            continue
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (int, float)) and np.isfinite(value):
            clean_values[key] = value
    if clean_values:
        wandb_run.log(clean_values, step=step)


def define_wandb_metrics(wandb_run):
    if wandb_run is None:
        return
    define_metric = getattr(wandb_run, "define_metric", None)
    if define_metric is None:
        return
    define_metric("global_step")
    define_metric("train/*", step_metric="global_step")
    define_metric("eval/*", step_metric="global_step")
    define_metric("rollout/*", step_metric="global_step")
    define_metric("time/*", step_metric="global_step")


def parse_args():
    parser = argparse.ArgumentParser(description="Train TD3 + HER on Fetch goal environments")
    parser.add_argument("--env", type=str, default="fetch_reach", help="Fetch task or sequence name")
    parser.add_argument("--task_order", type=str2none, default=None, help="Custom sequence, e.g. reach,push,slide")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--gpu", type=str, default="0", help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--total_timesteps", type=int, default=200000, help="Total environment steps")
    parser.add_argument("--change_freq", type=int, default=200000, help="Steps before FetchGoalEnvSequence changes task")
    parser.add_argument("--buffer_size", type=int, default=1000000, help="Replay buffer capacity")
    parser.add_argument("--learning_starts", type=int, default=10000, help="Random steps before TD3 updates")
    parser.add_argument("--train_freq", type=int, default=1, help="Environment steps between training calls")
    parser.add_argument("--gradient_steps", type=int, default=1, help="Gradient steps per training call; -1 means train_freq")
    parser.add_argument("--batch_size", type=int, default=256, help="Training batch size")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="Target network Polyak rate")
    parser.add_argument("--actor_lr", type=float, default=1e-3, help="Actor learning rate")
    parser.add_argument("--critic_lr", type=float, default=1e-3, help="Critic learning rate")
    parser.add_argument("--policy_delay", type=int, default=2, help="Delayed actor update frequency")
    parser.add_argument("--target_policy_noise", type=float, default=0.2, help="Target action smoothing noise")
    parser.add_argument("--target_noise_clip", type=float, default=0.5, help="Target smoothing noise clip")
    parser.add_argument("--action_noise", type=float, default=0.1, help="Gaussian exploration noise std")
    parser.add_argument("--n_sampled_goal", type=int, default=4, help="HER future goals per real transition ratio parameter")
    parser.add_argument("--hidden_dim", type=int, default=256, help="MLP hidden dimension")
    parser.add_argument("--hidden_depth", type=int, default=2, help="MLP hidden depth")
    parser.add_argument("--eval_freq", type=int, default=5000, help="Evaluation frequency")
    parser.add_argument("--num_eval_episodes", type=int, default=10, help="Episodes per evaluation")
    parser.add_argument("--save_path", type=str, default="results/td3_her", help="Directory for logs and models")
    parser.add_argument("--save_model_freq", type=int, default=-1, help="Model checkpoint frequency; -1 disables")
    parser.add_argument("--reset_buffer_on_task_change", type=int, default=1, help="Reset HER buffer after sequence task change")
    parser.add_argument(
        "--log_backends",
        type=str2log_backend,
        nargs="+",
        default=["wandb"],
        choices=["none", "wandb"],
        help="Logging backends to enable. Use 'wandb' for W&B monitoring or 'none' for local CSV only.",
    )
    parser.add_argument("--wandb_project_name", type=str, default="continual-quasimetric-rl-fetch")
    parser.add_argument("--wandb_entity", type=str2none, default=None)
    parser.add_argument("--wandb_group", type=str2none, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--gc_reward_type", type=str_choice, default="sparse", choices=["sparse", "dense", "success"])
    parser.add_argument("--gc_success_threshold", type=float, default=0.05)
    parser.add_argument("--fetch_goal_format", type=str_choice, default="native", choices=["online", "native"])
    parser.add_argument("--fetch_env_version", type=str2none, default="auto")
    parser.add_argument("--max_episode_steps", type=int, default=50)
    parser.add_argument("--normalize_obs", type=str2none, default=None)
    args = parser.parse_args()
    try:
        args.log_backends = normalize_log_backends(args.log_backends)
    except ValueError as error:
        parser.error(str(error))
    return args


def fetch_env_kwargs(args):
    env_name = args.env.lower()
    kwargs = {
        "change_freq": args.change_freq,
        "goal_conditioned": True,
        "gc_reward_type": args.gc_reward_type,
        "gc_success_threshold": args.gc_success_threshold,
        "fetch_goal_format": args.fetch_goal_format,
        "fetch_env_version": args.fetch_env_version,
        "max_episode_steps": args.max_episode_steps,
        "normalize_obs": args.normalize_obs,
        "task_order": args.task_order,
        "seed": args.seed,
    }

    if env_name.startswith("fetch_sequence_"):
        suffix = env_name[len("fetch_sequence_") :]
        if suffix and suffix != "custom":
            if suffix.startswith("set") or suffix in fetch_env.FETCH_SEQUENCE_PRESETS:
                kwargs["env_sequence"] = suffix
            else:
                kwargs["base_task_name"] = suffix
    elif env_name.startswith("fetch_"):
        kwargs["base_task_name"] = env_name[len("fetch_") :]
    else:
        kwargs["base_task_name"] = env_name
    return kwargs


def make_replay_buffer(env, args, device):
    obs_space = env.env.observation_space
    if not hasattr(obs_space, "spaces"):
        raise TypeError("TD3 + HER requires a dict Fetch observation space.")
    spaces = obs_space.spaces
    return HerReplayBuffer(
        observation_shape=spaces["observation"].shape,
        achieved_goal_shape=spaces["achieved_goal"].shape,
        desired_goal_shape=spaces["desired_goal"].shape,
        action_shape=env.env.action_space.shape,
        capacity=args.buffer_size,
        device=device,
        env=env,
        n_sampled_goal=args.n_sampled_goal,
    )


def build_agent(env, args, device):
    obs_space = env.env.observation_space.spaces
    return TD3HERAgent(
        observation_dim=obs_space["observation"].shape[0],
        goal_dim=obs_space["desired_goal"].shape[0],
        action_space=env.env.action_space,
        device=device,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        hidden_dim=args.hidden_dim,
        hidden_depth=args.hidden_depth,
        gamma=args.gamma,
        tau=args.tau,
        policy_delay=args.policy_delay,
        target_policy_noise=args.target_policy_noise,
        target_noise_clip=args.target_noise_clip,
        batch_size=args.batch_size,
    )


def summarize_eval(eval_results):
    returns = np.asarray(eval_results.get("episodic_returns", []), dtype=np.float32)
    successes = np.asarray(eval_results.get("successes", []), dtype=np.float32)
    goal_successes = np.asarray(eval_results.get("goal_successes", []), dtype=np.float32)
    distances = np.asarray(eval_results.get("final_goal_distances", []), dtype=np.float32)
    return {
        "return_mean": float(np.mean(returns)) if returns.size else np.nan,
        "success_mean": float(np.mean(successes)) if successes.size else np.nan,
        "goal_success_mean": float(np.mean(goal_successes)) if goal_successes.size else np.nan,
        "final_goal_distance": float(np.mean(distances)) if distances.size else np.nan,
    }


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_seed_everywhere(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = fetch_env.FetchGoalEnvSequence(**fetch_env_kwargs(args))
    replay_buffer = make_replay_buffer(env, args, device)
    agent = build_agent(env, args, device)

    os.makedirs(args.save_path, exist_ok=True)
    log_name = f"td3_her_{args.env}_{args.seed}"
    wandb_run = create_wandb_run(args, log_name)
    define_wandb_metrics(wandb_run)
    metrics_rows = []
    obs, _ = env.reset(seed=args.seed)
    current_task_counter = env.task_counter
    episode_return = 0.0
    episode_length = 0
    start_time = time.perf_counter()
    last_train_metrics = {}

    for step in range(args.total_timesteps):
        if step < args.learning_starts:
            action = env.env.action_space.sample()
        else:
            action = agent.act(obs["observation"], goal_obs=obs["desired_goal"], noise_std=args.action_noise)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        timeout = bool(truncated and not terminated)
        replay_buffer.add(obs, action, reward, next_obs, done, info, timeout=timeout)

        episode_return += float(reward)
        episode_length += 1
        obs = next_obs

        if step >= args.learning_starts and (step + 1) % args.train_freq == 0:
            gradient_steps = args.train_freq if args.gradient_steps < 0 else args.gradient_steps
            last_train_metrics = agent.update(replay_buffer, gradient_steps=gradient_steps)
            train_log = {f"train/{key}": value for key, value in last_train_metrics.items()}
            train_log["global_step"] = step + 1
            wandb_log(
                wandb_run,
                train_log,
                step + 1,
            )

        if done:
            success = bool(info.get("success", info.get("is_success", False)))
            print(
                f"episode step={step + 1} task={env.base_task_name} "
                f"return={episode_return:.2f} length={episode_length} success={int(success)}"
            )
            wandb_log(
                wandb_run,
                {
                    "global_step": step + 1,
                    "rollout/episode_return": episode_return,
                    "rollout/episode_length": episode_length,
                    "rollout/success": int(success),
                    "rollout/task_counter": env.task_counter,
                },
                step + 1,
            )
            obs, _ = env.reset()
            episode_return = 0.0
            episode_length = 0

            if current_task_counter != env.task_counter:
                current_task_counter = env.task_counter
                if args.reset_buffer_on_task_change:
                    replay_buffer.reset()
                print(f"switched task to {env.base_task_name}; replay_size={len(replay_buffer)}")

        if args.eval_freq > 0 and ((step + 1) % args.eval_freq == 0 or step + 1 == args.total_timesteps):
            eval_results = env.evaluate_agent(agent, args.num_eval_episodes)
            eval_metrics = summarize_eval(eval_results)
            elapsed = time.perf_counter() - start_time
            sps = int((step + 1) / elapsed) if elapsed > 0 else 0
            row = {
                "step": step + 1,
                "task": env.base_task_name,
                "task_counter": env.task_counter,
                "replay_size": len(replay_buffer),
                "sps": sps,
                **eval_metrics,
                **last_train_metrics,
            }
            metrics_rows.append(row)
            wandb_log(
                wandb_run,
                {
                    "global_step": step + 1,
                    "eval/return_mean": eval_metrics["return_mean"],
                    "eval/success_mean": eval_metrics["success_mean"],
                    "eval/goal_success_mean": eval_metrics["goal_success_mean"],
                    "eval/final_goal_distance": eval_metrics["final_goal_distance"],
                    "eval/replay_size": len(replay_buffer),
                    "eval/task_counter": env.task_counter,
                    "time/sps": sps,
                },
                step + 1,
            )
            print(
                f"eval step={step + 1} task={env.base_task_name} "
                f"return={eval_metrics['return_mean']:.3f} "
                f"success={eval_metrics['success_mean']:.3f} "
                f"goal_success={eval_metrics['goal_success_mean']:.3f} "
                f"replay={len(replay_buffer)} sps={sps}"
            )
            write_csv(os.path.join(args.save_path, f"{log_name}.csv"), metrics_rows)

        if args.save_model_freq > 0 and (step + 1) % args.save_model_freq == 0:
            agent.save(args.save_path, f"{log_name}_step_{step + 1}")

    agent.save(args.save_path, log_name)
    write_csv(os.path.join(args.save_path, f"{log_name}.csv"), metrics_rows)
    if wandb_run is not None:
        wandb_run.finish()
    env.close()


if __name__ == "__main__":
    main()