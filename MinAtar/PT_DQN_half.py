import numpy as np
import pickle
import itertools
import importlib
import torch
import torch.optim as optim
import copy

from model import *
from replay import *
from CL_envs import *

from argparse import ArgumentParser
from configparser import ConfigParser
from tqdm import tqdm
import os, time

parser = ArgumentParser(description="Parameters for the code - ARTD on gym envs")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--env-name", type=str, default="all", help="Environment Name")
parser.add_argument(
    "--t-steps", type=int, default=3500000, help="total number of steps"
)
parser.add_argument("--switch", type=int, default=500000, help="switch env steps")
parser.add_argument("--lr1", type=float, default=1e-8, help="learning rate for weights")
parser.add_argument(
    "--lr2", type=float, default=1e-4, help="learning rate for transient values"
)
parser.add_argument("--update", type=int, default=50000, help="PM update frequency")
parser.add_argument(
    "--decay", type=float, default=0, help="decay transient weights after transfer"
)
parser.add_argument(
    "--batch-size", type=int, default=64, help="Number of samples per batch"
)
parser.add_argument("--save", action="store_true")
parser.add_argument("--plot", action="store_true")
parser.add_argument("--save-model", action="store_true")
parser.add_argument(
    "--gpu", type=int, default=0, help="Random seed and device selector"
)
parser.add_argument(
    "--seq", type=int, default=0, help="selected sequence in the environment list"
)
parser.add_argument(
    "--CNNhalf", type=int, default=1, help="half CNN, 1: half CNN(default), 0: full CNN"
)
parser.add_argument(
    "--boundary", type=int, default=0, help="1: known boundary, 0: unknown boundary"
)
parser.add_argument(
    "--reset", type=int, default=1, help="default=1, reset every environment"
)
parser.add_argument(
    "--log-interval", type=int, default=1000, help="training steps between W&B logs"
)
parser.add_argument("--wandb-project", type=str, default="minatar-pt-dqn-half")
parser.add_argument("--wandb-entity", type=str, default=None)
parser.add_argument("--wandb-group", type=str, default=None)
parser.add_argument("--wandb-name", type=str, default=None)
parser.add_argument("--wandb-dir", type=str, default="results")
parser.add_argument(
    "--wandb-mode",
    type=str,
    default="online",
    choices=["online", "offline", "disabled"],
)

args = parser.parse_args()
config = ConfigParser()
config.read("misc_params.cfg")
misc_param = config[str(args.env_name)]
gamma = float(misc_param["gamma"])
epsilon = float(misc_param["epsilon"])


class WandbLogger:
    def __init__(self, args, run_name):
        self.run = None
        self.wandb = None
        if args.wandb_mode == "disabled":
            return
        try:
            self.wandb = importlib.import_module("wandb")
        except ImportError:
            print("W&B is unavailable; install wandb or use --wandb-mode disabled.")
            return

        os.makedirs(args.wandb_dir, exist_ok=True)
        try:
            self.run = self.wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                group=args.wandb_group,
                name=args.wandb_name or run_name,
                config={**vars(args), "gamma": gamma, "epsilon": epsilon},
                dir=args.wandb_dir,
                mode=args.wandb_mode,
            )
            self.run.define_metric("global_step")
            for namespace in ("train", "episode", "task", "permanent"):
                self.run.define_metric(f"{namespace}/*", step_metric="global_step")
            print(f"W&B run: {self.run.url or self.run.path}")
        except Exception as error:
            print(f"W&B initialization failed: {error}")
            self.run = None

    def log(self, metrics):
        if self.run is not None:
            self.run.log(metrics)

    def log_artifact(self, path, name, artifact_type):
        if self.run is None or self.wandb is None:
            return
        try:
            artifact = self.wandb.Artifact(name=name, type=artifact_type)
            artifact.add_file(path)
            self.run.log_artifact(artifact)
        except Exception as error:
            print(f"W&B artifact upload failed for {path}: {error}")

    def finish(self, games, final_average_return, episode_count):
        if self.run is not None:
            self.run.summary["task_sequence"] = games
            self.run.summary["final_average_return"] = final_average_return
            self.run.summary["episode_count"] = episode_count
            self.run.finish()
            self.run = None


def train_T_Net():
    states, actions, next_states, rewards, done = exp_replay.sample()
    with torch.no_grad():
        T_next_pred = Target_net(next_states)
        P_next_pred = P_Net(next_states)
        P_pred = P_Net(states)
        P_pred = P_pred.gather(1, actions)
    T_pred = T_Net(states)
    T_pred = T_pred.gather(1, actions)

    targets = rewards + (1 - done) * gamma * (
        (P_next_pred + T_next_pred).max(1)[0]
    ).reshape(-1, 1)
    loss = T_criterion(T_pred + P_pred, targets)
    T_opt.zero_grad()
    loss.backward()
    T_opt.step()
    return loss.item()


def train_P_Net():
    loss_u = 0
    u_steps = (exp_replay_PM.size() // args.batch_size) - 1
    for p_update in range(u_steps):
        curr_batch = list(
            itertools.islice(
                exp_replay_PM.memory,
                p_update * args.batch_size,
                (p_update + 1) * args.batch_size,
            )
        )
        states, actions, old_p_vals = map(torch.stack, zip(*curr_batch))
        states = states.to(device)
        actions = actions.to(device)
        old_p_vals = old_p_vals.to(device)
        with torch.no_grad():
            T_pred = T_Net(states).gather(1, actions)
        P_pred = P_Net(states).gather(1, actions)
        loss = P_criterion(P_pred, T_pred + old_p_vals)
        P_opt.zero_grad()
        loss.backward()
        P_opt.step()
        loss_u += loss.item()
    return loss_u / u_steps


def get_action(c_obs):
    c_obs = np.moveaxis(c_obs, 2, 0)
    c_obs = torch.tensor(c_obs, dtype=torch.float).to(device)
    with torch.no_grad():
        curr_T_vals = T_Net(c_obs.unsqueeze(0))
        curr_P_vals = P_Net(c_obs.unsqueeze(0))
        curr_Q_vals = curr_T_vals + curr_P_vals
    if np.random.random() <= epsilon:
        action = env.action_space.sample()
    else:
        action = curr_Q_vals.max(1)[1].item()
    return curr_P_vals[0][action], action


if torch.cuda.is_available():
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

filename = (
    "PT_DQN_0.5x"
    + "_env_name_"
    + str(args.env_name)
    + "_gamma_"
    + misc_param["gamma"]
    + "_steps_"
    + str(args.t_steps)
    + "_switch_"
    + str(args.switch)
    + "_update_"
    + str(args.update)
    + "_decay_"
    + str(args.decay)
    + "_lr1_"
    + str(args.lr1)
    + "_lr2_"
    + str(args.lr2)
    + "_batch_"
    + str(args.batch_size)
    + "_seq_"
    + str(args.seq)
    + "_CNNhalf_"
    + str(args.CNNhalf)
    + "_boundary"
    + str(args.boundary)
    + "_reset_"
    + str(args.reset)
    + "_seed_"
    + str(args.seed)
)

if args.log_interval <= 0:
    raise ValueError("log-interval must be positive")
logger = WandbLogger(args, filename)


Games = []
gameid = 0
env = CL_envs_func_replacement(seq=args.seq, game_id=gameid, seed=args.seed)
Games.append(env.game_name)

in_channels = env.observation_space.shape[2]
num_actions = env.action_space.n


if args.CNNhalf == 1:
    T_Net = CNN_half(in_channels, num_actions).to(device)
else:
    T_Net = CNN(in_channels, num_actions).to(device)
T_opt = optim.Adam(T_Net.parameters(), lr=args.lr2)
T_criterion = torch.nn.MSELoss()

if args.CNNhalf == 1:
    P_Net = CNN_half(in_channels, num_actions).to(device)
else:
    P_Net = CNN(in_channels, num_actions).to(device)
P_opt = optim.SGD(P_Net.parameters(), lr=args.lr1)
P_criterion = torch.nn.MSELoss()


if args.CNNhalf == 1:
    Target_net = CNN_half(in_channels, num_actions).to(device)
else:
    Target_net = CNN(in_channels, num_actions).to(device)

Target_net.load_state_dict(T_Net.state_dict())

exp_replay = expReplay(batch_size=args.batch_size, device=device)
if args.boundary == 1:
    exp_replay_PM = expReplay_PM(
        max_size=args.switch, batch_size=args.batch_size, device=device
    )
else:
    exp_replay_PM = expReplay_PM(
        max_size=args.update, batch_size=args.batch_size, device=device
    )

returns_array = np.zeros(args.t_steps)

avg_return = 0
epi_return = 0
done = False
cs = env.reset()
episode_count = 0
last_t_loss = None
last_p_loss = None

logger.log(
    {
        "global_step": 0,
        "task/id": gameid,
        "task/game": env.game_name,
        "task/start": 1,
        "task/reset": args.reset,
    }
)

for step in tqdm(range(args.t_steps)):
    if step % args.switch == 0 and step > 0:
        logger.log(
            {
                "global_step": step,
                "task/id": gameid,
                "task/game": env.game_name,
                "task/end": 1,
            }
        )

        gameid += 1
        env = CL_envs_func_replacement(seq=args.seq, game_id=gameid, seed=args.seed)
        Games.append(env.game_name)

        cs = env.reset()
        epi_return = 0

        if args.reset == 1:
            if args.CNNhalf == 1:
                T_Net = CNN_half(in_channels, num_actions).to(device)
            else:
                T_Net = CNN(in_channels, num_actions).to(device)
            T_opt = optim.Adam(T_Net.parameters(), lr=args.lr2)

            avg_return = 0

            print("Reset the fast learner")

        logger.log(
            {
                "global_step": step,
                "task/id": gameid,
                "task/game": env.game_name,
                "task/start": 1,
                "task/reset": args.reset,
            }
        )

    val_p, c_action = get_action(cs)
    ns, rew, done, _ = env.step(c_action)
    epi_return += rew
    exp_replay.store(cs, c_action, ns, rew, done)
    exp_replay_PM.store(cs, c_action, val_p)

    if step % 1000 == 0 and step > 0:
        Target_net.load_state_dict(T_Net.state_dict())

    if exp_replay.size() >= args.batch_size:
        last_t_loss = train_T_Net()

    cs = ns

    if args.boundary == 1:
        if (step + 1) % args.switch == 0:
            last_p_loss = train_P_Net()
            logger.log(
                {
                    "global_step": step + 1,
                    "permanent/loss": last_p_loss,
                    "permanent/task_id": gameid,
                    "permanent/update": 1,
                }
            )

            T_Net.__init__(in_channels, num_actions)
            T_Net = T_Net.to(device)
            exp_replay_PM.delete()

    else:
        if (step + 1) % args.update == 0:
            last_p_loss = train_P_Net()
            logger.log(
                {
                    "global_step": step + 1,
                    "permanent/loss": last_p_loss,
                    "permanent/task_id": gameid,
                    "permanent/update": 1,
                }
            )
            for params in T_Net.parameters():
                params.data *= args.decay

    if done:
        completed_return = epi_return
        cs = env.reset()
        avg_return = 0.99 * avg_return + 0.01 * completed_return
        epi_return = 0
        episode_count += 1
        logger.log(
            {
                "global_step": step + 1,
                "episode/return": float(completed_return),
                "episode/average_return": float(avg_return),
                "episode/count": episode_count,
                "episode/task_id": gameid,
            }
        )

    returns_array[step] = copy.copy(avg_return)

    if (step + 1) % args.log_interval == 0:
        train_metrics = {
            "global_step": step + 1,
            "train/task_id": gameid,
            "train/task_step": (step % args.switch) + 1,
            "train/reward": float(rew),
            "train/average_return": float(avg_return),
            "train/replay_size": exp_replay.size(),
            "train/permanent_replay_size": exp_replay_PM.size(),
            "train/epsilon": epsilon,
        }
        if last_t_loss is not None:
            train_metrics["train/transient_loss"] = last_t_loss
        if last_p_loss is not None:
            train_metrics["train/permanent_loss"] = last_p_loss
        logger.log(train_metrics)

    if (step + 1) % args.switch == 0:
        exp_replay.delete()
        print("Clear the buffer, the current memory size is :", exp_replay.size())

        if args.save_model:
            os.makedirs("models", exist_ok=True)
            model_path = "models/" + filename + "_Net" + str(gameid) + ".pt"
            torch.save(P_Net.state_dict(), model_path)
            logger.log_artifact(
                model_path,
                f"pt-dqn-half-seq{args.seq}-seed{args.seed}-task{gameid}",
                "model",
            )


if args.save:
    os.makedirs("results", exist_ok=True)
    results_path = "results/" + filename + "_returns.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(returns_array, f)
    logger.log_artifact(
        results_path,
        f"pt-dqn-half-seq{args.seq}-seed{args.seed}-returns",
        "results",
    )

print("Games: ", Games, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
logger.log(
    {
        "global_step": args.t_steps,
        "task/id": gameid,
        "task/game": env.game_name,
        "task/end": 1,
    }
)
logger.finish(Games, float(avg_return), episode_count)
