import sys

import importlib
import numpy as np
import pickle
import itertools
import torch
import torch.optim as optim
import copy

from model import *
from replay import *
from CL_envs import *

from argparse import ArgumentParser
from configparser import ConfigParser
from tqdm import tqdm
import os
import time

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
import matplotlib.backends.backend_pdf
from matplotlib.lines import Line2D

parser = ArgumentParser(description="Parameters for the code - DQN")
parser.add_argument('--seed', type=int, default=0, help="Random seed")
parser.add_argument('--env-name', type=str, default="all", help="Environment Name")
parser.add_argument('--t-steps', type=int, default=3500000, help="total number of steps")
parser.add_argument('--switch', type=int, default=500000, help="switch env steps")
parser.add_argument('--lr1', type=float, default=1e-5, help="learning rate for DQN")
parser.add_argument('--batch-size', type=int, default=64, help="Number of samples per batch")
parser.add_argument('--save', action="store_true")
parser.add_argument('--plot', action="store_true")
parser.add_argument('--save-model', action="store_true")
parser.add_argument("--gpu", type=int, default=0, help="Random seed and device selector")

parser.add_argument('--seq', type=int, default=0, help="selected sequence in the environment list")
parser.add_argument('--reset', type=int, default=0, help="reset every environment, 1: Reset, 0: finetune")
parser.add_argument('--log-interval', type=int, default=1000, help="training steps between W&B logs")
parser.add_argument('--wandb-project', type=str, default="minatar-dqn")
parser.add_argument('--wandb-entity', type=str, default=None)
parser.add_argument('--wandb-group', type=str, default=None)
parser.add_argument('--wandb-name', type=str, default=None)
parser.add_argument('--wandb-dir', type=str, default="results")
parser.add_argument('--wandb-mode', type=str, default="online", choices=["online", "offline", "disabled"])

args = parser.parse_args()
config = ConfigParser()
config.read('misc_params.cfg')
misc_param = config[str(args.env_name)]
gamma = float(misc_param['gamma'])
epsilon = float(misc_param['epsilon'])


class WandbLogger:
	def __init__(self, args, run_name):
		self.run = None
		self.training_mode = "reset" if args.reset == 1 else "finetune"
		if args.wandb_mode == "disabled":
			return
		try:
			wandb = importlib.import_module("wandb")
		except ImportError:
			print("W&B is unavailable; install wandb or use --wandb-mode disabled.")
			return

		os.makedirs(args.wandb_dir, exist_ok=True)
		try:
			self.run = wandb.init(
				project=args.wandb_project,
				entity=args.wandb_entity,
				group=args.wandb_group,
				name=args.wandb_name or run_name,
				config={**vars(args), "gamma": gamma, "epsilon": epsilon},
				dir=args.wandb_dir,
				mode=args.wandb_mode,
			)
			self.run.define_metric("global_step")
			for namespace in ("train", "episode", "task"):
				self.run.define_metric(f"{namespace}/*", step_metric="global_step")
			print(f"W&B run: {self.run.url or self.run.path}")
		except Exception as error:
			print(f"W&B initialization failed: {error}")
			self.run = None

	def log(self, metrics):
		if self.run is not None:
			self.run.log(metrics)

	def finish(self, games):
		if self.run is not None:
			self.run.summary["task_sequence"] = games
			self.run.summary["training_mode"] = self.training_mode
			self.run.finish()
			self.run = None


print("torch.__version__:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("torch.cuda.is_available():", torch.cuda.is_available())
print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))
print("device_count:", torch.cuda.device_count())


def train_Net():
	states, actions, next_states, rewards, done = exp_replay.sample()
	with torch.no_grad():
		next_pred = Target_net(next_states)
		next_pred = next_pred.max(1)[0]
	pred = Net(states)
	pred = pred.gather(1, actions)
	targets = rewards + (1 - done) * gamma * next_pred.reshape(-1, 1)
	loss = criterion(pred, targets)
	opt.zero_grad()
	loss.backward()
	opt.step()
	return loss.item()

def get_action(c_obs):
	c_obs = np.moveaxis(c_obs, 2, 0)
	c_obs = torch.tensor(c_obs, dtype=torch.float).to(device)
	with torch.no_grad():
		curr_Q_vals = Net(c_obs.unsqueeze(0))
	if np.random.random() <= epsilon:
		action = env.action_space.sample()
	else:
		action = curr_Q_vals.max(1)[1].item()
	return curr_Q_vals[0][action], action


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

filename = ("DQN"+"_env_name_"+args.env_name+"_gamma_"+misc_param['gamma']+\
		"_steps_"+str(args.t_steps)+"_switch_"+str(args.switch)+"_batch_"+\
		str(args.batch_size)+"_lr1_"+str(args.lr1) + "_seq_" + str(args.seq) + "_reset_"+str(args.reset) +"_seed_"+str(args.seed))

if args.log_interval <= 0:
	raise ValueError("log-interval must be positive")
logger = WandbLogger(args, filename)

def moving_average(a, n=3):
    cumsum_vec = np.cumsum(np.insert(a, 0, 0))
    ma_vec = (cumsum_vec[n:] - cumsum_vec[:-n]) / n
    return np.concatenate((a[0:n-1]/n, ma_vec))


Games = []
gameid = 0
env = CL_envs_func_replacement(seq=args.seq, game_id=gameid, seed=args.seed)
Games.append(env.game_name)

in_channels = env.observation_space.shape[2]
num_actions = env.action_space.n

Net = CNN(in_channels, num_actions).to(device)
opt = optim.Adam(Net.parameters(), lr=args.lr1)
criterion = torch.nn.MSELoss()

Target_net = CNN(in_channels, num_actions).to(device)
Target_net.load_state_dict(Net.state_dict())

exp_replay = expReplay(batch_size=args.batch_size, device=device)

returns_array = np.zeros(args.t_steps)

avg_return = 0
epi_return = 0
done = False
cs = env.reset()
episode_count = 0
last_loss = None

logger.log({
	"global_step": 0,
	"task/id": gameid,
	"task/game": env.game_name,
	"task/start": 1,
	"task/reset": args.reset,
	"task/finetune": int(args.reset == 0),
})


for step in tqdm(range(args.t_steps)):



	if step % args.switch == 0 and step > 0:
		logger.log({
			"global_step": step,
			"task/id": gameid,
			"task/game": env.game_name,
			"task/end": 1,
		})

		gameid += 1
		env = CL_envs_func_replacement(seq=args.seq, game_id=gameid, seed=args.seed)
		Games.append(env.game_name)
		cs = env.reset()
		epi_return = 0

		avg_return = 0

		if args.reset == 1:

			Net = CNN(in_channels, num_actions).to(device)
			opt = optim.Adam(Net.parameters(), lr=args.lr1)
			print('Reset the fast learner')

		logger.log({
			"global_step": step,
			"task/id": gameid,
			"task/game": env.game_name,
			"task/start": 1,
			"task/reset": args.reset,
			"task/finetune": int(args.reset == 0),
			"train/replay_size": exp_replay.size(),
		})
	
	_, c_action = get_action(cs)
	ns, rew, done, _ = env.step(c_action)
	epi_return += rew
	exp_replay.store(cs, c_action, ns, rew, done)


	if step % 1000 == 0 and step > 0:
		Target_net.load_state_dict(Net.state_dict())

	if exp_replay.size() >= args.batch_size:
		last_loss = train_Net()
	
	cs = ns

	if done:
		completed_return = epi_return
		cs = env.reset()
		avg_return = 0.99 * avg_return + 0.01 * completed_return
		epi_return = 0
		episode_count += 1
		logger.log({
			"global_step": step + 1,
			"episode/return": float(completed_return),
			"episode/average_return": float(avg_return),
			"episode/count": episode_count,
			"episode/task_id": gameid,
		})

	returns_array[step] = copy.copy(avg_return)

	if step % args.log_interval == 0:
		train_metrics = {
			"global_step": step + 1,
			"train/task_id": gameid,
			"train/task_step": step % args.switch,
			"train/reward": float(rew),
			"train/average_return": float(avg_return),
			"train/replay_size": exp_replay.size(),
			"train/epsilon": epsilon,
			"train/finetune": int(args.reset == 0),
		}
		if last_loss is not None:
			train_metrics["train/loss"] = last_loss
		logger.log(train_metrics)




	if (step + 1) % args.switch == 0:

		exp_replay.delete()
		print('Clear the buffer, the current memory size is :', exp_replay.size())

		if args.save_model:
			os.makedirs("models", exist_ok=True)
			torch.save(Net.state_dict(), "models/"+filename+"_Net" + str(gameid) +".pt")


if args.save:
	os.makedirs("results", exist_ok=True)
	with open("results/"+filename+"_returns.pkl", "wb") as f:
		pickle.dump(returns_array, f)

print('Games: ', Games, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
logger.log({
	"global_step": args.t_steps,
	"task/id": gameid,
	"task/game": env.game_name,
	"task/end": 1,
})
logger.finish(Games)
