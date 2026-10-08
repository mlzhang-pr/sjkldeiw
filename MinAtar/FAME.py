import time
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
from torch.optim.lr_scheduler import ExponentialLR
from scipy import stats

parser = ArgumentParser(description="Parameters for the FAME in MinAtari")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--env-name", type=str, default="all", help="Environment Name")
parser.add_argument(
    "--t-steps", type=int, default=3500000, help="total number of steps"
)
parser.add_argument("--switch", type=int, default=500000, help="switch env steps")
parser.add_argument(
    "--lr1", type=float, default=1e-3, help="learning rate for meta learner"
)
parser.add_argument(
    "--lr2", type=float, default=1e-5, help="learning rate for fast learner"
)
parser.add_argument("--update", type=int, default=50000, help="PM update frequency")
parser.add_argument(
    "--decay", type=float, default=0.75, help="decay transient weights after transfer"
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
    "--size_fast2meta", type=int, default=12000, help="size of fast2meta buffer"
)
parser.add_argument("--size_meta", type=int, default=100000, help="size of meta buffer")
parser.add_argument(
    "--detection_step",
    type=int,
    default=600,
    help="detection step: number of expisodes in detection, 300 steps average episode!",
)
parser.add_argument(
    "--epoch_meta", type=int, default=200, help="epoch to train meta learner"
)
parser.add_argument("--reset", type=int, default=1, help="reset the network every time")


parser.add_argument(
    "--warmstep", type=int, default=50000, help="the number of steps to do warm-up"
)
parser.add_argument(
    "--lambda_reg",
    type=float,
    default=1.0,
    help="hyperparameter for the regularization behavior cloning term, default method",
)
parser.add_argument(
    "--p_explore",
    type=float,
    default=0.0,
    help="probability of using the meta policy for guided exploration",
)


parser.add_argument(
    "--use_ttest",
    type=int,
    default=0,
    help="one-vs-one hypothesis test:, 1: on, 0: off, default off",
)


parser.add_argument(
    "--log-interval", type=int, default=1000, help="training steps between W&B logs"
)
parser.add_argument("--wandb-project", type=str, default="minatar-fame")
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
            for namespace in ("train", "episode", "task", "detection", "meta"):
                self.run.define_metric(f"{namespace}/*", step_metric="global_step")
            print(f"W&B run: {self.run.url or self.run.path}")
        except Exception as error:
            print(f"W&B initialization failed: {error}")
            self.run = None

    def log(self, metrics):
        if self.run is not None:
            self.run.log(metrics)

    def finish(self, games, regularization):
        if self.run is not None:
            self.run.summary["task_sequence"] = games
            self.run.summary["regularization_choices"] = regularization
            self.run.finish()
            self.run = None


print("torch.__version__:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("torch.cuda.is_available():", torch.cuda.is_available())
print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))
print("device_count:", torch.cuda.device_count())
torch.cuda.init()
device = torch.device(f"cuda:{args.gpu}")
torch.cuda.set_device(device)
print("device_count:", torch.cuda.device_count())

print(args)
num_envs = int(args.t_steps / args.switch)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)


def train_faster(reg=None):
    states, actions, next_states, rewards, done = exp_replay_fast.sample()
    with torch.no_grad():
        fast_next_pred = Target_net(next_states)

    targets = rewards + (1 - done) * gamma * (fast_next_pred.max(1)[0]).reshape(-1, 1)
    fast_pred = Fast_Learner(states).gather(1, actions)
    loss = Fast_criterion(fast_pred, targets)

    if reg is not None and args.lambda_reg > 0:
        with torch.no_grad():
            soft_target = F.softmax(reg(states), dim=-1)
        logit_input = F.log_softmax(Fast_Learner(states), dim=-1)
        loss_reg = F.kl_div(logit_input, soft_target, reduction="batchmean")
        loss = loss + args.lambda_reg * loss_reg

    Fast_opt.zero_grad()
    loss.backward()
    Fast_opt.step()
    return loss.item()


def train_meta():

    Meta_Learner_old = copy.deepcopy(Meta_Learner).to(device)

    u_steps = (exp_replay_meta.size() // args.batch_size) - 1
    for epoch in range(args.epoch_meta):
        for i, p_update in enumerate(range(u_steps)):
            states_meta, actions_meta = exp_replay_meta.sample()

            states_meta = states_meta.to(device)
            actions_meta = actions_meta.to(device)

            logits = Meta_Learner(states_meta)
            log_probs = F.log_softmax(logits, dim=-1)
            loss1 = Meta_criterion2(log_probs, actions_meta.view(-1))

            if i % gameid == 0:
                states_fast, actions_fast = exp_replay_fast2meta.sample()

                logits = Meta_Learner(states_fast)
                log_probs = F.log_softmax(logits, dim=-1)
                loss2 = Meta_criterion2(log_probs, actions_fast.view(-1))

                loss = loss1 + loss2
            else:
                loss = loss1

            Meta_opt.zero_grad()
            loss.backward()
            Meta_opt.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch: {epoch + 1}/{args.epoch_meta}, Meta Loss: {loss.item():.2e}, current lr: {Meta_opt.param_groups[0]['lr']:.2e}",
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            )
            logger.log(
                {
                    "global_step": step,
                    "meta/task_id": gameid,
                    "meta/epoch": epoch + 1,
                    "meta/loss": loss.item(),
                    "meta/learning_rate": Meta_opt.param_groups[0]["lr"],
                }
            )

        if (epoch + 1) % 2 == 0:
            Meta_scheduler.step()


def get_action_detection(c_obs, testQ):
    c_obs = np.moveaxis(c_obs, 2, 0)
    c_obs = torch.tensor(c_obs, dtype=torch.float).to(device)
    with torch.no_grad():
        curr_Q_vals = testQ(c_obs.unsqueeze(0))
    action = curr_Q_vals.max(1)[1].item()
    return action, curr_Q_vals[0][action]


def get_action(c_obs, LEARNER):
    c_obs = np.moveaxis(c_obs, 2, 0)
    c_obs = torch.tensor(c_obs, dtype=torch.float).to(device)
    with torch.no_grad():
        curr_Q_vals = LEARNER(c_obs.unsqueeze(0))
    if np.random.random() <= epsilon:
        action = env.action_space.sample()
    else:
        action = curr_Q_vals.max(1)[1].item()
    return action


def get_action_exploration(c_obs, learner, expert, p_explore):
    c_obs = np.moveaxis(c_obs, 2, 0)
    c_obs = torch.tensor(c_obs, dtype=torch.float).to(device)
    with torch.no_grad():
        curr_Q_vals = learner(c_obs.unsqueeze(0))
    if np.random.random() <= epsilon:
        if np.random.random() <= p_explore:
            with torch.no_grad():
                expert_Q_vals = expert(c_obs.unsqueeze(0))
            action = expert_Q_vals.max(1)[1].item()
        else:
            action = env.action_space.sample()
    else:
        action = curr_Q_vals.max(1)[1].item()
    return action


if torch.cuda.is_available():
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


filename = (
    "FAME"
    + "_steps_"
    + str(args.t_steps)
    + "_switch_"
    + str(args.switch)
    + "_update_"
    + str(args.update)
    + "_lr1_"
    + str(args.lr1)
    + "_lr2_"
    + str(args.lr2)
    + "_size_fast2meta_"
    + str(args.size_fast2meta)
    + "_detection_step_"
    + str(args.detection_step)
    + "_seq_"
    + str(args.seq)
    + "_epoch_meta_"
    + str(args.epoch_meta)
    + "_warmstep_"
    + str(args.warmstep)
    + "_lambda_reg_"
    + str(args.lambda_reg)
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


Fast_Learner = CNN(in_channels, num_actions).to(device)
Fast_opt = optim.Adam(Fast_Learner.parameters(), lr=args.lr2)
Fast_criterion = torch.nn.MSELoss()


Random_Learner = CNN(in_channels, num_actions).to(device)
Meta_Learner = CNN(in_channels, num_actions).to(device)
Meta_opt = optim.Adam(Meta_Learner.parameters(), lr=args.lr1)
Meta_scheduler = ExponentialLR(Meta_opt, gamma=0.95)
Meta_criterion = torch.nn.MSELoss()
Meta_criterion2 = torch.nn.NLLLoss()
Meta2fast_criterion = torch.nn.MSELoss()


Target_net = CNN(in_channels, num_actions).to(device)
Target_net.load_state_dict(Fast_Learner.state_dict())


exp_replay_fast = expReplay(batch_size=args.batch_size, device=device)

exp_replay_fast2meta = expReplay_Meta(
    max_size=args.size_fast2meta, batch_size=args.batch_size, device=device
)
exp_replay_meta = expReplay_Meta(
    max_size=args.size_meta, batch_size=args.batch_size, device=device
)


returns_array = np.zeros(args.t_steps)

avg_return = 0
epi_return = 0
done = False
cs = env.reset()
print(
    f"##################### Environment {gameid + 1}/{num_envs}: {env.game_name}#####################"
)
logger.log(
    {
        "global_step": 0,
        "task/id": gameid,
        "task/game": env.game_name,
        "task/start": 1,
    }
)


interval = [
    (i * args.switch - args.size_fast2meta - 1, i * args.switch - 1)
    for i in range(1, int(args.t_steps / args.switch) + 1)
]


def in_intervals(x):
    return any(start <= x <= end for start, end in interval)


Reg_Learner = None
pbar = tqdm(total=args.t_steps)

Flag_Reg = []
MAX_STEP = 300
step = 0

Q_normalize = []

META_WARMUP = 0
episode_count = 0
last_loss = None

while step < args.t_steps:
    if step % args.switch == 0 and step > 0:
        logger.log(
            {
                "global_step": step,
                "task/id": gameid,
                "task/game": env.game_name,
                "task/end": 1,
            }
        )

        if args.reset == 1:
            avg_return = 0

        META_WARMUP = 0

        gameid += 1
        old_envname = env.game_name
        env = CL_envs_func_replacement(seq=args.seq, game_id=gameid, seed=args.seed)
        print(
            f"##################### Environment {gameid + 1}/{num_envs}: {env.game_name}#####################"
        )
        Games.append(env.game_name)
        cs = env.reset()
        cs_initial = cs
        logger.log(
            {
                "global_step": step,
                "task/id": gameid,
                "task/game": env.game_name,
                "task/start": 1,
            }
        )

        print("##################### Step 1: Detection via Policy Evaluation !")

        FLAG_ENV2 = True if step != args.switch else False

        if not FLAG_ENV2:
            print(
                "No Detection for meta, only compare fast and reset as in the 2nd environment!"
            )

        epi_return = 0

        max_step = 0

        Num_detection_meta = args.detection_step * FLAG_ENV2
        Num_detection_fast = args.detection_step

        epi_return_fast = 0
        avereward_fast = []

        for step_small in range(Num_detection_fast):
            c_action, _ = get_action_detection(cs, Fast_Learner)
            ns, rew, done, _ = env.step(c_action)
            exp_replay_fast.store(cs, c_action, ns, rew, done)
            epi_return_fast += rew
            cs = ns
            step += 1
            max_step += 1
            if done or max_step > MAX_STEP:
                cs = env.reset()
                avg_return = 0.99 * avg_return + 0.01 * epi_return_fast
                avereward_fast.append(epi_return_fast)
                epi_return_fast = 0
                max_step = 0
            returns_array[step] = copy.copy(avg_return)
            pbar.update(1)
        if Num_detection_fast > 0:
            if len(avereward_fast) == 0:
                print(
                    f"Evaluation on Fast Learner, Number of Episodes: {len(avereward_fast)}",
                    "Even one episode is not finished yet....",
                )
            else:
                print(
                    f"Evaluation on Fast Learner, Average Reward: {np.mean(avereward_fast)}, Number of Episodes: {len(avereward_fast)}, all: ",
                    avereward_fast,
                )
        else:
            print(f"No Evaluation on Fast Learner")

        epi_return_meta = 0
        avereward_meta = []
        max_step = 0

        if Num_detection_meta > 0:
            cs = env.reset()

        for step_small in range(Num_detection_meta):
            c_action, _ = get_action_detection(cs, Meta_Learner)
            ns, rew, done, _ = env.step(c_action)
            exp_replay_fast.store(cs, c_action, ns, rew, done)
            epi_return_meta += rew
            cs = ns
            step += 1
            max_step += 1
            if done or max_step > MAX_STEP:
                cs = env.reset()
                avg_return = 0.99 * avg_return + 0.01 * epi_return_meta
                avereward_meta.append(epi_return_meta)
                epi_return_meta = 0
                max_step = 0
            returns_array[step] = copy.copy(avg_return)
            pbar.update(1)

        if Num_detection_meta > 0:
            if len(avereward_meta) == 0:
                print(
                    f"Evaluation on Meta Learner, Number of Episodes: {len(avereward_meta)}",
                    "Even one episode is not finished yet....",
                )
            else:
                print(
                    f"Evaluation on Meta Learner, Average Reward: {np.mean(avereward_meta)}, Number of Episodes: {len(avereward_meta)}, all: ",
                    avereward_meta,
                )
        else:
            print(f"No Evaluation on Meta Learner")

        Avereward_meta = -1000 if len(avereward_meta) == 0 else np.mean(avereward_meta)
        Avereward_fast = -1000 if len(avereward_fast) == 0 else np.mean(avereward_fast)

        _, value_rand = get_action_detection(cs_initial, Random_Learner)

        Avereward_rand = float(value_rand.cpu().numpy())
        print(
            "Reward meta",
            round(Avereward_meta, 2),
            "Reward_fast",
            round(Avereward_fast, 2),
            "Reward_random",
            round(Avereward_rand, 2),
        )

        def Hypothesis_test(avereward_list1, avereward_list2, Avereward1, Avereward2):

            if len(avereward_list1) < 2 or len(avereward_list2) < 2:
                return Avereward1 > Avereward2
            else:
                t_statistic, p_value = stats.ttest_ind(
                    avereward_list1,
                    avereward_list2,
                    alternative="greater",
                    equal_var=False,
                )
                return p_value < 0.05

        Meta_Fast = (
            Hypothesis_test(
                avereward_meta, avereward_fast, Avereward_meta, Avereward_fast
            )
            if args.use_ttest == 1
            else (Avereward_meta > Avereward_fast)
        )
        Fast_Meta = (
            Hypothesis_test(
                avereward_fast, avereward_meta, Avereward_fast, Avereward_meta
            )
            if args.use_ttest == 1
            else (Avereward_fast > Avereward_meta)
        )

        if Meta_Fast and Avereward_meta > Avereward_rand:
            print(
                "##################### Step 2: Use Meta Initialization and Start Training !"
            )
            META_WARMUP = 1
            Flag_Reg.append("Meta")

        elif Fast_Meta and Avereward_fast > Avereward_rand:
            Flag_Reg.append("Fast")
            print(
                "##################### Step 2: Use Fast Initialization / FineTune Fast Learner!"
            )
        else:
            if args.reset == 1:
                Fast_Learner = CNN(in_channels, num_actions).to(device)
                Fast_opt = optim.Adam(Fast_Learner.parameters(), lr=args.lr2)
                Flag_Reg.append("Random")
                print("##################### Step 2: Use Random Initialization")
            else:
                Flag_Reg.append("Fast")
                print(
                    "##################### Step 2: Use Fast Initialization / FineTune Fast Learner!"
                )

        logger.log(
            {
                "global_step": step,
                "detection/task_id": gameid,
                "detection/meta_return": float(Avereward_meta),
                "detection/fast_return": float(Avereward_fast),
                "detection/random_value": float(Avereward_rand),
                "detection/selected_initialization": Flag_Reg[-1],
            }
        )

        if Num_detection_meta + Num_detection_fast > 0:
            Target_net.load_state_dict(Fast_Learner.state_dict())

            cs = env.reset()

    if args.p_explore > 0 and META_WARMUP == 1 and (step % args.switch < args.warmstep):
        if step % args.switch == args.warmstep - 1:
            print(f"Finished the guided exploration at the step {step}")
        c_action = get_action_exploration(
            cs, Fast_Learner, Meta_Learner, args.p_explore
        )
    else:
        c_action = get_action(cs, Fast_Learner)
    ns, rew, done, _ = env.step(c_action)
    epi_return += rew

    exp_replay_fast.store(cs, c_action, ns, rew, done)

    if in_intervals(step):
        exp_replay_fast2meta.store(cs, c_action)

    if step % 1000 == 0 and step > 0:
        Target_net.load_state_dict(Fast_Learner.state_dict())

    if exp_replay_fast.size() >= args.batch_size:
        if (
            META_WARMUP == 1
            and args.lambda_reg > 0
            and (step % args.switch < args.warmstep)
        ):
            if step % args.switch == args.warmstep - 1:
                print(
                    f"Use the behavior cloning as an regularization at the step {step}"
                )
            last_loss = train_faster(reg=Meta_Learner)
        else:
            last_loss = train_faster(reg=None)

    cs = ns

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

    if step % args.log_interval == 0:
        train_metrics = {
            "global_step": step + 1,
            "train/task_id": gameid,
            "train/task_step": step % args.switch,
            "train/reward": float(rew),
            "train/average_return": float(avg_return),
            "train/fast_buffer_size": exp_replay_fast.size(),
            "train/meta_buffer_size": exp_replay_meta.size(),
            "train/epsilon": epsilon,
        }
        if last_loss is not None:
            train_metrics["train/loss"] = last_loss
        logger.log(train_metrics)

    if (step + 1) % args.switch == 0:
        if step + 1 == args.switch:
            print("First time: No need to update Meta learner")
        else:
            print("##################### Step 3: Updating Meta Learner!")
            print(
                "Old Meta data set: ",
                exp_replay_meta.size(),
                "fast data set: ",
                exp_replay_fast2meta.size(),
            )

            Meta_opt = optim.Adam(Meta_Learner.parameters(), lr=args.lr1)
            Meta_scheduler = ExponentialLR(Meta_opt, gamma=0.95)
            train_meta()

        exp_replay_fast2meta.copy_to(exp_replay_meta)
        exp_replay_fast2meta.delete()
        print(
            "##################### Step 4: Fast2Meta Copy to Meta buffer: New Meta data set: ",
            exp_replay_meta.size(),
            "fast data set: ",
            exp_replay_fast2meta.size(),
        )

        exp_replay_fast.delete()

        if args.save_model:
            os.makedirs("models", exist_ok=True)
            torch.save(
                Meta_Learner.state_dict(),
                "models/" + filename + "_Meta" + str(gameid) + ".pt",
            )

    step += 1
    pbar.update(1)

pbar.close()


if args.save:
    os.makedirs("results", exist_ok=True)
    with open("results/" + filename + "_returns.pkl", "wb") as f:
        pickle.dump(returns_array, f)


print(
    "Regularization: ", Flag_Reg, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
)
print("Games: ", Games, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print(args)
logger.log(
    {
        "global_step": min(step, args.t_steps),
        "task/id": gameid,
        "task/game": env.game_name,
        "task/end": 1,
    }
)
logger.finish(Games, Flag_Reg)
