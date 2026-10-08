r"""
A base configuration class that can be useful for setting up both online and
offline experiments.
"""

from typing import *

import os
import abc

import attrs
import yaml
import logging
import socket
import subprocess
import tempfile

import hydra
import hydra.types
import hydra.core.config_store
from omegaconf import OmegaConf, DictConfig, SCMode

from tqdm.auto import tqdm
import numpy as np
import torch
import torch.backends.cudnn
import torch.multiprocessing
from torch.utils.tensorboard import SummaryWriter

from . import data, modules, utils


@attrs.define(kw_only=True, auto_attribs=True)
class DeviceConfig:
    type: str = "cuda"
    index: Optional[int] = 0

    def make(self):
        return torch.device(self.type, self.index)


class ExperimentWriter:
    def __init__(
        self, tensorboard_writer: SummaryWriter, wandb_run: Optional[Any] = None
    ) -> None:
        self.tensorboard_writer = tensorboard_writer
        self.wandb_run = wandb_run

    def add_scalar(
        self, tag: str, scalar_value: Any, global_step: Optional[int] = None
    ):
        self.tensorboard_writer.add_scalar(tag, scalar_value, global_step)
        if self.wandb_run is not None:
            value = (
                scalar_value.item() if hasattr(scalar_value, "item") else scalar_value
            )
            payload = {tag: value}
            if global_step is not None:
                payload["global_step"] = global_step
            self.wandb_run.log(payload)

    def add_text(self, tag: str, text_string: str, global_step: Optional[int] = None):
        self.tensorboard_writer.add_text(tag, text_string, global_step)
        if self.wandb_run is not None:
            payload = {tag: text_string}
            if global_step is not None:
                payload["global_step"] = global_step
            self.wandb_run.log(payload)

    def close(self):
        self.tensorboard_writer.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()


@attrs.define(kw_only=True)
class BaseConf(abc.ABC):
    hydra: Dict = dict(
        output_subdir=None,
        job=dict(chdir=False),
        run=dict(dir=tempfile.TemporaryDirectory().name),
        mode=hydra.types.RunMode.RUN,
    )

    base_git_dir: str = subprocess.check_output(
        r"git rev-parse --show-toplevel".split(),
        cwd=os.path.dirname(__file__),
        encoding="utf-8",
    ).strip()
    git_commit: str = subprocess.check_output(
        r"git rev-parse HEAD".split(), cwd=os.path.dirname(__file__), encoding="utf-8"
    ).strip()
    git_status: Tuple[str] = tuple(
        l.strip()
        for l in subprocess.check_output(
            r"git status --short".split(),
            cwd=os.path.dirname(__file__),
            encoding="utf-8",
        )
        .strip()
        .split("\n")
    )

    overwrite_output: bool = False

    @property
    @abc.abstractmethod
    def output_base_dir(self) -> str:

        pass

    output_folder: Optional[str] = None
    output_folder_suffix: Optional[str] = None
    output_dir: Optional[str] = attrs.field(default=None, init=False)

    wandb: bool = True
    wandb_project: str = "quasimetric-rl"
    wandb_entity: Optional[str] = None
    wandb_mode: Optional[str] = None
    wandb_tags: Tuple[str, ...] = ()

    @property
    def completion_file(self) -> str:
        return os.path.join(self.output_dir, "COMPLETE")

    device: DeviceConfig = DeviceConfig()

    seed: int = 60912

    @property
    @abc.abstractmethod
    def env(self) -> data.Dataset.Conf:

        pass

    agent: modules.QRLConf = modules.QRLConf()

    @classmethod
    def from_DictConfig(cls, cfg: DictConfig) -> "BaseConf":
        return OmegaConf.to_container(cfg, structured_config_mode=SCMode.INSTANTIATE)

    def setup_for_experiment(self) -> ExperimentWriter:
        r"""
        1. Finalize conf fields
        2. Do basic checks
        3. Setup logging, seeding, etc.
        4. Returns an experiment logger
        """

        if self.output_dir is not None:
            raise RuntimeError("setup_for_experiment() can only be called once")

        if self.output_folder is None:
            specs = [
                self.agent.quasimetric_critic.model.quasimetric_model.quasimetric_head_spec,
                f"dyn={self.agent.quasimetric_critic.losses.latent_dynamics.weight:g}",
            ]
            if self.agent.num_critics > 1:
                specs.append(f"{self.agent.num_critics}critic")
            if self.agent.actor is not None:
                aspecs = []
                if self.agent.actor.losses.min_dist.add_goal_as_future_state:
                    aspecs.append("goal=Rand+Future")
                else:
                    aspecs.append("goal=Rand")
                if self.agent.actor.losses.min_dist.adaptive_entropy_regularizer:
                    aspecs.append("ent")
                if self.agent.actor.losses.behavior_cloning.weight > 0:
                    aspecs.append(
                        f"BC={self.agent.actor.losses.behavior_cloning.weight:g}"
                    )
                specs.append("actor(" + ",".join(aspecs) + ")")
            specs.append(
                f"seed={self.seed}",
            )
            if self.output_folder_suffix is not None:
                specs.append(self.output_folder_suffix)
            self.output_folder = os.path.join(
                f"{self.env.kind}_{self.env.name}",
                "_".join(specs),
            )
        assert os.path.exists(self.output_base_dir)
        self.output_dir = os.path.join(self.output_base_dir, self.output_folder)
        utils.mkdir(self.output_dir)

        if os.path.exists(self.completion_file):
            if self.overwrite_output:
                logging.warning(f"Overwriting output directory {self.output_dir}")
            else:
                raise RuntimeError(
                    f"Output directory {self.output_dir} exists and is complete"
                )

        utils.logging.configure(os.path.join(self.output_dir, "output.log"))
        config_yaml = OmegaConf.to_yaml(self)

        wandb_run = None
        if self.wandb:
            try:
                import wandb as wandb_lib

                init_kwargs = dict(
                    project=self.wandb_project,
                    entity=self.wandb_entity,
                    name=self.output_folder.replace(os.sep, "/"),
                    dir=self.output_dir,
                    config=yaml.safe_load(config_yaml),
                    tags=list(self.wandb_tags) if len(self.wandb_tags) else None,
                    reinit=True,
                )
                if self.wandb_mode is not None:
                    init_kwargs["mode"] = self.wandb_mode
                wandb_run = wandb_lib.init(**init_kwargs)
                wandb_run.define_metric("global_step")
                wandb_run.define_metric("*", step_metric="global_step")
                logging.info(f"W&B run initialized: {wandb_run.url}")
            except Exception as exc:
                logging.warning(f"W&B disabled because initialization failed: {exc}")

        writer = ExperimentWriter(SummaryWriter(self.output_dir), wandb_run)

        logging.info("")
        logging.info(config_yaml)
        logging.info("")
        logging.info(f"Running on {socket.getfqdn()}:")
        logging.info(f"\t{'PID':<30}{os.getpid()}")
        for var in ["CUDA_VISIBLE_DEVICES", "EGL_DEVICE_ID"]:
            logging.info(f"\t{var:<30}{os.environ.get(var, None)}")
        logging.info("")
        logging.info(f"Output directory {self.output_dir}")
        logging.info("")

        with open(os.path.join(self.output_dir, "config.yaml"), "w") as f:
            f.write(config_yaml)
        writer.add_text("config", f"```\n{config_yaml}\n```")

        logging.info("")
        logging.info(f"Base Git directory {self.base_git_dir}")
        logging.info(f"Git COMMIT: {self.git_commit}")
        logging.info(f"Git status:\n    " + "\n    ".join(self.git_status))
        with open(os.path.join(self.output_dir, "git_summary.yaml"), "w") as f:
            f.write(
                yaml.safe_dump(
                    dict(
                        base_dir=self.base_git_dir,
                        commit=self.git_commit,
                        status=self.git_status,
                    )
                )
            )
        with open(
            os.path.join(self.output_dir, f"git_{self.git_commit}.patch"), "w"
        ) as f:
            f.write(subprocess.getoutput(f"git diff {self.git_commit}"))
        logging.info("")

        torch_seed, np_seed = utils.split_seed(cast(int, self.seed), 2)
        np.random.seed(np.random.Generator(np.random.PCG64(np_seed)).integers(1 << 31))
        torch.manual_seed(
            np.random.Generator(np.random.PCG64(torch_seed)).integers(1 << 31)
        )

        torch.backends.cudnn.benchmark = True
        torch.set_num_threads(12)

        return writer
