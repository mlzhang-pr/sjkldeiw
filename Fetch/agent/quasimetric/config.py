from dataclasses import dataclass
from typing import Optional


@dataclass
class QuasimetricConfig:
    latent_dim: int = 256
    hidden_dim: int = 256
    hidden_depth: int = 2
    transition_input: str = "state"
    components: int = 8
    batch_size: int = 256
    lr: float = 1e-4
    discount: float = 0.995
    lambda_: float = 0.95
    next_state_sample: float = 0.2
    backup_clip: float = 5.0
    backup_coef: float = 1.0
    diag_backup: float = 1.0
    action_invariance_coef: float = 0.0
    transition_consistency_coef: float = 1.0
    contrastive_coef: float = 0.05
    ranking_coef: float = 0.0
    ranking_margin: float = 0.1
    nce_mode: str = "forward_nce"
    target_tau: float = 0.01
    current_batch_ratio: float = 0.5
    max_grad_norm: Optional[float] = None
    min_buffer_size: int = 256


@dataclass
class ContinualQuasimetricAgentConfig:
    structure_update_frequency: int = 4    #######
    structure_updates_per_step: int = 1
    structure_bonus_coef: float = 1.0
    q_loss_coef: float = 1.0
    bc_alpha: float = 0.1
    memory_max_tasks: Optional[int] = None
    memory_max_transitions_per_task: Optional[int] = None
    share_sac_batch: bool = True
    shared_batch_size: Optional[int] = None
    goal_reward_scale: float = 0.1
    task_reward_scale: float = 1.0
    goal_reward_type: str = "binary"
    behavior_goal_success_only: bool = True
    encode_actor_critic_goal: bool = False
