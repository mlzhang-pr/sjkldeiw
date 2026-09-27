"""Resume CQRL4 online training after updating meta from the task-1 buffer.

All CQRL4 arguments remain available. This entrypoint defaults to
``--start_task 2`` and requires ``--resume_run_dir`` to identify the task-1
checkpoint and accumulated quasimetric buffer. Before task-2 policy selection,
the restored meta agent is updated from that buffer using CQRL4's normal
task-boundary update schedule. Use ``--student_mode shared`` to continue the
restored task-1 student or ``--student_mode independent`` to keep it separate
and warm up a freshly initialized task-2 student.
"""

import os
import sys

import torch

try:
	from . import CQRL4
except ImportError:
	import CQRL4


def _option_value(argv, *options):
	for index, argument in enumerate(argv):
		for option in options:
			if argument == option:
				return argv[index + 1] if index + 1 < len(argv) else None
			if argument.startswith(f"{option}="):
				return argument.split("=", 1)[1]
	return None


def _has_option(argv, *options):
	return _option_value(argv, *options) is not None


def _checkpoint_transition_input(argv):
	resume_run_dir = _option_value(argv, "--resume_run_dir")
	if resume_run_dir is None:
		return None

	start_task = int(_option_value(argv, "--start_task") or 2)
	model_name = _option_value(argv, "--resume_model_name")
	if model_name is None:
		source_run_name = os.path.basename(os.path.normpath(resume_run_dir))
		model_name = f"{source_run_name}_task{start_task - 1}"

	model_dir = os.path.join(os.path.abspath(resume_run_dir), "model")
	quasimetric_path = os.path.join(
		model_dir,
		f"{model_name}_meta_quasimetric.pt",
	)
	actor_path = os.path.join(model_dir, f"{model_name}_meta_actor.pt")
	if not os.path.isfile(quasimetric_path) or not os.path.isfile(actor_path):
		return None

	payload = torch.load(quasimetric_path, map_location="cpu", weights_only=False)
	quasimetric_config = payload.get("quasimetric_cfg", {})
	saved_mode = quasimetric_config.get("transition_input")
	if saved_mode in {"state", "latent"}:
		return saved_mode

	model_state = payload["quasimetric"]["model_state"]
	observation_dim = int(model_state["state_encoder.trunk.0.weight"].shape[1])
	transition_dim = int(
		model_state["latent_transition_encoder.trunk.0.weight"].shape[1]
	)
	latent_dim = int(quasimetric_config.get("latent_dim", 256))

	actor_state = torch.load(actor_path, map_location="cpu", weights_only=False)
	actor_weights = [
		(key, value)
		for key, value in actor_state.items()
		if key.startswith("trunk.") and key.endswith(".weight")
	]
	if not actor_weights:
		raise ValueError(f"Cannot infer action dimension from {actor_path}")
	_, output_weight = max(
		actor_weights,
		key=lambda item: int(item[0].split(".")[1]),
	)
	if output_weight.shape[0] % 2 != 0:
		raise ValueError(f"Invalid Gaussian actor output shape in {actor_path}")
	action_dim = int(output_weight.shape[0] // 2)

	matches = []
	if transition_dim == observation_dim + action_dim:
		matches.append("state")
	if transition_dim == latent_dim + action_dim:
		matches.append("latent")
	if len(matches) != 1:
		raise ValueError(
			"Cannot uniquely infer --qm_transition_input from the resume checkpoint. "
			"Pass either 'state' or 'latent' explicitly."
		)
	return matches[0]


def main():
	if not _has_option(sys.argv[1:], "--start_task"):
		sys.argv.extend(["--start_task", "2"])
	if "--update_meta_on_resume" not in sys.argv[1:]:
		sys.argv.append("--update_meta_on_resume")
	if not _has_option(
		sys.argv[1:],
		"--qm_transition_input",
		"--qm-transition-input",
	):
		transition_input = _checkpoint_transition_input(sys.argv[1:])
		if transition_input is not None:
			sys.argv.extend(["--qm_transition_input", transition_input])
			print("restored qm_transition_input:", transition_input)
	return CQRL4.main()


if __name__ == "__main__":
	sys.exit(main())

	'''
	cd Fetch/quasimetric-rl

python -m online_continual.CQRL4_offline_to_online2 \
  --resume_run_dir online_continual/results/cqrl_wd/fetch_cqrl_push-slide-pick-and-place_seed0_gc-sparse_slide-scale0.795 \
  --save_path online_continual/results/cqrl4_from_task1 \
  --gpu 0 \
  --distill_loss_type kl

cd Fetch/quasimetric-rl

python -m online_continual.CQRL4_offline_to_online2 \
  --resume_run_dir online_continual/results/cqrl_wd/fetch_cqrl_push-slide-pick-and-place_seed0_gc-sparse_slide-scale0.795 \
  --save_path online_continual/results/cqrl4_from_task1_kl \
  --gpu 0 \
  --distill_loss_type kl

  cd Fetch/quasimetric-rl

conda run -n RLL3 python -m online_continual.CQRL4_offline_to_online2 \
  --resume_run_dir online_continual/results/cqrl_wd/fetch_cqrl_push-slide-pick-and-place_seed0_gc-sparse_slide-scale0.795 \
  --start_task 2 \
  --task_switch_selection meta \
  --distill_loss_type wd \
  --save_path online_continual/results/cqrl4_independent_kl \
  --gpu 0

python -m online_continual.CQRL4_offline_to_online2 \
  --resume_run_dir online_continual/results/cqrl_wd/fetch_cqrl_push-slide-pick-and-place_seed0_gc-sparse_slide-scale0.795 \
  --start_task 2 \
  --task_switch_selection meta \
  --distill_loss_type kl \
  --distill_goal_source current_replay \
  --lambda_reg 0.5 \
  --warmup_steps 25000 \
  --save_path online_continual/results/cqrl4_independent_kl_current_goal \
  --gpu 0

-----------------------------------
cd Fetch/quasimetric-rl
python -m online_continual.CQRL4_offline_to_online2 \
  --resume_run_dir online_continual/results/cqrl_wd/fetch_cqrl_push-slide-pick-and-place_seed0_gc-sparse_slide-scale0.795 \
  --start_task 2 \
  --student_mode shared \
  --task_switch_selection meta \
  --distill_loss_type kl \
  --distill_goal_source current_replay \
  --lambda_reg 0.5 \
  --warmup_steps 25000 \
  --save_path online_continual/results/cqrl4_shared_kl_lambda_reg_0.5 \
  --gpu 0   best

python -m online_continual.CQRL4_offline_to_online2 \
  --resume_run_dir online_continual/results/cqrl_wd/fetch_cqrl_push-slide-pick-and-place_seed0_gc-sparse_slide-scale0.795 \
  --start_task 2 \
  --student_mode shared \
  --task_switch_selection meta \
  --distill_loss_type kl \
  --distill_goal_source quasimetric_buffer \
  --lambda_reg 0.1 \
  --warmup_steps 10000 \
  --save_path online_continual/results/cqrl4_shared_kl_lambda_reg_0.1_quasimetric_buffer \
  --gpu 1 \
  --qm_transition_input latent


python -m online_continual.CQRL4_offline_to_online2 \
  --resume_run_dir online_continual/results/cqrl_wd/fetch_cqrl_push-slide-pick-and-place_seed0_gc-sparse_slide-scale0.795 \
  --start_task 2 \
  --student_mode shared \
  --task_switch_selection meta \
  --distill_loss_type wd \
  --distill_goal_source current_replay \
  --lambda_reg 0.5 \
  --warmup_steps 20000 \
  --save_path online_continual/results/cqrl4_shared_wd_lambda_reg_0.5 \
  --gpu 0   


python -m online_continual.CQRL4_offline_to_online2 \
  --resume_run_dir online_continual/results/cqrl_wd/fetch_cqrl_push-slide-pick-and-place_seed0_gc-sparse_slide-scale0.795 \
  --start_task 2 \
  --student_mode shared \
  --task_switch_selection meta \
  --distill_loss_type kl \
  --distill_goal_source quasimetric_buffer \
  --lambda_reg 0.5 \
  --warmup_steps 25000 \
  --save_path online_continual/results/cqrl4_shared_kl_lambda_reg_0.5_quasimetric_buffer \
  --gpu 1  

	python -m online_continual.CQRL4_offline_to_online2 \
  --resume_run_dir online_continual/results/cqrl_wd/fetch_cqrl_push-slide-pick-and-place_seed0_gc-sparse_slide-scale0.795 \
  --start_task 2 \
  --student_mode shared \
  --task_switch_selection meta \
  --distill_loss_type kl \
  --distill_goal_source current_replay \
  --lambda_reg 1.0 \
  --warmup_steps 25000 \
  --save_path online_continual/results/cqrl4_shared_kl_lambda_reg_1.0 \
  --gpu 0
  
  python -m online_continual.CQRL4_offline_to_online2 \
	--resume_run_dir online_continual/results/cqrl_wd/fetch_cqrl_push-slide-pick-and-place_seed0_gc-sparse_slide-scale0.795 \
	--start_task 2 \
	--student_mode shared \
	--task_switch_selection meta \
	--distill_loss_type kl \
	--distill_goal_source current_replay \
	--lambda_reg 0.5 \
	--warmup_steps 50000 \
	--save_path online_continual/results/cqrl4_shared_kl_lambda_reg_0.5_warm_50000 \
	--gpu 1     best

	 python -m online_continual.CQRL4_offline_to_online2 \
		--resume_run_dir online_continual/results/cqrl_wd/fetch_cqrl_push-slide-pick-and-place_seed0_gc-sparse_slide-scale0.795 \
		--start_task 2 \
		--student_mode shared \
		--task_switch_selection meta \
		--distill_loss_type kl \
		--distill_goal_source current_replay \
		--lambda_reg 0.5 \
		--warmup_steps 20000 \
		--save_path online_continual/results/cqrl4_shared_kl_lambda_reg_0.5_warm_20000 \
		--gpu 1

		python -m online_continual.CQRL4_offline_to_online2 \
			--resume_run_dir online_continual/results/cqrl_wd/fetch_cqrl_push-slide-pick-and-place_seed0_gc-sparse_slide-scale0.795 \
			--start_task 2 \
			--student_mode shared \
			--task_switch_selection meta \
			--distill_loss_type kl \
			--distill_goal_source current_replay \
			--lambda_reg 0.5 \
			--warmup_steps 50000 \
			--save_path online_continual/results/offline_to_online2/cqrl4_shared_kl_lambda_reg_0.5_warm_50000_diag_backup1.0 \
			--gpu 1 \
			--seed 0 \
			--qm_diag_backup 1.0

  quasimetric_buffer
	'''
