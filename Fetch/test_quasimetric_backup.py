import pytest
import torch

from agent.quasimetric.config import QuasimetricConfig
from agent.quasimetric.structure import MultistepQuasimetricLearner


class ScalarDistanceLearner(MultistepQuasimetricLearner):
    def distance(self, x, y):
        return torch.abs(x - y).sum(dim=-1)


def make_scalar_learner(diag_backup):
    config = QuasimetricConfig(
        latent_dim=2,
        hidden_dim=4,
        hidden_depth=1,
        components=1,
        discount=0.5,
        backup_clip=100.0,
        diag_backup=diag_backup,
    )
    return ScalarDistanceLearner(1, 1, "cpu", config=config)


def backup_inputs():
    return (
        torch.tensor([[1.0], [3.0]]),
        torch.tensor([[0.0], [4.0]]),
        torch.tensor([[0.5], [2.0]]),
        torch.tensor([[0.0], [4.0]]),
        torch.tensor([0.0, 1.0]),
        torch.tensor([1.0, 2.0]),
    )


def test_default_backup_matches_existing_per_sample_formula():
    assert QuasimetricConfig().diag_backup == 1.0
    transition, goal, intermediate, target_goal, dones, offsets = backup_inputs()

    loss, dist, dist_next = make_scalar_learner(1.0)._backup_loss(
        transition,
        goal,
        intermediate,
        target_goal,
        dones,
        offsets,
    )

    expected_dist = torch.abs(transition - goal).sum(dim=-1)
    expected_dist_next = torch.abs(intermediate - target_goal).sum(dim=-1)
    delta = expected_dist - expected_dist_next
    bootstrap = torch.pow(torch.full_like(offsets, 0.5), offsets) * (1.0 - dones)
    expected_loss = (bootstrap * torch.exp(delta) - expected_dist).mean()

    assert dist.shape == (2,)
    assert torch.equal(dist, expected_dist)
    assert torch.equal(dist_next, expected_dist_next)
    assert torch.allclose(loss, expected_loss)


@pytest.mark.parametrize("diag_backup", [0.0, 0.25])
def test_cross_sample_backup_matches_mqe_matrix_formula(diag_backup):
    transition, goal, intermediate, target_goal, dones, offsets = backup_inputs()

    loss, dist, dist_next = make_scalar_learner(diag_backup)._backup_loss(
        transition,
        goal,
        intermediate,
        target_goal,
        dones,
        offsets,
    )

    expected_dist = torch.abs(
        transition[:, None, :] - goal[None, :, :]
    ).sum(dim=-1)
    expected_dist_next = torch.abs(
        intermediate[:, None, :] - target_goal[None, :, :]
    ).sum(dim=-1)
    delta = expected_dist - expected_dist_next
    bootstrap = (
        torch.pow(torch.full_like(offsets, 0.5), offsets) * (1.0 - dones)
    )[:, None]
    backup = bootstrap * torch.exp(delta) - expected_dist
    expected_loss = (
        (1.0 - diag_backup) * backup
        + diag_backup * torch.diagonal(backup)[:, None]
    ).mean()

    assert dist.shape == (2, 2)
    assert torch.equal(dist, expected_dist)
    assert torch.equal(dist_next, expected_dist_next)
    assert torch.allclose(loss, expected_loss)


@pytest.mark.parametrize("diag_backup", [-0.1, 1.1])
def test_diag_backup_rejects_values_outside_unit_interval(diag_backup):
    with pytest.raises(ValueError, match="diag_backup"):
        make_scalar_learner(diag_backup)


def test_cross_sample_backup_supports_real_network_backward():
    config = QuasimetricConfig(
        latent_dim=8,
        hidden_dim=8,
        hidden_depth=1,
        components=2,
        diag_backup=0.0,
        contrastive_coef=0.0,
    )
    learner = MultistepQuasimetricLearner(3, 2, "cpu", config=config)
    batch_size = 4
    batch = {
        "obses": torch.randn(batch_size, 3),
        "actions": torch.randn(batch_size, 2),
        "next_obses": torch.randn(batch_size, 3),
        "dones": torch.zeros(batch_size),
        "value_goals": torch.randn(batch_size, 3),
        "intermediate_value_goals": torch.randn(batch_size, 3),
        "intermediate_value_goals_offsets": torch.ones(batch_size),
    }

    loss, metrics = learner.compute_loss(batch)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["backup_loss"])
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in learner._trainable_parameters
    )