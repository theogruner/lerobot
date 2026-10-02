"""Relative end-effector poses (xyz + column-convention rot6d) for relative-action training.

Covers lerobot.utils.pose6d and the pose-group path of RelativeActionsProcessorStep /
AbsoluteActionsProcessorStep / compute_relative_action_stats. No dataset download needed.
"""

import numpy as np
import pytest
import torch

from lerobot.datasets.compute_stats import _compute_relative_chunk_batch, compute_relative_action_stats
from lerobot.processor import TransitionKey, create_transition
from lerobot.processor.relative_action_processor import (
    AbsoluteActionsProcessorStep,
    RelativeActionsProcessorStep,
)
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.pose6d import (
    matrix_to_rot6d,
    pose_group_names,
    pose_to_absolute,
    pose_to_relative,
    rot6d_to_matrix,
)

Rotation = pytest.importorskip("scipy.spatial.transform").Rotation

IDENTITY_6D = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float64)

# The oopsie bimanual EE layout: per arm xyz + rot6d + gripper; state carries finger width.
ACTION_NAMES = pose_group_names("right_ee") + ["right_gripper"] + pose_group_names("left_ee") + ["left_gripper"]
STATE_NAMES = (
    pose_group_names("right_ee") + ["right_gripper_finger"] + pose_group_names("left_ee") + ["left_gripper_finger"]
)


def _random_poses(n: int, seed: int = 0) -> torch.Tensor:
    rot = Rotation.random(n, random_state=seed).as_matrix()
    pos = np.random.default_rng(seed).normal(size=(n, 3))
    return torch.cat([torch.from_numpy(pos), matrix_to_rot6d(torch.from_numpy(rot))], dim=-1)


def _random_ee_vectors(shape: tuple[int, ...], seed: int = 0) -> torch.Tensor:
    """(..., 20) vectors in the oopsie EE layout with valid poses and random grippers."""
    n = int(np.prod(shape))
    right, left = _random_poses(n, seed), _random_poses(n, seed + 1)
    grip = torch.from_numpy(np.random.default_rng(seed + 2).uniform(0, 1.1, size=(n, 2)))
    return torch.cat([right, grip[:, :1], left, grip[:, 1:]], dim=-1).reshape(*shape, 20).float()


# pose6d


def test_rot6d_uses_matrix_columns():
    mats = torch.from_numpy(Rotation.random(16, random_state=1).as_matrix())
    r6 = matrix_to_rot6d(mats)
    torch.testing.assert_close(r6[:, :3], mats[:, :, 0])
    torch.testing.assert_close(r6[:, 3:], mats[:, :, 1])
    torch.testing.assert_close(rot6d_to_matrix(r6), mats)


def test_rot6d_to_matrix_orthonormalises_noisy_input():
    r6 = matrix_to_rot6d(torch.from_numpy(Rotation.random(64, random_state=2).as_matrix()))
    mats = rot6d_to_matrix(r6 + 0.05 * torch.randn_like(r6))
    eye = torch.eye(3, dtype=mats.dtype).expand_as(mats)
    torch.testing.assert_close(mats.transpose(-1, -2) @ mats, eye, atol=1e-6, rtol=0)
    torch.testing.assert_close(torch.linalg.det(mats), torch.ones(64, dtype=mats.dtype))


@pytest.mark.parametrize("frame", ["ee", "world"])
def test_pose_roundtrip(frame):
    pose, ref = _random_poses(256, seed=3), _random_poses(256, seed=4)
    back = pose_to_absolute(pose_to_relative(pose, ref, frame), ref, frame)
    torch.testing.assert_close(back, pose, atol=1e-9, rtol=0)


@pytest.mark.parametrize("frame", ["ee", "world"])
def test_pose_relative_to_itself_is_identity(frame):
    pose = _random_poses(32, seed=5)
    rel = pose_to_relative(pose, pose, frame)
    torch.testing.assert_close(rel[:, :3], torch.zeros(32, 3, dtype=rel.dtype), atol=1e-9, rtol=0)
    torch.testing.assert_close(rel[:, 3:], IDENTITY_6D.expand(32, 6), atol=1e-9, rtol=0)


def test_ee_frame_matches_scipy_reference():
    pose, ref = _random_poses(64, seed=6), _random_poses(64, seed=7)
    rel = pose_to_relative(pose, ref, "ee")
    r_ref = Rotation.from_matrix(rot6d_to_matrix(ref[:, 3:]).numpy())
    r_pose = Rotation.from_matrix(rot6d_to_matrix(pose[:, 3:]).numpy())
    expected_p = r_ref.inv().apply((pose[:, :3] - ref[:, :3]).numpy())
    expected_r = (r_ref.inv() * r_pose).as_matrix()
    np.testing.assert_allclose(rel[:, :3].numpy(), expected_p, atol=1e-9)
    np.testing.assert_allclose(rot6d_to_matrix(rel[:, 3:]).numpy(), expected_r, atol=1e-9)


def test_world_frame_matches_delta_cartesian_convention():
    """data_werkzeug's DeltaCartesian: R_err = R_target * R_cmd^-1, p_err = p_target - p_cmd."""
    pose, ref = _random_poses(64, seed=8), _random_poses(64, seed=9)
    rel = pose_to_relative(pose, ref, "world")
    r_ref = Rotation.from_matrix(rot6d_to_matrix(ref[:, 3:]).numpy())
    r_pose = Rotation.from_matrix(rot6d_to_matrix(pose[:, 3:]).numpy())
    np.testing.assert_allclose(rel[:, :3].numpy(), (pose[:, :3] - ref[:, :3]).numpy(), atol=1e-9)
    np.testing.assert_allclose(
        rot6d_to_matrix(rel[:, 3:]).numpy(), (r_pose * r_ref.inv()).as_matrix(), atol=1e-9
    )


def test_ee_frame_is_invariant_to_a_global_rigid_transform():
    pose, ref = _random_poses(64, seed=10), _random_poses(64, seed=11)
    g_rot = torch.from_numpy(Rotation.random(random_state=12).as_matrix())
    g_pos = torch.tensor([0.3, -1.2, 0.7], dtype=torch.float64)

    def move(p):
        rot = g_rot @ rot6d_to_matrix(p[:, 3:])
        return torch.cat([p[:, :3] @ g_rot.T + g_pos, matrix_to_rot6d(rot)], dim=-1)

    torch.testing.assert_close(
        pose_to_relative(move(pose), move(ref), "ee"), pose_to_relative(pose, ref, "ee"), atol=1e-9, rtol=0
    )
    assert not torch.allclose(pose_to_relative(move(pose), move(ref), "world"), pose_to_relative(pose, ref, "world"))


def test_float32_input_stays_float32():
    pose, ref = _random_poses(8, seed=13).float(), _random_poses(8, seed=14).float()
    assert pose_to_relative(pose, ref).dtype == torch.float32
    assert pose_to_absolute(pose, ref).dtype == torch.float32


def test_invalid_frame_rejected():
    pose = _random_poses(2)
    with pytest.raises(ValueError, match="pose frame"):
        pose_to_relative(pose, pose, "base")


# processor steps


def _ee_step(**kwargs) -> RelativeActionsProcessorStep:
    defaults = dict(
        enabled=True,
        exclude_joints=["gripper"],
        action_names=ACTION_NAMES,
        state_names=STATE_NAMES,
        pose_groups=["right_ee", "left_ee"],
        pose_frame="ee",
    )
    return RelativeActionsProcessorStep(**{**defaults, **kwargs})


@pytest.mark.parametrize("frame", ["ee", "world"])
def test_step_roundtrip_through_transitions(frame):
    step = _ee_step(pose_frame=frame)
    absolute = AbsoluteActionsProcessorStep(enabled=True, relative_step=step)
    action = _random_ee_vectors((4, 50), seed=20)
    state = _random_ee_vectors((4,), seed=21)

    rel = step(create_transition(observation={OBS_STATE: state}, action=action))[TransitionKey.ACTION]
    back = absolute(create_transition(action=rel))[TransitionKey.ACTION]
    torch.testing.assert_close(back, action, atol=1e-5, rtol=0)


def test_step_converts_poses_and_keeps_grippers_absolute():
    step = _ee_step()
    action = _random_ee_vectors((3, 50), seed=22)
    state = _random_ee_vectors((3,), seed=23)
    rel = step.to_relative(action, state)

    for gripper in (ACTION_NAMES.index("right_gripper"), ACTION_NAMES.index("left_gripper")):
        torch.testing.assert_close(rel[..., gripper], action[..., gripper])
    for prefix in ("right_ee", "left_ee"):
        idx = [ACTION_NAMES.index(n) for n in pose_group_names(prefix)]
        expected = pose_to_relative(action[..., idx], state[:, None, idx], "ee")
        torch.testing.assert_close(rel[..., idx], expected)


def test_step_state_equal_to_first_action_gives_identity_at_t0():
    step = _ee_step()
    action = _random_ee_vectors((2, 10), seed=24)
    rel = step.to_relative(action, action[:, 0].clone())
    idx = [ACTION_NAMES.index(n) for n in pose_group_names("right_ee")]
    torch.testing.assert_close(rel[:, 0, idx[:3]], torch.zeros(2, 3), atol=1e-6, rtol=0)
    torch.testing.assert_close(rel[:, 0, idx[3:]], IDENTITY_6D.float().expand(2, 6), atol=1e-6, rtol=0)


def test_step_resolves_state_dims_by_name_not_position():
    """A state whose pose dims sit elsewhere than the action's still anchors correctly."""
    step = _ee_step()
    action = _random_ee_vectors((2, 5), seed=25)
    state = _random_ee_vectors((2,), seed=26)
    perm = list(range(20))[::-1]
    shuffled = _ee_step(state_names=[STATE_NAMES[i] for i in perm])
    torch.testing.assert_close(shuffled.to_relative(action, state[:, perm]), step.to_relative(action, state))


def test_step_without_pose_groups_is_unchanged_subtraction():
    step = RelativeActionsProcessorStep(enabled=True, exclude_joints=["gripper"], action_names=ACTION_NAMES)
    action = _random_ee_vectors((2, 5), seed=27)
    state = _random_ee_vectors((2,), seed=28)
    mask = torch.tensor(step._build_mask(20), dtype=action.dtype)
    torch.testing.assert_close(step.to_relative(action, state), action - state[:, None] * mask)


def test_step_rejects_missing_pose_names():
    step = _ee_step(pose_groups=["right_arm"])
    with pytest.raises(ValueError, match="right_arm"):
        step.to_relative(_random_ee_vectors((1, 2)), _random_ee_vectors((1,)))


def test_step_rejects_excluded_pose_dims():
    step = _ee_step(exclude_joints=["gripper", "rot6d"])
    with pytest.raises(ValueError, match="both excluded"):
        step.to_relative(_random_ee_vectors((1, 2)), _random_ee_vectors((1,)))


def test_step_rejects_invalid_frame():
    with pytest.raises(ValueError, match="pose_frame"):
        _ee_step(pose_frame="base")


def test_step_config_roundtrip():
    step = _ee_step(pose_frame="world")
    rebuilt = RelativeActionsProcessorStep(**step.get_config())
    action, state = _random_ee_vectors((2, 5), seed=29), _random_ee_vectors((2,), seed=30)
    torch.testing.assert_close(rebuilt.to_relative(action, state), step.to_relative(action, state))
    assert step.get_config()["pose_groups"] == ["right_ee", "left_ee"]


# stats


def _fake_dataset(n_episodes=3, length=40, seed=31):
    actions, states, episodes = [], [], []
    for ep in range(n_episodes):
        actions.append(_random_ee_vectors((length,), seed=seed + 2 * ep).numpy())
        states.append(_random_ee_vectors((length,), seed=seed + 2 * ep + 1).numpy())
        episodes.append(np.full(length, ep))
    hf = {
        ACTION: list(np.concatenate(actions)),
        OBS_STATE: list(np.concatenate(states)),
        "episode_index": list(np.concatenate(episodes)),
    }
    features = {
        ACTION: {"shape": (20,), "names": ACTION_NAMES},
        OBS_STATE: {"shape": (20,), "names": STATE_NAMES},
    }
    return hf, features


def test_chunk_batch_with_step_matches_step_conversion():
    hf, _ = _fake_dataset()
    all_actions = np.asarray(hf[ACTION], dtype=np.float32)
    all_states = np.asarray(hf[OBS_STATE], dtype=np.float32)
    starts = np.array([0, 5, 45])
    step = _ee_step()
    out = _compute_relative_chunk_batch(starts, all_actions, all_states, 10, step)
    frame_idx = starts[:, None] + np.arange(10)[None]
    expected = step.to_relative(torch.from_numpy(all_actions[frame_idx]), torch.from_numpy(all_states[starts]))
    np.testing.assert_allclose(out, expected.reshape(-1, 20).numpy(), atol=1e-6)


def test_relative_stats_with_pose_groups():
    hf, features = _fake_dataset()
    stats = compute_relative_action_stats(
        hf, features, chunk_size=10, exclude_joints=["gripper"], pose_groups=["right_ee", "left_ee"]
    )
    plain = compute_relative_action_stats(hf, features, chunk_size=10, exclude_joints=["gripper"])
    grip = [ACTION_NAMES.index("right_gripper"), ACTION_NAMES.index("left_gripper")]
    np.testing.assert_allclose(stats["mean"][grip], plain["mean"][grip], atol=1e-6)
    rot = [ACTION_NAMES.index(n) for n in pose_group_names("right_ee")[3:]]
    # Composed rot6d stays a unit-column representation; subtraction does not.
    assert np.all(stats["max"][rot] <= 1.0 + 1e-5) and np.all(stats["min"][rot] >= -1.0 - 1e-5)
    assert not np.allclose(stats["mean"][rot], plain["mean"][rot])


# config plumbing


def test_pi05_config_validates_pose_groups():
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    with pytest.raises(ValueError, match="requires use_relative_actions"):
        PI05Config(relative_pose_groups=["right_ee"])
    with pytest.raises(ValueError, match="relative_pose_frame"):
        PI05Config(use_relative_actions=True, relative_pose_frame="base")
    cfg = PI05Config(use_relative_actions=True, relative_pose_groups=["right_ee", "left_ee"])
    assert cfg.relative_pose_frame == "ee" and cfg.state_feature_names is None


def test_factory_reads_state_and_grouped_names():
    from types import SimpleNamespace

    from lerobot.policies.factory import _dataset_feature_names

    meta = SimpleNamespace(
        features={
            ACTION: {"names": {"right": ACTION_NAMES[:10], "left": ACTION_NAMES[10:]}},
            "observation.state_raw": {"names": STATE_NAMES},
        }
    )
    assert _dataset_feature_names(meta, ACTION, None) == ACTION_NAMES
    assert _dataset_feature_names(meta, OBS_STATE, {"observation.state_raw": OBS_STATE}) == STATE_NAMES
    assert _dataset_feature_names(meta, OBS_STATE, None) is None


# pipelines built from checkpoints that predate relative actions (e.g. xvla-base)


def _bare_pipelines():
    from lerobot.processor import (
        AddBatchDimensionProcessorStep,
        DeviceProcessorStep,
        UnnormalizerProcessorStep,
        make_policy_processor_pipelines,
    )

    return make_policy_processor_pipelines(
        input_steps=[AddBatchDimensionProcessorStep(), DeviceProcessorStep(device="cpu")],
        output_steps=[UnnormalizerProcessorStep(features={}, norm_map={}), DeviceProcessorStep(device="cpu")],
    )


def _ee_step_config(**kwargs):
    return dict(
        exclude_joints=["gripper"],
        action_names=ACTION_NAMES,
        state_names=STATE_NAMES,
        pose_groups=["right_ee", "left_ee"],
        pose_frame="ee",
        **kwargs,
    )


def test_ensure_inserts_steps_where_factories_put_them():
    from lerobot.processor.relative_action_processor import ensure_relative_action_steps

    pre, post = _bare_pipelines()
    relative = ensure_relative_action_steps(pre, post, **_ee_step_config())
    assert [type(s).__name__ for s in pre.steps] == [
        "AddBatchDimensionProcessorStep",
        "RelativeActionsProcessorStep",
        "DeviceProcessorStep",
    ]
    assert [type(s).__name__ for s in post.steps] == [
        "UnnormalizerProcessorStep",
        "AbsoluteActionsProcessorStep",
        "DeviceProcessorStep",
    ]
    assert relative.enabled and post.steps[1].enabled and post.steps[1].relative_step is relative

    action, state = _random_ee_vectors((2, 50), seed=40), _random_ee_vectors((2,), seed=41)
    out = pre({ACTION: action, OBS_STATE: state})
    torch.testing.assert_close(out[ACTION], relative.to_relative(action, state))
    torch.testing.assert_close(post(out[ACTION]), action, atol=1e-5, rtol=0)


def test_ensure_configures_existing_steps_in_place():
    from lerobot.processor.relative_action_processor import ensure_relative_action_steps

    pre, post = _bare_pipelines()
    ensure_relative_action_steps(pre, post, **_ee_step_config())
    n_pre, n_post = len(pre.steps), len(post.steps)
    relative = ensure_relative_action_steps(pre, post, **{**_ee_step_config(), "pose_frame": "world"})
    assert (len(pre.steps), len(post.steps)) == (n_pre, n_post)
    assert relative.pose_frame == "world"
    with pytest.raises(ValueError, match="pose_frame"):
        ensure_relative_action_steps(pre, post, **{**_ee_step_config(), "pose_frame": "base"})


def test_xvla_config_validates_pose_groups():
    pytest.importorskip("transformers")
    from lerobot.policies.xvla.configuration_xvla import XVLAConfig

    with pytest.raises(ValueError, match="requires use_relative_actions"):
        XVLAConfig(relative_pose_groups=["right_ee"])
    with pytest.raises(ValueError, match="relative_pose_frame"):
        XVLAConfig(use_relative_actions=True, relative_pose_frame="base")
    cfg = XVLAConfig(use_relative_actions=True, relative_pose_groups=["right_ee", "left_ee"])
    assert cfg.relative_exclude_joints == ["gripper"] and cfg.state_feature_names is None


@pytest.mark.parametrize("enabled", [True, False])
def test_xvla_factory_places_relative_steps(enabled):
    pytest.importorskip("transformers")
    from lerobot.policies.xvla.configuration_xvla import XVLAConfig
    from lerobot.policies.xvla.processor_xvla import make_xvla_pre_post_processors

    cfg = XVLAConfig(
        use_relative_actions=enabled,
        relative_pose_groups=["right_ee", "left_ee"] if enabled else [],
        action_feature_names=ACTION_NAMES,
        state_feature_names=STATE_NAMES,
    )
    try:
        pre, post = make_xvla_pre_post_processors(cfg)
    except OSError as e:  # tokenizer not downloadable in this environment
        pytest.skip(f"xvla tokenizer unavailable: {e}")
    pre_names = [type(s).__name__ for s in pre.steps]
    post_names = [type(s).__name__ for s in post.steps]
    rel_idx = pre_names.index("RelativeActionsProcessorStep")
    assert pre_names[rel_idx - 1] == "AddBatchDimensionProcessorStep"
    assert rel_idx < pre_names.index("NormalizerProcessorStep")
    abs_idx = post_names.index("AbsoluteActionsProcessorStep")
    assert post_names[abs_idx - 1] == "UnnormalizerProcessorStep"
    assert pre.steps[rel_idx].enabled is enabled and post.steps[abs_idx].enabled is enabled
    assert post.steps[abs_idx].relative_step is pre.steps[rel_idx]


# xvla: wider state than xvla-base (latent controller dims) -> widened action_encoder


def test_widened_action_encoder_matches_the_original_for_any_new_dims():
    pytest.importorskip("transformers")
    from lerobot.policies.xvla.modeling_xvla import widen_action_encoder_weight
    from lerobot.policies.xvla.soft_transformer import DomainAwareLinear

    dim_action, dim_time, hidden, domains = 4, 2, 5, 3
    old = DomainAwareLinear(dim_action + 3 + dim_time, hidden, num_domains=domains)
    new = DomainAwareLinear(dim_action + 7 + dim_time, hidden, num_domains=domains)
    new.fc.weight.data = widen_action_encoder_weight(old.fc.weight.data, hidden, dim_action, dim_time, 7)
    new.bias.weight.data = old.bias.weight.data.clone()

    action, proprio, time = torch.randn(2, 6, dim_action), torch.randn(2, 6, 3), torch.randn(2, 6, dim_time)
    extra = torch.randn(2, 6, 4)  # the new state dims: must not change the output at initialisation
    domain = torch.tensor([0, 2])
    y_old = old(torch.cat([action, proprio, time], -1), domain)
    y_new = new(torch.cat([action, proprio, extra, time], -1), domain)
    torch.testing.assert_close(y_new, y_old)


def test_widen_action_encoder_refuses_to_shrink():
    pytest.importorskip("transformers")
    from lerobot.policies.xvla.modeling_xvla import widen_action_encoder_weight

    with pytest.raises(ValueError, match="shrink"):
        widen_action_encoder_weight(torch.zeros(1, (4 + 7 + 2) * 5), 5, 4, 2, 3)


# xvla: model state input restricted by name (proprio_state_names) + reset of pretrained proprio rows


def test_resolve_proprio_indices_by_name():
    pytest.importorskip("transformers")
    from lerobot.policies.xvla.modeling_xvla import resolve_proprio_indices

    names = ["right_ee_x", "right_gripper_finger", "right_cmd_dx", "left_cmd_dx"]
    assert resolve_proprio_indices(names, ["right_cmd_dx", "right_gripper_finger", "left_cmd_dx"]) == [2, 1, 3]
    with pytest.raises(ValueError, match="not in observation.state"):
        resolve_proprio_indices(names, ["left_ee_x"])
    with pytest.raises(ValueError, match="state_feature_names"):
        resolve_proprio_indices(None, ["right_cmd_dx"])


def test_zeroed_proprio_rows_make_the_encoder_ignore_the_state():
    pytest.importorskip("transformers")
    from lerobot.policies.xvla.modeling_xvla import zero_action_encoder_proprio_rows
    from lerobot.policies.xvla.soft_transformer import DomainAwareLinear

    dim_action, dim_proprio, dim_time, hidden = 4, 3, 2, 5
    enc = DomainAwareLinear(dim_action + dim_proprio + dim_time, hidden, num_domains=2)
    before = enc.fc.weight.data.clone()
    enc.fc.weight.data = zero_action_encoder_proprio_rows(enc.fc.weight.data, hidden, dim_action, dim_proprio)
    a, t, d = torch.randn(2, 6, dim_action), torch.randn(2, 6, dim_time), torch.tensor([0, 1])
    y1 = enc(torch.cat([a, torch.randn(2, 6, dim_proprio), t], -1), d)
    y2 = enc(torch.cat([a, torch.randn(2, 6, dim_proprio), t], -1), d)
    torch.testing.assert_close(y1, y2)  # the state no longer matters
    w_old = before.view(2, -1, hidden)
    w_new = enc.fc.weight.data.view(2, -1, hidden)
    torch.testing.assert_close(w_new[:, :dim_action], w_old[:, :dim_action])  # action + time rows untouched
    torch.testing.assert_close(w_new[:, dim_action + dim_proprio :], w_old[:, dim_action + dim_proprio :])


def test_xvla_config_rejects_more_proprio_names_than_max_state_dim():
    pytest.importorskip("transformers")
    from lerobot.policies.xvla.configuration_xvla import XVLAConfig

    with pytest.raises(ValueError, match="truncated"):
        XVLAConfig(max_state_dim=2, proprio_state_names=["a", "b", "c"])
    assert XVLAConfig(proprio_state_names=["a"]).reset_proprio_weights is False
