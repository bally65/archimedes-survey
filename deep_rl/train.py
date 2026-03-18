"""
train.py
========
Master curriculum training script for Archimedes Survey robot.

Training pipeline (3 stages):
  Stage 1: Train locomotion only (loco_env.py) — fast convergence
  Stage 2: Train arm reach only  (arm_env.py)  — precise control
  Stage 3: Fine-tune full system (survey_env.py) — transfer learning from Stage 1+2

Based on bally65/Archimedes-Hand-/5. Deep_LR/train_whole_body.py
Using TD3 + VecNormalize (same as bally65 approach).

Run:
  # Train all stages sequentially:
  python deep_rl/train.py

  # Train only one stage:
  python deep_rl/train.py --stage 1
  python deep_rl/train.py --stage 2
  python deep_rl/train.py --stage 3

  # Resume from checkpoint:
  python deep_rl/train.py --stage 3 --resume

  # With pretrained bally65 weights (transfer learning):
  python deep_rl/train.py --stage 3 --pretrained models/pretrained/td3_terrain_aware_final.zip

  # Visualise trained policy:
  python deep_rl/train.py --eval --stage 1
"""

import argparse
import os
import sys
import numpy as np

try:
    from stable_baselines3 import TD3, SAC
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize
    from stable_baselines3.common.callbacks import (
        EvalCallback, CheckpointCallback, CallbackList
    )
except ImportError:
    sys.exit("pip install stable-baselines3[extra]")

from loco_env  import LocoEnv
from arm_env   import ArmEnv
from survey_env import SurveyEnv

# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
LOG_DIR   = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)

# ---------------------------------------------------------------------------
# Shared TD3 hyperparameters (matching bally65 whole_body settings)
# ---------------------------------------------------------------------------
TD3_KWARGS = dict(
    learning_rate   = 1e-4,
    batch_size      = 256,
    buffer_size     = 1_000_000,
    learning_starts = 5_000,
    train_freq      = 1,
    gradient_steps  = 1,
    tau             = 0.005,
    gamma           = 0.99,
    policy_kwargs   = dict(net_arch=dict(pi=[512, 512, 256],
                                         qf=[512, 512, 256])),
    verbose         = 1,
    device          = "auto",
)


def _make_noise(env):
    n = env.action_space.shape[-1]
    return NormalActionNoise(np.zeros(n), 0.15 * np.ones(n))


# ---------------------------------------------------------------------------
# Stage 1: Locomotion
# ---------------------------------------------------------------------------
def train_locomotion(n_envs=4, total_steps=2_000_000, resume=False):
    tag = "stage1_loco"
    model_path  = os.path.join(MODEL_DIR, tag)
    vec_path    = os.path.join(MODEL_DIR, f"{tag}_norm.pkl")
    eval_path   = os.path.join(MODEL_DIR, f"{tag}_eval")

    print(f"\n{'='*60}")
    print(f" Stage 1: Locomotion Training  ({total_steps:,} steps)")
    print(f"{'='*60}\n")

    # Curriculum: start easy (wet sand), then random
    def make_env(i):
        terrain = 1 if i < n_envs // 2 else -1   # half fixed, half random
        return lambda: LocoEnv(terrain_idx=terrain)

    envs = make_vec_env(make_env(0), n_envs=n_envs)
    eval_env = make_vec_env(lambda: LocoEnv(terrain_idx=-1), n_envs=1)

    if os.path.exists(vec_path):
        envs     = VecNormalize.load(vec_path, envs)
        eval_env = VecNormalize.load(vec_path, eval_env)
        eval_env.training = False
        print(f"  Loaded normalisation: {vec_path}")
    else:
        envs     = VecNormalize(envs, norm_obs=True, norm_reward=True)
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)

    if resume and os.path.exists(model_path + ".zip"):
        print(f"  Resuming from: {model_path}.zip")
        model = TD3.load(model_path, env=envs, **{
            k: v for k, v in TD3_KWARGS.items()
            if k not in ("verbose", "device", "policy_kwargs")})
    else:
        model = TD3("MlpPolicy", envs,
                    action_noise=_make_noise(envs),
                    **TD3_KWARGS)

    callbacks = CallbackList([
        EvalCallback(eval_env, best_model_save_path=eval_path,
                     eval_freq=20_000 // n_envs, n_eval_episodes=5,
                     verbose=0),
        CheckpointCallback(save_freq=50_000 // n_envs,
                           save_path=MODEL_DIR,
                           name_prefix=tag),
    ])

    model.learn(total_timesteps=total_steps,
                callback=callbacks,
                reset_num_timesteps=not resume,
                log_interval=100)

    model.save(model_path)
    envs.save(vec_path)
    print(f"\n  Stage 1 complete → {model_path}.zip")
    return model_path, vec_path


# ---------------------------------------------------------------------------
# Stage 2: Arm reach
# ---------------------------------------------------------------------------
def train_arm(total_steps=1_000_000, resume=False):
    tag = "stage2_arm"
    model_path = os.path.join(MODEL_DIR, tag)
    vec_path   = os.path.join(MODEL_DIR, f"{tag}_norm.pkl")
    eval_path  = os.path.join(MODEL_DIR, f"{tag}_eval")

    print(f"\n{'='*60}")
    print(f" Stage 2: Arm Reach Training  ({total_steps:,} steps)")
    print(f"{'='*60}\n")

    envs     = make_vec_env(ArmEnv, n_envs=2)
    eval_env = make_vec_env(ArmEnv, n_envs=1)

    if os.path.exists(vec_path):
        envs     = VecNormalize.load(vec_path, envs)
        eval_env = VecNormalize.load(vec_path, eval_env)
        eval_env.training = False
    else:
        envs     = VecNormalize(envs, norm_obs=True, norm_reward=True)
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)

    if resume and os.path.exists(model_path + ".zip"):
        model = TD3.load(model_path, env=envs, **{
            k: v for k, v in TD3_KWARGS.items()
            if k not in ("verbose", "device", "policy_kwargs")})
    else:
        # Use SAC for arm (better sample efficiency for precise control)
        model = SAC("MlpPolicy", envs,
                    learning_rate   = 3e-4,
                    batch_size      = 256,
                    buffer_size     = 500_000,
                    learning_starts = 2_000,
                    policy_kwargs   = dict(net_arch=[512, 512, 256]),
                    verbose         = 1,
                    device          = "auto")

    callbacks = CallbackList([
        EvalCallback(eval_env, best_model_save_path=eval_path,
                     eval_freq=10_000, n_eval_episodes=5, verbose=0),
        CheckpointCallback(save_freq=25_000, save_path=MODEL_DIR,
                           name_prefix=tag),
    ])

    model.learn(total_timesteps=total_steps,
                callback=callbacks,
                reset_num_timesteps=not resume,
                log_interval=50)

    model.save(model_path)
    envs.save(vec_path)
    print(f"\n  Stage 2 complete → {model_path}.zip")
    return model_path, vec_path


# ---------------------------------------------------------------------------
# Stage 3: Full survey (transfer learning from Stage 1)
# ---------------------------------------------------------------------------
def train_survey(loco_model_path=None, pretrained_path=None,
                 total_steps=3_000_000, resume=False):
    tag = "stage3_survey"
    model_path = os.path.join(MODEL_DIR, tag)
    vec_path   = os.path.join(MODEL_DIR, f"{tag}_norm.pkl")
    eval_path  = os.path.join(MODEL_DIR, f"{tag}_eval")

    print(f"\n{'='*60}")
    print(f" Stage 3: Full Survey Training  ({total_steps:,} steps)")
    print(f"{'='*60}\n")

    envs     = make_vec_env(SurveyEnv, n_envs=4)
    eval_env = make_vec_env(lambda: SurveyEnv(terrain_idx=1), n_envs=1)

    if os.path.exists(vec_path):
        envs     = VecNormalize.load(vec_path, envs)
        eval_env = VecNormalize.load(vec_path, eval_env)
        eval_env.training = False
    else:
        envs     = VecNormalize(envs, norm_obs=True, norm_reward=True)
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)

    if resume and os.path.exists(model_path + ".zip"):
        print(f"  Resuming: {model_path}.zip")
        model = TD3.load(model_path, env=envs, **{
            k: v for k, v in TD3_KWARGS.items()
            if k not in ("verbose", "device", "policy_kwargs")})
    elif pretrained_path and os.path.exists(pretrained_path):
        # Transfer from bally65's whole_body_v3 or our loco model
        print(f"  Transfer from: {pretrained_path}")
        model = TD3.load(pretrained_path, env=envs,
                         custom_objects={"action_space": envs.action_space,
                                         "observation_space": envs.observation_space})
    else:
        model = TD3("MlpPolicy", envs,
                    action_noise=_make_noise(envs),
                    **TD3_KWARGS)

    callbacks = CallbackList([
        EvalCallback(eval_env, best_model_save_path=eval_path,
                     eval_freq=30_000 // 4, n_eval_episodes=5, verbose=0),
        CheckpointCallback(save_freq=100_000 // 4,
                           save_path=MODEL_DIR,
                           name_prefix=tag),
    ])

    model.learn(total_timesteps=total_steps,
                callback=callbacks,
                reset_num_timesteps=not resume,
                log_interval=200)

    model.save(model_path)
    envs.save(vec_path)
    print(f"\n  Stage 3 complete → {model_path}.zip")
    return model_path, vec_path


# ---------------------------------------------------------------------------
# Evaluation / playback
# ---------------------------------------------------------------------------
def evaluate(stage: int, n_episodes: int = 5):
    stage_map = {
        1: ("stage1_loco",    LocoEnv,    "stage1_loco_norm.pkl"),
        2: ("stage2_arm",     ArmEnv,     "stage2_arm_norm.pkl"),
        3: ("stage3_survey",  SurveyEnv,  "stage3_survey_norm.pkl"),
    }
    tag, EnvClass, norm_tag = stage_map[stage]
    model_path = os.path.join(MODEL_DIR, tag + ".zip")
    vec_path   = os.path.join(MODEL_DIR, norm_tag)

    if not os.path.exists(model_path):
        print(f"No model found at {model_path}. Train first.")
        return

    env = EnvClass(render_mode="human")
    if os.path.exists(vec_path):
        from stable_baselines3.common.vec_env import DummyVecEnv
        venv = DummyVecEnv([lambda: env])
        venv = VecNormalize.load(vec_path, venv)
        venv.training = False

        # Try both TD3 and SAC
        try:
            model = TD3.load(model_path, env=venv)
        except Exception:
            model = SAC.load(model_path, env=venv)

        for ep in range(n_episodes):
            obs = venv.reset()
            done = False
            total_rew = 0.0
            steps = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, rew, done, info = venv.step(action)
                total_rew += float(rew[0])
                steps += 1
            print(f"  Episode {ep+1}: reward={total_rew:.1f}  steps={steps}")
        venv.close()
    else:
        print(f"Normalisation file not found: {vec_path}")


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Archimedes Survey RL Training")
    p.add_argument("--stage",      type=int, default=0,
                   help="Stage to run (1=loco, 2=arm, 3=survey, 0=all)")
    p.add_argument("--resume",     action="store_true")
    p.add_argument("--eval",       action="store_true")
    p.add_argument("--pretrained", type=str, default="",
                   help="Path to pretrained model (for Stage 3 transfer)")
    p.add_argument("--steps",      type=int, default=0,
                   help="Override total training steps")
    args = p.parse_args()

    if args.eval:
        stage = args.stage if args.stage > 0 else 3
        evaluate(stage)
        return

    loco_model = None

    if args.stage in (0, 1):
        n = args.steps or 2_000_000
        loco_model, _ = train_locomotion(total_steps=n, resume=args.resume)

    if args.stage in (0, 2):
        n = args.steps or 1_000_000
        train_arm(total_steps=n, resume=args.resume)

    if args.stage in (0, 3):
        n = args.steps or 3_000_000
        pretrained = args.pretrained or loco_model or ""
        train_survey(loco_model_path=loco_model,
                     pretrained_path=pretrained,
                     total_steps=n,
                     resume=args.resume)

    print("\n=== All training complete ===")
    print("  Deploy with: ros2 run archimedes_survey rl_agent")


if __name__ == "__main__":
    main()
