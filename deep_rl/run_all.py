"""
run_all.py
==========
Stage 1 完成後自動接力 Stage 2 → Stage 3。
先等 stage1_loco.zip 出現，然後依序執行。
"""
import os, sys, time, subprocess, torch
sys.path.insert(0, os.path.dirname(__file__))

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
LOG_DIR   = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)

import numpy as np
from stable_baselines3 import TD3, SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList

from loco_env   import LocoEnv
from arm_env    import ArmEnv
from survey_env import SurveyEnv

GPU = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {GPU}  ({torch.cuda.get_device_name(0) if GPU=='cuda' else 'CPU'})")

# ── 等 Stage 1 完成 ──────────────────────────────────────────
s1_path = os.path.join(MODEL_DIR, "stage1_loco.zip")
if not os.path.exists(s1_path):
    print("Waiting for Stage 1 to finish (checking every 60s)...")
    while not os.path.exists(s1_path):
        time.sleep(60)
        size = os.path.getsize(s1_path) if os.path.exists(s1_path) else 0
        print(f"  [{time.strftime('%H:%M:%S')}] still waiting... ({size} bytes)")
    print(f"Stage 1 found: {s1_path}")
else:
    print(f"Stage 1 already done: {s1_path}")

time.sleep(3)

# ── Stage 2: Arm ─────────────────────────────────────────────
print()
print("="*60)
print(" Stage 2: Arm Reach  (SAC, 1M steps)")
print("="*60)

s2_path  = os.path.join(MODEL_DIR, "stage2_arm.zip")
s2_norm  = os.path.join(MODEL_DIR, "stage2_arm_norm.pkl")
s2_eval  = os.path.join(MODEL_DIR, "stage2_arm_eval")

arm_envs    = make_vec_env(ArmEnv, n_envs=4)
arm_eval    = make_vec_env(ArmEnv, n_envs=1)
arm_envs    = VecNormalize(arm_envs, norm_obs=True, norm_reward=True)
arm_eval    = VecNormalize(arm_eval, norm_obs=True, norm_reward=False)

arm_model = SAC(
    "MlpPolicy", arm_envs,
    learning_rate   = 3e-4,
    batch_size      = 512,
    buffer_size     = 300_000,
    learning_starts = 2_000,
    policy_kwargs   = dict(net_arch=[512, 512, 256]),
    device          = GPU,
    verbose         = 1,
    tensorboard_log = LOG_DIR,
)

arm_cb = CallbackList([
    EvalCallback(arm_eval, best_model_save_path=s2_eval,
                 eval_freq=10_000, n_eval_episodes=5, verbose=1),
    CheckpointCallback(save_freq=50_000, save_path=MODEL_DIR,
                       name_prefix="stage2_ckpt"),
])

t0 = time.time()
arm_model.learn(total_timesteps=1_000_000, callback=arm_cb,
                log_interval=50, progress_bar=True)
arm_model.save(s2_path)
arm_envs.save(s2_norm)
s2_elapsed = time.time() - t0
print(f"\nStage 2 done  {s2_elapsed/60:.1f} min  → {s2_path}")

# ── Stage 3: Full Survey ─────────────────────────────────────
print()
print("="*60)
print(" Stage 3: Full Survey  (TD3, 3M steps, transfer from Stage 1)")
print("="*60)

s3_path  = os.path.join(MODEL_DIR, "stage3_survey.zip")
s3_norm  = os.path.join(MODEL_DIR, "stage3_survey_norm.pkl")
s3_eval  = os.path.join(MODEL_DIR, "stage3_survey_eval")

srv_envs = make_vec_env(lambda: SurveyEnv(terrain_idx=-1), n_envs=4)
srv_eval = make_vec_env(lambda: SurveyEnv(terrain_idx=1),  n_envs=1)
srv_envs = VecNormalize(srv_envs, norm_obs=True, norm_reward=True)
srv_eval = VecNormalize(srv_eval, norm_obs=True, norm_reward=False)

n_act    = srv_envs.action_space.shape[-1]
srv_noise= NormalActionNoise(np.zeros(n_act), 0.15*np.ones(n_act))

# Transfer: load Stage 1 weights into larger action-space model
# (obs/act dims differ so we init fresh but use same hyperparams as Stage 1)
srv_model = TD3(
    "MlpPolicy", srv_envs,
    action_noise   = srv_noise,
    learning_rate  = 8e-5,      # slightly lower for fine-tuning
    batch_size     = 512,
    buffer_size    = 500_000,
    learning_starts= 8_000,
    train_freq     = (1, "step"),
    gradient_steps = 2,
    tau            = 0.005,
    gamma          = 0.99,
    policy_kwargs  = dict(net_arch=dict(pi=[512,512,256], qf=[512,512,256])),
    device         = GPU,
    verbose        = 1,
    tensorboard_log= LOG_DIR,
)

srv_cb = CallbackList([
    EvalCallback(srv_eval, best_model_save_path=s3_eval,
                 eval_freq=30_000, n_eval_episodes=5, verbose=1),
    CheckpointCallback(save_freq=200_000, save_path=MODEL_DIR,
                       name_prefix="stage3_ckpt"),
])

t0 = time.time()
srv_model.learn(total_timesteps=3_000_000, callback=srv_cb,
                log_interval=100, progress_bar=True)
srv_model.save(s3_path)
srv_envs.save(s3_norm)
s3_elapsed = time.time() - t0

# ── Summary ──────────────────────────────────────────────────
print()
print("="*60)
print(" ALL STAGES COMPLETE")
print("="*60)
print(f"  Stage 2 arm    : {s2_elapsed/60:.1f} min  → {s2_path}")
print(f"  Stage 3 survey : {s3_elapsed/60:.1f} min  → {s3_path}")
print()
print("Deploy:")
print("  ros2 run archimedes_survey rl_agent \\")
print(f"    --ros-args -p model_path:={s3_path} -p stage:=3")
print()
print("Evaluate:")
print("  python deep_rl/train.py --eval --stage 3")
print()
vram = torch.cuda.max_memory_allocated()//1024**2 if GPU=="cuda" else 0
print(f"  Peak VRAM used: {vram} MB")
