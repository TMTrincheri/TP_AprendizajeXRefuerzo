import stable_baselines3
import torch
import gymnasium
from stable_baselines3 import PPO
import os
import time
from snakeenv import SnakeEnv

models_dir = f"models/PPO-{int(time.time())}"
logdir = f"logs/PPO-{int(time.time())}"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)


env = SnakeEnv()
env.reset()

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir, seed=40)

TIMESTEPS = 10000
for i in range(1,100000000):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS*i}")



env.close()