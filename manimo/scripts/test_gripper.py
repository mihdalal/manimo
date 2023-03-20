import faulthandler
import hydra
from omegaconf import OmegaConf
from manimo.environments.single_arm_env import SingleArmEnv
import time
import torch

# create a single arm environment

hydra.initialize(
        config_path="../conf", job_name="collect_demos_test"
    )
actuators_cfg = hydra.compose(config_name="actuators_gripper")
sensors_cfg = hydra.compose(config_name="sensors")
# sensors_cfg = []

faulthandler.enable()
env = SingleArmEnv(sensors_cfg, actuators_cfg)
obs = env.get_obs()

# close
for i in range(20):
    env.step([1])

# open 
for i in range(20):
    env.step([1])


env.reset()
# env.step([0])

for key in obs:
    print(f"obs key: {key}, values: {obs[key]}")