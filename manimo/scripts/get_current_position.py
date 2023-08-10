import faulthandler

import hydra
from manimo.environments.single_arm_env import SingleArmEnv
import numpy as np
import cv2
# create a single arm environment

def crop_and_resize_fixed_view(img, output_size=(84, 84)):
    out = cv2.resize(img[175:, 150:-100], output_size)
    return out

def crop_and_resize_wrist_view(img, output_size=(84, 84)):
    out = cv2.resize(img, output_size)
    return out


def vis_depth(cv_depth):
    invalid_inds = cv_depth == 0
    min_val = np.min(cv_depth)
    max_val = np.max(cv_depth)
    depth_range = max_val - min_val
    depth8 = (255.0 / depth_range * (cv_depth - min_val)).astype("uint8")
    depth8_rgb = cv2.cvtColor(depth8, cv2.COLOR_GRAY2RGB)
    depth_color = cv2.applyColorMap(depth8_rgb, cv2.COLORMAP_JET)
    depth_color[invalid_inds] = [255, 255, 255]
    return depth_color

hydra.initialize(config_path="../conf", job_name="collect_demos_test")
env_cfg = hydra.compose(config_name="env")
actuators_cfg = hydra.compose(config_name="actuators")
sensors_cfg = hydra.compose(config_name="sensors")

faulthandler.enable()
env = SingleArmEnv(sensors_cfg, actuators_cfg, env_cfg)
obs = env.get_obs()
print(obs['eef_pos'])
depth = obs['cam1c_depth'][0]
cv_depth = vis_depth(depth)
cv2.imwrite('depth.png', cv_depth)
cv2.imwrite('rgb.png', obs['cam1c'][0][:, :, ::-1])
img = obs['cam1c'][0][:, :, ::-1]
combined = cv2.addWeighted(img, 0.5, cv_depth, 0.5, 0)
cv2.imwrite('combined.png', combined)