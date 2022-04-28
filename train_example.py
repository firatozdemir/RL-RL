## modification of rlym-tools/examples/sb3_multi_example.py based on youtube guide
import numpy as np
import os, glob

from pyparsing import Combine
from rlgym.envs import Match
from rlgym.utils.action_parsers import DiscreteAction
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan
from stable_baselines3.ppo import MlpPolicy

from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from rlgym.utils.reward_functions.common_rewards.misc_rewards import EventReward
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import VelocityPlayerToBallReward
from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import VelocityBallToGoalReward
from rlgym.utils.reward_functions import CombinedReward

import torch.nn as nn

def find_last_ckpt(save_dir):
    fl = glob.glob(os.path.join(save_dir, '*'))
    if len(fl) == 0:
        print('No previously saveed model found.')
        return None
    ckpts = [int(str(fname).split("_")[-2]) for fname in fl]
    ckptname = fl[np.argmax(ckpts)]
    return ckptname

if __name__ == '__main__':  # Required for multiprocessing
    frame_skip = 8          # Number of ticks to repeat an action
    half_life_seconds = 5   # Easier to conceptualize, after this many seconds the reward discount is 0.5

    fps = 120 / frame_skip
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))  # Quick mafs

    target_steps = 100_000
    agents_per_match = 2 # for 1vs1
    num_instances = 4 # #RL instances to run at the same time
    steps = target_steps // (num_instances * agents_per_match)
    batch_size = steps
    wait_time= 1*60 # time to wait between each individual RL instance
    save_dir = "./models"
    model_name= "rl_model"
    ckpt_freq = 500_000
    print(f"fps={fps}, gamma={gamma})")

    reward_function = CombinedReward(
        reward_functions=(
            VelocityPlayerToBallReward(),
            VelocityBallToGoalReward(),
            EventReward(
                team_goal=100., 
                concede=-100., #getting scored on
                shot=5., 
                save=30., 
                demo=10.0),
        ),
        reward_weights=(0.1, 1., 1.),
    )


    def get_match():  # Need to use a function so that each instance can call it and produce their own objects
        return Match(
            # team_size=3,  # 3v3 to get as many agents going as possible, will make results more noisy
            team_size=1, #1vs1
            tick_skip=frame_skip,
            reward_function=reward_function,
            self_play=True,
            terminal_conditions=[TimeoutCondition(round(fps * 30)), GoalScoredCondition()],  # Some basic terminals
            obs_builder=AdvancedObs(),  # Not that advanced, good default
            state_setter=DefaultState(),  # Resets to kickoff position
            action_parser=DiscreteAction()  # Discrete > Continuous don't @ me
        )

    env = SB3MultipleInstanceEnv(get_match, num_instances, wait_time=wait_time)# Start num_instances instances, waiting 60 seconds between each
    env = VecCheckNan(env)                                # Optional
    env = VecMonitor(env)                                 # Recommended, logs mean reward and ep_len to Tensorboard
    env = VecNormalize(env, norm_obs=False, gamma=gamma)  # Highly recommended, normalizes rewards

    ckptname = find_last_ckpt(save_dir)
    if ckptname is not None:
        print('Ckpt %s is found. Continuing training' % ckptname)
        model = PPO.load(
        ckptname,
        env,
        device="auto",  # Need to set device again (if using a specific one)
        # force_reset=True  # Make SB3 reset the env so it doesn't think we're continuing from last state
    )
    else: #start from scratch 
        policy_kwargs = dict(
            activation_fn=nn.Tanh,
            net_arch=[512, 512, dict(
                pi=[256, 256, 256], # controls actions of the model
                vf=[256, 256, 256] # value function 
                )],
        )

        model = PPO(
            MlpPolicy,
            env,
            n_epochs=1,                 # PPO calls for multiple epochs
            policy_kwargs=policy_kwargs, #
            learning_rate=5e-5,          # Around this is fairly common for PPO
            ent_coef=0.01,               # From PPO Atari
            vf_coef=1.,                  # From PPO Atari
            gamma=gamma,                 # Gamma as calculated using half-life
            verbose=3,                   # Print out all the info as we're going
            batch_size=batch_size,       # Batch size as high as possible within reason
            n_steps=steps,               # Number of steps to perform before optimizing network
            tensorboard_log="logs",      # `tensorboard --logdir logs` in terminal to see graphs
            device="auto"                # Uses GPU if available
        )

    # Save model every so often
    # Divide by num_envs (number of agents) because callback only increments every time all agents have taken a step
    # This saves to specified folder with a specified name
    callback = CheckpointCallback(round(ckpt_freq / env.num_envs), save_path=save_dir, name_prefix=model_name)

    while True:
        model.learn(total_timesteps=25_000_000, callback=callback)
        model.save(save_dir+"/exit_save.zip")
        model.save(f"mmr_models/{model.num_timesteps}")
        