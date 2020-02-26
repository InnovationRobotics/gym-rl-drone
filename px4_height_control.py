#!/usr/bin/env python3

import os
from stable_baselines.sac.policies import MlpPolicy as sac_MlpPolicy
from stable_baselines.ddpg.policies import MlpPolicy as ddpg_MlpPolicy
from stable_baselines.common.policies import MlpPolicy as Common_MlpPolicy

from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import TRPO
from stable_baselines import DDPG
from stable_baselines import PPO1
from stable_baselines import SAC
from stable_baselines import logger
from os import system
import gym
import time
import numpy as np

n_steps=0
save_interval = 2000
best_mean_reward = -np.inf

def save_fn(_locals, _globals):
    global model, n_steps, best_mean_reward, best_model_path, last_model_path
    if (n_steps + 1) % save_interval == 0:
        
        # Evaluate policy training performance
        mean_reward = round(float(np.mean(_locals['episode_rewards'][-101:-1])), 1)
        print(n_steps + 1, 'timesteps')
        print("Best mean reward: {:.2f} - Last mean reward: {:.2f}".format(best_mean_reward, mean_reward))
        # New best model, save the agent
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            print("Saving new best model")
            model.save(best_model_path+'_rew_'+str(np.round(best_mean_reward,2)))
        model.save(last_model_path+'_'+str(time.localtime().tm_mday)+'_'+str(time.localtime().tm_hour)+'_'+str(time.localtime().tm_min))
    n_steps+=1
    pass

def main():
    global model, best_model_path, last_model_path
    env = gym.make('gym_px4:px4-v0').unwrapped
    train_model = True

    if train_model:

        for k in range(405,410):

            best_model_path = '/home/graphics/gym_px4_mavros_stable_bl_ffa/model_dir/sac/test_{}'.format(k)
            last_model_path = '/home/graphics/gym_px4_mavros_stable_bl_ffa/model_dir/sac/test_{}'.format(k)   
            
            num_timesteps = int(1e6)

            log_dir = '/home/graphics/gym_px4_mavros_stable_bl_ffa/log_dir/sac/test_{}'.format(k)

            policy_kwargs = dict(layers = [64,64,64])

            logger.configure(folder=log_dir, format_strs=['stdout', 'log', 'csv', 'tensorboard'])

            # model = SAC(sac_MlpPolicy, env, gamma=0.99, learning_rate=2e-4, buffer_size=500000,
            #      learning_starts=3000, train_freq=16, batch_size=64,
            #      tau=0.01, ent_coef='auto', target_update_interval=4,
            #      gradient_steps=4, target_entropy='auto', action_noise=None,
            #      random_exploration=0.0, verbose=2, tensorboard_log=None,
            #      _init_setup_model=True, policy_kwargs=policy_kwargs, full_tensorboard_log=False,
            #      seed=None, n_cpu_tf_sess=None)
            
            # new_objects=dict(learning_starts = 0)
            model = SAC.load('/home/graphics/gym_px4_mavros_stable_bl_ffa/model_dir/sac/test_405_rew_-281.4', env=env, custom_objects=dict(learning_starts = 0))
            
            model.learn(total_timesteps=num_timesteps, callback=save_fn)

            # model = PPO1(Common_MlpPolicy, env, gamma=0.99, timesteps_per_actorbatch=256, clip_param=0.2, entcoeff=0.01,
            #      optim_epochs=4, optim_stepsize=1e-3, optim_batchsize=64, lam=0.95, adam_epsilon=1e-5,
            #      schedule='linear', verbose=0, tensorboard_log=None, _init_setup_model=True,
            #      policy_kwargs=None, full_tensorboard_log=False, seed=None, n_cpu_tf_sess=1)


            # log_dir = '/home/graphics/gym_px4_installed_sb/log_dir/ddpg_p_{}'.format(str(k))
            # model = TRPO(MlpPolicy, env, timesteps_per_batch=4096, tensorboard_log=log_dir, verbose=1)
            # model.learn(total_timesteps=500000)
            # model.save("/home/graphics/gym_px4_installed_sb/model_dir/trpo_gym_px4_p_{}".format(str(k)))

            # action_noise = AdaptiveParamNoiseSpec()

            # model = PPO1(policy=MlpPolicy, env=env, gamma=0.99, timesteps_per_actorbatch=512, clip_param=0.2, entcoeff=0.01,  
            #      optim_epochs=32, optim_stepsize=1e-3, optim_batchsize=64, lam=0.95, adam_epsilon=1e-5,  
            #      schedule='linear', verbose=0, tensorboard_log=log_dir, _init_setup_model=True,  
            #      policy_kwargs=policy_kwargs, full_tensorboard_log=True, seed=k, n_cpu_tf_sess=None)


    else:
        env = gym.make('gym_px4:px4-v0')
        model = SAC.load('/home/graphics/gym_px4_mavros_stable_bl_ffa/model_dir/test_546', env=env, custom_objects=dict(learning_starts = 0))
            
        for _ in range(20):

            obs = env.reset()
            done = False
            while not done:
                action, _states =  model.predict(obs) #model.predict(np.array([0,0.0895]))#
                action[0]+=0.04
                obs, reward, done, info = env.step(action)
                # print('state: ', obs[0:3], 'action: ', action)


if __name__ == '__main__':
    main()
