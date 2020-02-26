#!/usr/bin/env python3

import os
from stable_baselines.sac.policies import MlpPolicy as sac_MlpPolicy
from stable_baselines.ddpg.policies import MlpPolicy as ddpg_MlpPolicy
from stable_baselines.common.policies import MlpPolicy as Common_MlpPolicy
from stable_baselines.common.noise import OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec, NormalActionNoise
from stable_baselines.common import make_vec_env

from stable_baselines import TRPO
from stable_baselines import DDPG
from stable_baselines import PPO1
from stable_baselines import PPO2
from stable_baselines import SAC
from stable_baselines import logger

import matplotlib.pyplot as plt
import time


from os import system
import gym
import time
import numpy as np
import gym_reinmav

n_steps=0
save_interval = 25000
best_mean_reward = -np.inf

def save_fn(_locals, _globals):
    global model, n_steps, best_mean_reward, best_model_path, last_model_path
    if (n_steps + 1) % save_interval == 0:
        
        # Evaluate policy training performance
       
        # reward_buffer=np.array(_locals['epoch_episode_rewards'])  ## for ddpg
        mean_reward = round(float(np.mean(_locals['episode_rewards'][-101:-1])), 1)
        print(n_steps + 1, 'timesteps')
        print("Best mean reward: {:.2f} - Last mean reward: {:.2f}".format(best_mean_reward, mean_reward))

        # New best model, save the agent
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            print("Saving new best model")
            model.save(best_model_path+'_rew_'+str(best_mean_reward))
        else:
            model.save(last_model_path)
    n_steps+=1
    pass

def main():
    global model, best_model_path, last_model_path  
    train_model = True
    if train_model:
        for k in range(606,610,1):
###############################################################################################################
            best_model_path = '/home/graphics/reinmav-gym/model_dir/sac/test_{}'.format(k)
            last_model_path = '/home/graphics/reinmav-gym/model_dir/sac/test_{}'.format(k)   

            policy_kwargs = dict(layers = [64,64,64])
            num_timesteps = int(1e7)

            log_dir = '/home/graphics/reinmav-gym/log_dir/sac/test_{}'.format(k)
            env = gym.make('quadrotor3d-v0').unwrapped
            logger.configure(folder=log_dir, format_strs=['stdout', 'log', 'csv', 'tensorboard'])
            # model = SAC(sac_MlpPolicy, env, gamma=0.99, learning_rate=1e-4, buffer_size=500000,
            #      learning_starts=10000, train_freq=16, batch_size=64,
            #      tau=0.01, ent_coef='auto', target_update_interval=4,
            #      gradient_steps=4, target_entropy='auto', action_noise=None,
            #      random_exploration=0.0, verbose=2, tensorboard_log=log_dir,
            #      _init_setup_model=True, policy_kwargs=policy_kwargs, full_tensorboard_log=False,
            #      seed=k, n_cpu_tf_sess=None)
            model = SAC.load('/home/graphics/reinmav-gym/model_dir/sac/test_605', env=env, custom_objects=dict(learning_starts = 0))

            model.learn(total_timesteps=num_timesteps, log_interval=1, callback=save_fn, eval_freq=50000)
###############################################################################################################
            # best_model_path = '/home/graphics/reinmav-gym/model_dir/ppo2/test_{}'.format(k)
            # last_model_path = '/home/graphics/reinmav-gym/model_dir/ppo2/test_{}'.format(k)   

            # policy_kwargs = dict(layers = [64,64,64])
            # num_timesteps = int(1e7)

            # env = make_vec_env('quadrotor3d-v0', n_envs=4)
            # log_dir = '/home/graphics/reinmav-gym/log_dir/ppo2/test_{}'.format(k)
            # logger.configure(folder=log_dir, format_strs=['stdout', 'log', 'csv', 'tensorboard'])
            # model = PPO2(Common_MlpPolicy, env, gamma=0.99, n_steps=128, ent_coef=0.00, learning_rate=2e-4, vf_coef=0.5,
            #      max_grad_norm=0.5, lam=0.95, nminibatches=4, noptepochs=4, cliprange=0.2, cliprange_vf=None,
            #      verbose=2, tensorboard_log=None, _init_setup_model=True, policy_kwargs=policy_kwargs,
            #      full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None)

            # model.learn(total_timesteps=num_timesteps, log_interval=1, callback=save_fn)
###############################################################################################################            
            # env = gym.make('quadrotor3d-v0')
            # best_model_path = '/home/graphics/reinmav-gym/model_dir/ddpg/test_{}'.format(k)
            # last_model_path = '/home/graphics/reinmav-gym/model_dir/ddpg/test_{}'.format(k)   
            # log_dir = '/home/graphics/reinmav-gym/log_dir/ddpg/test_{}'.format(k)
            # model = DDPG(ddpg_MlpPolicy, env, gamma=0.99, eval_env=env, nb_train_steps=50,
            #      nb_rollout_steps=100000, nb_eval_steps=500, param_noise=None, action_noise=None,
            #      tau=0.001, batch_size=128, param_noise_adaption_interval=50,
            #      observation_range=(env.observation_space.low, env.observation_space.high),
            #      actor_lr=1e-4, critic_lr=1e-3,
            #      render=False, render_eval=True, memory_limit=None, buffer_size=500000, random_exploration=0.0,
            #      verbose=3, tensorboard_log=log_dir, _init_setup_model=True, policy_kwargs=None,
            #      full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None)
            # model.learn(total_timesteps=num_timesteps, log_interval=1, callback=save_fn)

            # model = SAC.load('/home/graphics/reinmav-gym/model_dir/test_1/sac_simple/sac_simple_10001', env=env)
            
            
            # env = gym.make('quadrotor3d-v0')
            # best_model_path = '/home/graphics/reinmav-gym/model_dir/ppo1/test_{}'.format(k)
            # last_model_path = '/home/graphics/reinmav-gym/model_dir/ppo1/test_{}'.format(k)   
            # num_timesteps = int(5e6)
            # log_dir = '/home/graphics/reinmav-gym/log_dir/ppo1/test_{}'.format(k)
            # model = PPO1(policy=Common_MlpPolicy, env=env, gamma=0.97, timesteps_per_actorbatch=512, clip_param=0.2, entcoeff=0,  
            #      optim_epochs=32, optim_stepsize=1e-3, optim_batchsize=128, lam=0.95, adam_epsilon=1e-5,  
            #      schedule='linear', verbose=1, tensorboard_log=log_dir, _init_setup_model=True,  
            #      full_tensorboard_log=False, seed=k, n_cpu_tf_sess=None)
            
            # model.learn(total_timesteps=num_timesteps, log_interval=1, callback=save_fn)


            # env = lambda : gym.make('quadrotor3d-v0')

            # logger_kwargs = dict(output_dir = '/home/graphics/reinmav-gym/log_dir/sac/test_{}'.format(k))
            # ac_kwargs = dict(hidden_sizes=[256,256,16])

            # steps_per_epoch = 512

            # ppo(env, actor_critic=ppo_core.mlp_actor_critic, ac_kwargs=ac_kwargs, seed=3*k+1, \
            # steps_per_epoch=steps_per_epoch, epochs=5000, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4, \
            # vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=5000, \
            # target_kl=0.01, logger_kwargs=logger_kwargs, save_freq=10)


    else:

        env = gym.make('quadrotor3d-v0') 
        model = SAC.load('/home/graphics/reinmav-gym/model_dir/sac/test_604', env=env)

        for _ in range(20):

            ob = env.reset()
            obs = []
            env.render()
            done = False
            while not done:
                obs.append(ob[0:3])
                action, _states =  model.predict(ob) #model.predict(np.array([0,0.0895]))#
                env.render()
                ob, reward, done, info = env.step(action)
                print(ob)
                if np.abs(ob[10:13]).any() > 1.57:
                    print('bigger')
                if np.abs(ob[13:16]).any() > 10:
                    print('bigger')


                
                # ob[0:2] = ob[0:2] + 1
                # print('pos: ', ob[0:3])
                # time.sleep(0.015)
            # obs = np.array(obs)
            # obs[:,2] = 3+obs[:,2]
            # plot_results(np.array(obs))

def plot_results(obs):
	plt.plot(obs)
	plt.xlabel('step')
	plt.ylabel('distance [m]')
	plt.grid(True)
	plt.show()

if __name__ == '__main__':
    main()
