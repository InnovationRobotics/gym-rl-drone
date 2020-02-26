import asyncio
import sys
import time
import gym
import pygazebo
import os
import subprocess
import gym_reinmav

def main():

	# thrust = action[0] # Thrust command
	# w = action[1:4] # Angular velocity command

	env = gym.make('quadrotor3d-v0')
	ob = env.reset()


	num_tries = 1
	Kp = 0.05
	Kd = 0.05
	rewards=0
	while True:
		env.render()
		action = [15.1,3,-3,0]

		# ob, reward, done, info = env.step(action)
		ob, reward, done, info = env.step(action)
		rewards=rewards+reward
		
		

		# if done:
		# 	print('try number: ', num_tries, 'total reward: ', rewards, 'reset reason: ')
		# 	print('nest run')
		# 	ob = env.reset()
		# 	rewards=0
		# 	num_tries+=1

if __name__ == "__main__":
	main()