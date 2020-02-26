import asyncio
import sys
import time
import gym
import pygazebo
import os
import subprocess
import numpy as np
import numpy.linalg as alg
def main():

	env = gym.make('gym_px4:px4-v0')
	time.sleep(1)
	ob = env.reset()

	num_tries = 1
	KTp = 0.05
	KTd = 0.05

	KRp = 0.05
	KRd = 0.1

	KPp = 0.05
	KPd = 0.05

	KPRp = 0.05
	# KPRD = 0.05

	KRRp = 0.05
	# KRRd = 0.05
	rewards=0
	while True:

		dist = alg.norm(ob[0:3])
		
		thrust = 0.5645 #+ np.sign(ob[2])*KTp*dist - KTd*ob[5]

		# print ('err: ', ob[0:3])

		pitch = KRp*ob[0]#-KRd*ob[3]
		roll = KPp*ob[1]#-KPd*ob[4]

		# print('roll: ', roll, 'pitch: ', pitch)

		roll_rate = KRRp*roll
		pitch_rate = KPRp*pitch

		# action = [thrust,roll_rate,pitch_rate,0]
		action = np.array([0.6,0,0,0])
		

		# print('action: ', action)
		# print(info['step length'])

		ob, reward, done, info = env.step(action)
		rewards=rewards+reward
		# print(ob)
		
		# sys.stdout.write("\033[F")

		if done:
			print('try number: ', num_tries, 'total reward: ', rewards, 'reset reason: ', info['reset reason'])
			print('nest run')
			ob = env.reset()
			rewards=0
			num_tries+=1

if __name__ == "__main__":
	main()