#!/usr/bin/env python3
import gym
from gym import spaces

import numpy as np
import asyncio
import math

from mavsdk import System
from mavsdk import (OffboardError, PositionNedYaw, ActuatorControl, AttitudeRate)

import subprocess
import time
import sys

import pygazebo
from pygazebo import Manager
from pygazebo.msg import world_control_pb2
from pygazebo.msg.world_control_pb2 import WorldControl

def lat_lon_to_coords(cords):
    radius_of_earth = 6371000
    rad_lat = math.radians(cords[0])
    rad_lon = math.radians(cords[1])
    x = radius_of_earth * math.cos(rad_lat) * math.cos(rad_lon)
    y = radius_of_earth * math.cos(rad_lat) * math.sin(rad_lon)
    z = cords[2]
    return x, y, z



# async def reset_async(reset_pos):

#     # unpause_msg = WorldControl()
#     # unpause_msg.pause = False
#     # await asyncio.sleep(0.001)
#     # await pub_world_control.publish(unpause_msg)



#     print('-- Resetting position')

#     await drone.offboard.set_position_ned(PositionNedYaw(reset_pos[0],reset_pos[1],-reset_pos[2],reset_pos[3]))
#     while True:
#         lin_pos = await get_lin_pos()
#         lin_vel = await get_lin_vel()
#         await asyncio.sleep(0.1)
#         if np.abs(np.linalg.norm(lin_pos - reset_pos[0:3])) < 0.7 and np.abs(np.linalg.norm(lin_vel)) < 0.7 :
#             await asyncio.sleep(1)
#             break
#     print('-- Position reset')

#     # pause_msg = WorldControl()
#     # pause_msg.pause = True
#     # await asyncio.sleep(0.001)
#     # await pub_world_control.publish(pause_msg)
#     return lin_pos, lin_vel

class gymPX4(gym.Env):

    

    def __init__(self):

        self.loop = asyncio.get_event_loop()

        self.loop.run_until_complete(self.init_env())

        self.observation_space = spaces.Box( np.array([1,-2.0]), np.array([20.0,2.0]), dtype=np.float32)
        self.action_space = spaces.Box(0, 1, shape=(1,), dtype=np.float32)

        # self.unpause_msg = WorldControl()
        # self.unpause_msg.pause = False
        # self.pause_msg = WorldControl()
        # self.pause_msg.pause = True
        # self.step_msg = WorldControl()
        # self.step_msg.step = True

    async def init_env(self):

    ## connect to mavsdk ##
        self.drone = System()
        await self.drone.connect(system_address="udp://:14550")   ## connect to mavsdk
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                break
        await asyncio.sleep(1)

    ## connect to gazebo ##
        print('-- Connecting to Gazebo')
        try:
            self.manager = await pygazebo.connect(('localhost', 11345))   ## connect to pygazebo
            await asyncio.sleep(0.5)
            print('-- Connected')
        except:
            print('unable to connect')

    ## prepare gazebo world control message ##  
        self.pub_world_control = await self.manager.advertise('/gazebo/default/world_control','gazebo.msgs.WorldControl')
        await asyncio.sleep(1)
    
    ## get home position ##
        async for home_pos in self.drone.telemetry.home():  ## get absolute home position
            glob_home_pos = np.array([home_pos.latitude_deg, home_pos.longitude_deg, home_pos.absolute_altitude_m])
            self.home_pos = np.array(lat_lon_to_coords(glob_home_pos))
            break

        # asyncio.ensure_future(get_lin_pos())  ## initiate linear position stream
        # asyncio.ensure_future(get_lin_vel())  ## initiate linear velocity stream
    
    ## arm quadcopter ##
        async for is_armed in self.drone.telemetry.armed():  ## check arm status
            if not is_armed:  ## if not armed, arm and change to OFFBOARD mode
                await self.arm_offb()
                break

    async def arm_offb(self):
        print("-- Arming")
        await self.drone.action.arm()
        await self.drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, 0.0, 0.0))
        print("-- Starting offboard")
        try:
            await self.drone.offboard.start()
        except OffboardError as error:
            print(f"Starting offboard mode failed with error code: {error._result.result}")
            print("-- Disarming")
            await self.drone.action.disarm()

    def reset(self):

        self.steps=0

        self.desired = np.random.randint(4,20)
        self.initial = np.random.randint(2,20)

        print('Initial: ', self.initial, 'Desired: ', self.desired)

        reset_pos=[0,0,self.initial,0]
        lin_pos, lin_vel = self.loop.run_until_complete(self.reset_async(reset_pos))

        observation = [self.desired - lin_pos[2]]
        return observation  # reward, done, info can't be included

    async def reset_async(self,reset_pos):

        # await pub_world_control.publish(unpause_msg)
        reset_steps = 0

        print('-- Resetting position')
        await self.drone.offboard.set_position_ned(PositionNedYaw(reset_pos[0],reset_pos[1],-reset_pos[2],reset_pos[3]))
        
        while True:
            
            async for is_armed in self.drone.telemetry.armed():  ## check arm status
                if not is_armed:  ## if not armed, arm and change to OFFBOARD mode
                    await self.arm_offb()
                    break

            lin_pos = await self.get_lin_pos()
            lin_vel = await self.get_lin_vel()

            if np.abs(np.linalg.norm(lin_pos[2] - reset_pos[2])) < 0.7 and np.abs(np.linalg.norm(lin_vel)) < 0.2 :   ### wait for drone to reach desired position
                await asyncio.sleep(0.2)
                break

            if (reset_steps+1) % 50 == 0:      ### if reset takes too long, reset simulation
                print('Reset failed, restarting simulation')
                subprocess.Popen(args='make px4_sitl gazebo', cwd='../Firmware', shell=True)
                reset_steps = 0
                await asyncio.sleep(10)
                self.loop.run_until_complete(self.init_env())
                await asyncio.sleep(0.5)
            
            print('Resetting position: ', reset_steps, '/50')
            sys.stdout.write("\033[F")

        print('-- Position reset')

        # await pub_world_control.publish(pause_msg)

        return lin_pos, lin_vel

    def step(self, action):

        lin_pos, lin_vel = self.loop.run_until_complete(self.step_async(action))
        
        reward = -np.power( self.desired - lin_pos[2], 2)
        ob = [ self.desired - lin_pos[2] ] 
        
        done = False
        reset = 'No'

        if  np.abs(lin_pos[0]) > 5 or np.abs(lin_pos[1]) > 5 or np.abs(lin_pos[2]) > 8 or np.abs(lin_pos[2]) < 0.5 :
            done = True
            reset = '--------- limit time steps ----------'
            print(reset)

        if self.steps > 5000 :
            done = True
            info = 'limit time steps'

        if  np.abs(ob[0]) < 0.2 and np.abs(ob[1] < 0.2 ):
            done = True
            reset = '---------- sim success ----------'
            print(reset)
            reward += 10000
        
        self.steps=self.steps+1

        return ob, reward, done, info


    async def step_async(self,action):
       
        action = [0,0,0,action]   ### for attitude contorl


        ###### angular velocity control task: pitch,roll,yaw,thrust
        await self.drone.offboard.set_attitude_rate(AttitudeRate(action[0],action[1],action[2],action[3]))   ## publish action in deg/s, thrust [0:1]
        
        ###### actuator control task: pitch,roll,yaw,thrust
        # await drone.offboard.set_actuator_control(ActuatorControl(action[0],action[1],action[2],action[3]))
        
        # await asyncio.sleep(0.1)
        # await self.pub_world_control.publish(step_msg)  ## perform one step in gazebo
        
        lin_pos = await self.get_lin_pos()
        lin_vel = await self.get_lin_vel()
        return lin_pos, lin_vel

    async def get_lin_pos(self):  ## in m
        async for position in self.drone.telemetry.position():
            glob_pos = np.array([position.latitude_deg, position.longitude_deg, position.absolute_altitude_m])
            lin_pos = np.array(lat_lon_to_coords(glob_pos)) - self.home_pos
            return lin_pos

    async def get_lin_vel(self):  ## in m/s
        async for vel in self.drone.telemetry.ground_speed_ned():
            lin_vel = np.array([vel.velocity_north_m_s, vel.velocity_east_m_s, vel.velocity_down_m_s])
            return lin_vel

    async def get_ang_pos(self): ## in rad
        async for ang_pos in self.drone.telemetry.attitude_euler():
            return [ang_pos.roll_deg, ang_pos.pitch_deg, ang_pos.yaw_deg]

    async def get_ang_vel(self):  ## in rad/s
        async for ang_vel in self.drone.telemetry.attitude_angular_velocity_body():
            return np.array([ang_vel.roll_rad_s, ang_vel.pitch_rad_s, ang_vel.yaw_rad_s])

    def render(self):
        pass

    def close (self):
        pass

    def land(self):
        self.loop.run_until_complete(self.asyland())

    async def asyland(self):
        print('-- Landing')
        unpause_msg = WorldControl()
        unpause_msg.pause = False
        await asyncio.sleep(0.5)
        await self.pub_world_control.publish(unpause_msg)
        await asyncio.sleep(0.5)
        await self.drone.action.land()