from copy import deepcopy
import numpy as np
import math
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces

class chipEnv(gym.Env):
    def __init__(self,args):
        mp = np.load("./environments/chipmap.npy")
        #change mp data type to float32
        mp = mp.astype(np.float32)
        self.mp_k=deepcopy(mp)
        self.agent_pos = [24, 1]
        self.target=[3, 16]
        self.path = []
        self.rendermode = args.rendermode
        self.mask = []
        self.observation_space = spaces.Discrete(len(np.array(mp).flatten()))
        self.action_space = spaces.Discrete(8)
        if self.rendermode=="human":
            plt.show()
            
    def render(self,raw_state):
        if self.rendermode == 'human':
            plt.imshow(raw_state)
            plt.pause(0.01)
            plt.clf()

    def reset(self):
        self.mp=deepcopy(self.mp_k)
        self.agent_pos = [24, 1]
        self.target=[3, 16]
        self.path = []
        self.mp[self.agent_pos[0],self.agent_pos[1]] = 11
        self.mp[self.target[0],self.target[1]] = 12
        
        # self.mask_update()
        state = deepcopy(self.mp)
        state = np.array(state).flatten()

        return state

    def move(self,action):
        if action == 0:
            self.agent_pos[0] += 1
        elif action == 1:
            self.agent_pos[0] += 1
            self.agent_pos[1] += 1
        elif action == 2:
            self.agent_pos[1] += 1
        elif action == 3:
            self.agent_pos[0] -= 1
            self.agent_pos[1] += 1
        elif action == 4:
            self.agent_pos[0] -= 1
        elif action == 5:
            self.agent_pos[0] -= 1
            self.agent_pos[1] -= 1
        elif action == 6:
            self.agent_pos[1] -= 1
        elif action == 7:
            self.agent_pos[0] += 1
            self.agent_pos[1] -= 1
        else:
            assert("action error")

    def step(self,action):
        last_pos=deepcopy(self.agent_pos)
        done = 0
        reward = -1
        info = {}
        self.move(action)
        #判断是否出界、撞墙、或者走了走过的路，如果是，则结束游戏
        if self.agent_pos[0] < 0 or self.agent_pos[0] >= 30 or self.agent_pos[1] < 0 or self.agent_pos[1] >= 30 \
        or self.mp[self.agent_pos[0],self.agent_pos[1]] == 1 or self.mp[self.agent_pos[0],self.agent_pos[1]] == 2: 
            done = 1
            reward = -10
        else :
            self.mp[last_pos[0], last_pos[1]] = 1
            self.mp[self.agent_pos[0],self.agent_pos[1]] = 11
            self.mp[self.target[0],self.target[1]] = 12

            if self.agent_pos[0]==self.target[0] and self.agent_pos[1]==self.target[1]:
                reward = 10
                done = 1
        # new_state = new_state.squeeze().detach().reshape(30, 30)
        new_state = deepcopy(self.mp)
        new_state = np.array(new_state).flatten()

        return new_state,reward,done,info

    # def mask_update(self):
    #     mask=np.zeros(8)
    #     out_c=np.ones((32,32))
    #     out_c[1:31,1:31]=self.mp

    #     mask[0] = 1-out_c[self.agent_pos[0] + 1+1,self.agent_pos[1]+1]
    #     mask[1] = 1-out_c[self.agent_pos[0] + 1+1, self.agent_pos[1] + 1+1]
    #     mask[2] = 1-out_c[self.agent_pos[0] + 1, self.agent_pos[1] + 1+1]
    #     mask[3] = 1-out_c[self.agent_pos[0] + 1-1, self.agent_pos[1] + 1+1]
    #     mask[4] = 1-out_c[self.agent_pos[0] + 1-1, self.agent_pos[1] + 1]
    #     mask[5] = 1-out_c[self.agent_pos[0] + 1-1, self.agent_pos[1] + 1-1]
    #     mask[6] = 1-out_c[self.agent_pos[0] + 1, self.agent_pos[1] + 1-1]
    #     mask[7] = 1-out_c[self.agent_pos[0] + 1+1, self.agent_pos[1] + 1-1]

    #     if mask[0] == 0 and mask[6] == 0:
    #         mask[7] = 0
    #     if mask[0] == 0 and mask[2] == 0:
    #         mask[1] = 0
    #     if mask[6] == 0 and mask[4] == 0:
    #         mask[5] = 0
    #     if mask[4] == 0 and mask[2] == 0:
    #         mask[3] = 0

    #     mask[mask == -11] = 10
    #     mask[mask < 0] = 0

    #     if mask[0] == 0 and mask[6] == 0 :
    #         mask[7] = 0
    #     if mask[0] == 0 and mask[2] == 0:
    #         mask[1] = 0
    #     if mask[6] == 0 and mask[4] == 0:
    #         mask[5] = 0
    #     if mask[4] == 0 and mask[2] == 0:
    #         mask[3] = 0

    #     self.mask = mask
