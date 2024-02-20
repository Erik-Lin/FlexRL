import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import copy


class mazeEnv(gym.Env):
    def __init__(self,args):
        super(mazeEnv, self).__init__()
        self.args = args
        self.initial_maze = env_map1
        self.maze = copy.deepcopy(self.initial_maze) # 0为可通行，1为障碍，-1为已走过，2为终点
        self.height, self.width = len(self.initial_maze), len(self.initial_maze[0])
        self.x = 0
        self.y = 0
        self.start = (0, 0)  # 左上角为起点
        # self.goal = (self.height - 1, self.width - 1)  # 右下角为终点
        self.goal = (self.height - 1, self.width - 1)
        self.maze[self.goal[0]][self.goal[1]] = 2 # 标记终点
        self.maze[self.x][self.y] = -1 # 标记当前位置为已走过
        self.current_position = (self.x, self.y)
        self.distance = abs(self.x - self.goal[0]) + abs(self.y - self.goal[1])
        self.steps = 0
        self.reward = None
        self.path = [(0,0)] # 记录路径
        self.d = 0 # 平滑度比较参数（执行动作后位置与上一位置的曼哈顿距离，d=1说明走直线，d!=1说明走斜线）
        self.max_steps = args['max_steps']
        self.action_space = spaces.Discrete(8)  # 八个方向
        self.observation_space = spaces.Discrete(np.prod(np.shape(self.initial_maze)))
        self.action_mask = [[0 for i in range(self.action_space.n)] for j in range(self.action_space.n)]
        self.max_steps = 200

    def step(self, action):
        self.steps += 1

        # 0：上、1：下、2：左、3：右、4：左上、5：右上、6：左下、7：右下
        change = [[0, -1], [0, 1], [-1, 0], [1, 0], [-1, -1], [1, -1], [-1, 1], [1, 1]]  

        ax = self.x     # 记录执行动作前的位置
        ay = self.y
        self.dx = self.x + change[action][0]    # 执行动作后位置变化
        self.dy = self.y + change[action][1]
        D = self.distance       # 执行动作前距离目标前的距离
        self.x = min(self.width - 1, max(0, self.dx))    # 防止越界
        self.y = min(self.height - 1, max(0, self.dy))

        if self.maze[self.y][self.x] != 0  and self.maze[self.y][self.x] != 2: # 不可通行,本次step无效
            self.x = ax
            self.y = ay
            return np.array(self.maze).flatten(), -1, True, {}
        
        self.maze[self.y][self.x] = -1 # 标记为已走过

        self.path.append((self.x, self.y)) # 记录路径
        self.distance = abs(self.y - self.goal[0]) + abs(self.x - self.goal[1])   # 执行动作后当前位置与目标点距离

        done = False

        self.reward = -1
        # 到达目标点
        # if self.distance == 0:
        if self.y == self.goal[0] and self.x == self.goal[1]:
            done = True
            self.reward = 100
        # 到达最大步数

        if self.steps > self.max_steps:
            done = True
            self.reward = -50 
        
        next_state = np.array(self.maze).flatten()

        return next_state, self.reward, done, {}

    def reset(self):
        # 重置到起点
        self.maze = copy.deepcopy(self.initial_maze)
        self.x = 0
        self.y = 0
        self.current_position = (self.x, self.y)
        self.distance = abs(self.x - self.goal[0]) + abs(self.y - self.goal[1])
        self.steps = 0
        self.path = [(0,0)]
        self.d = 0
        self.maze[self.goal[0]][self.goal[1]] = 2 # 标记终点
        self.maze[self.y][self.x] = -1
        self.action_mask = [[0 for i in range(self.action_space.n)] for j in range(self.action_space.n)]

        self.max_steps = 200

        return np.array(self.maze).flatten()
    
    def get_action_mask(self):
        # [y][x]，因为我们说的[1][0]是第二列第一行，但是存储的[1][0]是第二行第一列
        for i in range(self.action_space.n):
            self.action_mask[i][i] = 0 
        # 上
        if self.y - 1 >= 0 and self.maze[self.y - 1][self.x] != -1 and self.maze[self.y - 1][self.x] != 1:
            self.action_mask[0][0] = 1
        # 下
        if self.y + 1 <= self.height - 1 and self.maze[self.y + 1][self.x] != -1 and self.maze[self.y + 1][self.x] != 1:
            self.action_mask[1][1] = 1
        # 左
        if self.x - 1 >= 0 and self.maze[self.y][self.x - 1] != -1 and self.maze[self.y][self.x - 1] != 1:
            self.action_mask[2][2] = 1
        # 右
        if self.x + 1 <= self.width - 1 and self.maze[self.y][self.x + 1] != -1 and self.maze[self.y][self.x + 1] != 1:
            self.action_mask[3][3] = 1
        # 左上
        if self.y - 1 >= 0 and self.x - 1 >= 0 and self.maze[self.y - 1][self.x - 1] != -1 and self.maze[self.y - 1][self.x - 1] != 1:
            self.action_mask[4][4] = 1
        # 右上
        if self.y - 1 >= 0 and self.x + 1 <= self.width - 1 and self.maze[self.y - 1][self.x + 1] != -1 and self.maze[self.y - 1][self.x + 1] != 1:
            self.action_mask[5][5] = 1
        # 左下
        if self.y + 1 <= self.height - 1 and self.x - 1 >= 0 and self.maze[self.y + 1][self.x - 1] != -1 and self.maze[self.y + 1][self.x - 1] != 1:
            self.action_mask[6][6] = 1
        # 右下
        if self.y + 1 <= self.height - 1 and self.x + 1 <= self.width - 1 and self.maze[self.y + 1][self.x + 1] != -1 and self.maze[self.y + 1][self.x + 1] != 1:
            self.action_mask[7][7] = 1

        return self.action_mask

    def render(self):
        # 打印当前地图状态，用数字表示不同的地点
        # print(self.maze)
        self.visualize_map()
    
    def visualize_map(self):
        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 5))
        # 整体地图
        ax.imshow(self.initial_maze, cmap='gray_r', interpolation='nearest')
        # 添加网格线
        for i in range(max(self.height, self.width)):
            ax.axhline(i - 0.5, color='black', lw=1)
            ax.axvline(i - 0.5, color='black', lw=1)
        ax.set_xlabel('self.width')
        ax.set_ylabel('self.height')
        ax.legend()
        # 显示图形
        plt.show()
        # plt.pause(0.02)
        # plt.close()

    def visualize_map_path(self, map, title):
        #  Path
        x1 = [point[0] for point in self.path]
        y1 = [point[1] for point in self.path]
        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 5))
        # 整体地图
        ax.imshow(map, cmap='gray_r', interpolation='nearest')
        # 添加网格线
        for i in range(20):
            ax.axhline(i - 0.5, color='black', lw=1)
            ax.axvline(i - 0.5, color='black', lw=1)
        ax.set_title(title)
        ax.set_xlabel('ncols')
        ax.set_ylabel('nrows')
        ax.plot(x1, y1, color='red', linestyle='solid', label='Path', linewidth=0.5)
        ax.legend()
        # 显示图形
        plt.show()
    
    def save_map_path(self, map, title):
        #  Path
        x1 = [point[0] for point in self.path]
        y1 = [point[1] for point in self.path]
        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 5))
        # 整体地图
        ax.imshow(map, cmap='gray_r', interpolation='nearest')
        # 添加网格线
        for i in range(20):
            ax.axhline(i - 0.5, color='black', lw=1)
            ax.axvline(i - 0.5, color='black', lw=1)
        ax.set_title(title)
        ax.set_xlabel('ncols')
        ax.set_ylabel('nrows')
        ax.plot(x1, y1, color='red', linestyle='solid', label='Path', linewidth=0.5)
        ax.legend()
        # 显示图形
        if self.args['train']:
            plt.savefig("./train_results/" + self.args['algo'] + "/map_" + title + ".png")
        else:
            plt.savefig("./test_results/" + self.args['algo'] + "/map_" + title + ".png")
        plt.close()

# 测试地图
env_map1 = [[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 20*20
            [0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0]]

env_map2 = [[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 20*20
            [0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 1, 2, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 0, 0, 1, 1, 1, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0]]