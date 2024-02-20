import gymnasium as gym
from gymnasium import spaces
import numpy as np
from itertools import combinations
import copy

class medicationEnv(gym.Env):
    def __init__(self, args):
        super(medicationEnv, self).__init__()
        self.args = args
        self.max_steps = args.max_steps  # 环境最大步数
        self.drugs = np.array([  # 药物名称
            'Rosiglitazone',
            'Glimepiride',
            'Pioglitazone',
            'Glyburide',
            'Glipizide'
        ])
        self.costs = np.array([1.31, 0.916, 0.866, 0.807, 1.267])  # 药物花费
        self.initial_prices = np.array([3.43, 2.3628, 2.234, 2.0826, 3.269])  # 初始价格
        self.probabilities_path = './environments/normalized_probabilities.npy'  # 初始概率文件路径
        self.drug_probabilities = np.load(self.probabilities_path, allow_pickle=True).item()  # 药物初始概率
        # self.all_combinations = list(combinations(self.drugs, 3))
        # self.combination_probabilities = self.calculate_combination_probabilities()

        self.current_state = {  # 当前状态
            drug: {'probability': self.drug_probabilities[drug], 'price': self.initial_prices[i]}
            for i, drug in enumerate(self.drugs)
        }
        self.action_space = spaces.Discrete(6)  # 动作空间
        self.observation_space = spaces.Discrete(len(self.drugs) * 2)  # 状态空间
        self.action_mapping = {  # 动作映射关系
            0: ('Glyburide', 0.88, 1.20),
            1: ('Glyburide', 1.0, 1.0),
            2: ('Glyburide', 1.12, 0.80),
            3: ('Glipizide', 0.88, 1.20),
            4: ('Glipizide', 1.0, 1.0),
            5: ('Glipizide', 1.12, 0.80),
        }
        self.current_step = 0  # 当前步
    
    def calculate_combination_probabilities(self):
        # 计算每种组合的概率
        probabilities = [self.calculate_probability(combination) for combination in self.all_combinations]
        total_prob = sum(probabilities)

        # 归一化概率
        return [prob / total_prob for prob in probabilities]
    
    def calculate_probability(self, combination):
        # 根据药物选择概率计算给定组合的概率
        probability = 1.0
        for drug in self.drugs:
            if drug in combination:
                probability *= self.drug_probabilities[drug]
            else:
                probability *= (1 - self.drug_probabilities[drug])
        return probability

    def reset(self):
        # 重置环境到初始状态
        self.current_state = {drug: {'probability': self.drug_probabilities[drug], 'price': self.initial_prices[i]} for i, drug in enumerate(self.drugs)}
        return self.dic2state(self.current_state)

    def take_action(self, drug, rate_change, price_change):
        # 对药物价格进行调整
        next_state = copy.deepcopy(self.current_state)
        # original_prob = self.drug_probabilities.copy()
        next_state[drug]['price'] *= price_change
        next_state[drug]['probability'] *= rate_change
        total_change = self.current_state[drug]['probability'] - next_state[drug]['probability']

        # 平均分配给其他药物
        for i in self.drugs:
            if i != drug:
                next_state[i]['probability'] += total_change / (len(self.drugs) - 1)

        return next_state

    def step(self, action):
        # 对药物价格进行调整
        drug, rate_change, price_change = self.action_mapping[action]
        next_state = self.take_action(drug, rate_change, price_change)
        # new_combination_probability = self.calculate_combination_probabilities()

        # 计算奖励
        # combination_probability = transition_probabilities[0] / self.drug_probabilities[1]
        reward = self.calculate_reward(self.current_state, next_state, self.costs)
        self.current_state = next_state

        # 返回下一个状态、奖励和是否结束的信息
        # next_state = transition_probabilities
        done = self.current_step >= self.max_steps
        if done:
            self.current_step = 0
        else:
            self.current_step += 1
        info = {}  # 额外的信息，可以为空

        # 将字典转成列表形式
        next_state = self.dic2state(next_state)
        return next_state, reward, done, info

    def render(self, mode='human'):
        # 打印当前状态信息
        for key in self.current_state:
            print("drug: {0}, probability: {1}, price: {2}".format(self.current_state[key],
                                                                   self.current_state[key]["probability"],
                                                                   self.current_state[key]["price"]))
        print(self.current_state)

    @staticmethod
    def calculate_reward(old_state, new_state, costs):
        # 计算奖励
        drugs_of_interest = ['Glyburide', 'Glipizide']

        old_rewards = np.sum([old_state[drug]['probability'] * (old_state[drug]['price'] - costs[i]) for i, drug in enumerate(old_state) if drug in drugs_of_interest])
        new_rewards = np.sum([new_state[drug]['probability'] * (new_state[drug]['price'] - costs[i]) for i, drug in enumerate(new_state) if drug in drugs_of_interest])

        reward = new_rewards - old_rewards
        return reward

    @staticmethod
    def dic2state(dic):
        # 将字典转成列表
        state = []
        for key in dic:
            for k in dic[key]:
                state.append(dic[key][k])
        return state