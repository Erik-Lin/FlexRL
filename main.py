
from algorithms.dqn.runner import dqn_runner
from algorithms.ddpg.runner import ddpg_runner
from algorithms.ppo.runner import ppo_runner
from algorithms.sac.runner import sac_runner
import gymnasium as gym
import yaml

def load_config(yaml_file_path):
    with open(yaml_file_path, 'r', encoding='utf-8') as file:  # 指定编码为 utf-8
        config = yaml.safe_load(file)
    return config


if __name__ == '__main__':
    # args = load_config('./configs/dqn_maze_config.yaml')
    # args = load_config('./configs/ddpg_maze_config.yaml')
    args = load_config('./configs/ppo_maze_config.yaml')
    # args = load_config('./configs/sac_maze_config.yaml')

    env = gym.make('CartPole-v1',render_mode='human')

    if args["algo"] == "dqn":
        dqn_runner(env, args)
    elif args["algo"] == "ddpg":
        ddpg_runner(env, args)
    elif args["algo"] == "ppo":
        ppo_runner(env, args)
    elif args["algo"] == "sac":
        sac_runner(env, args)
    else:
        raise ValueError("请指定正确的算法")