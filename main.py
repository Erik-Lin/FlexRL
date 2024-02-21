
from algorithms.dqn.runner import dqn_runner
from algorithms.ddpg.runner import ddpg_runner
from algorithms.ppo.runner import ppo_runner
from algorithms.sac.runner import sac_runner
import gymnasium as gym
import yaml
import argparse

def load_config(yaml_file_path):
    with open(yaml_file_path, 'r', encoding='utf-8') as file:  # 指定编码为 utf-8
        config = yaml.safe_load(file)
    return config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="dqn")
    args = parser.parse_args()

    if args.algo == 'dqn':
        config = load_config('./configs/dqn_maze_config.yaml')
    elif args.algo == 'ddpg':
        config = load_config('./configs/ddpg_maze_config.yaml')
    elif args.algo == 'ppo':
        config = load_config('./configs/ppo_maze_config.yaml')
    elif args.algo == 'sac':
        config = load_config('./configs/sac_maze_config.yaml')
    else:
        raise ValueError("请指定正确的算法")
    
    env = gym.make('CartPole-v1',render_mode='human')

    if config["algo"] == "dqn":
        dqn_runner(env, config)
    elif config["algo"] == "ddpg":
        ddpg_runner(env, config)
    elif config["algo"] == "ppo":
        ppo_runner(env, config)
    elif config["algo"] == "sac":
        sac_runner(env, config)
    else:
        raise ValueError("请检查配置文件是否正确")