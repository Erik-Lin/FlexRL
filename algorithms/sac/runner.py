from algorithms.sac.utils import *
from algorithms.sac.sac import SAC
import numpy as np
import torch
import random
import matplotlib.pyplot as plt

def sac_runner(env, args):
    target_entropy = args['target_entropy']
    hidden_dim = args['hidden_dim']
    actor_lr = eval(args['actor_lr'])
    critic_lr = eval(args['critic_lr'])
    alpha_lr = eval(args['alpha_lr'])
    gamma = args['gamma']
    tau = args['tau']
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")
    state_dim = env.observation_space.n
    action_dim = env.action_space.n
    agent = SAC(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, alpha_lr,
                target_entropy, tau, gamma, device)
    ensure_directory_exists("./train_results/sac")
    ensure_directory_exists("./test_results/sac")
    if args['train'] == True:
        train(env, agent, args)
    else:
        agent.load_models("./train_results/sac/sac_model_{}_".format(args['episodes']))
        test(env,agent,args)

def train(env,agent,args):
    num_episodes = 200
    buffer_size = 1000
    minimal_size = 100
    batch_size = 16

    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    replay_buffer = ReplayBuffer(buffer_size)
    return_list = train_off_policy_agent(env, agent, num_episodes,
                                                  replay_buffer, minimal_size,
                                                  batch_size)
    agent.save_models("./train_results/sac/sac_model_{}_".format(args['episodes']))
    episodes_list = list(range(len(return_list)))
    mv_return = moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('SAC on {}'.format(args['env']))
    plt.savefig('./train_results/sac/{}_mv_rewards_plot_on_{}.png'.format(args['episodes'],args['env']))
    plt.pause(0.01)
    plt.clf()

def test(env,agent,args):
    pass

