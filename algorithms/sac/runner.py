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
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    replay_buffer = ReplayBuffer(int(args['capacity']))
    return_list = []
    for i_episode in range(int(args['episodes'])):
        episode_return = 0
        state = env.reset()
        # state = dic2state(state)
        done = False
        while not done:
            action = agent.take_action(state)
            next_state, reward, done, _ = env.step(action)
            # next_state = dic2state(next_state)
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_return += reward
            if replay_buffer.size() > int(args['minimal_size']):
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(int(args['batch_size']))
                transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                agent.update(transition_dict)
        # print(state)
        return_list.append(episode_return)

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
    print("Completed the Train")
    
def test(env,agent,args):
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    return_list = []
    for i_episode in range(int(args['test_episodes'])):
        episode_return = 0
        state = env.reset()
        # state = dic2state(state)
        done = False
        while not done:
            action = agent.take_action(state)
            next_state, reward, done, _ = env.step(action)
            # next_state = dic2state(next_state)
            state = next_state
            episode_return += reward
        # print(state)
        return_list.append(episode_return)

    episodes_list = list(range(len(return_list)))
    mv_return = moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('SAC on {}'.format(args['env']))
    plt.savefig('./test_results/sac/test_{}_rewards_plot_on_{}.png'.format(args['episodes'],args['env']))
    plt.pause(0.01)
    plt.clf()
    print("Completed the Test")