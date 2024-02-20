import os
import glob
import time
from datetime import datetime
from matplotlib import pyplot as plt
import torch
import numpy as np
# import roboschool
from algorithms.ppo.ppo import PPO
import torch.distributions as distributions
from algorithms.ppo.utils import ensure_directory_exists
def smooth(input, window_size=10):
    # 使用窗口大小为 window_size 的移动平均平滑奖励
    smoothed_input = np.convolve(input, np.ones(window_size)/window_size, mode='valid')
    return smoothed_input


################################### Training ###################################
def train(env, ppo_agent, args, model_path):
    random_seed = 0         # set random seed if required (0 = no random seed)
    if random_seed:
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

    rewards = []
    losses = []
    # training loop
    for ep in range(int(args['episodes'])):
        state = env.reset()
        current_ep_reward = 0
        current_ep_loss = 0
        done = False
        while not done:
            action_probs,state_val = ppo_agent.select_action(state)
            dist = distributions.Categorical(probs=action_probs)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            state, reward, done, _ = env.step(action)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)
            ppo_agent.buffer.states.append(state)
            ppo_agent.buffer.actions.append(action)
            ppo_agent.buffer.logprobs.append(action_logprob)
            ppo_agent.buffer.state_values.append(state_val)

            current_ep_reward += reward
            current_ep_loss += ppo_agent.get_last_loss()
            # update PPO agent
            if ep!=0 and ep % args['update_interval'] == 0:
                ppo_agent.update()

            # break; if the episode is over
            if done:
                rewards.append(current_ep_reward)
                losses.append(current_ep_loss)

                # Update the plot
                plt.plot(rewards)
                plt.xlabel('Timestep')
                plt.ylabel('Average  Reward')
                plt.title('Reward over Timesteps')
                plt.pause(0.01)  # Add a small pause to update the plot
                plt.clf()  # Clear the plot for the next update
                break

    ppo_agent.save(model_path)

    smoothed_rewards = smooth(rewards)
    plt.plot(smoothed_rewards)
    plt.xlabel('Timestep')
    plt.ylabel('Average Reward')
    plt.title('Reward over Timesteps')
    plt.savefig('./train_results/ppo/smoothed_reward_plot_tmiestep_{}.png'.format(int(args['episodes'])))  # Save the plot as an image
    plt.clf()  # Clear the plot for the next update

    # 绘制损失曲线
    smoothed_losses = smooth(losses)
    plt.plot(smoothed_losses)
    plt.xlabel('Timestep')
    plt.ylabel('Average Loss')
    plt.title('Loss over Timesteps')
    plt.savefig('./train_results/ppo/smoothed_loss_plot_timestep_{}.png'.format(int(args['episodes'])))  # Save the plot as an image
    plt.clf()  # Clear the plot for the next update
    plt.close()

################################### Training ###################################
def test(env, ppo_agent,args):
    random_seed = 0         # set random seed if required (0 = no random seed)
    if random_seed:
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

    running_reward = 0
    time_step = 0
    rewards = []
    ep = 0
    # training loop
    while ep <= int(args['episodes']):
        state = env.reset()
        current_ep_reward = 0
        for t in range(int(args['max_steps'])):
            action_probs,state_val = ppo_agent.select_action(state)
            dist = distributions.Categorical(probs=action_probs)
            action = dist.sample()
            state, reward, done, _ = env.step(action)
            time_step +=1
            current_ep_reward += reward
            # break; if the episode is over
            if done:
                break
        ep += 1
        running_reward += current_ep_reward
        rewards.append(running_reward)

    smoothed_rewards = smooth(rewards)
    plt.plot(smoothed_rewards)
    plt.xlabel('Timestep')
    plt.ylabel('Average Reward')
    plt.title('Reward over Timesteps')
    plt.savefig('./test_results/ppo/smoothed_reward_plot_{}.png'.format(int(args['episodes'])))  # Save the plot as an image
    plt.clf()  # Clear the plot for the next update

# #################################### Testing ###################################
# def test(env, ppo_agent, max_test_episodes, model_path):

#     max_ep_len = 1000           # max timesteps in one episode
#     render = False             # render environment on screen
#     frame_delay = 0             # if required; add delay b/w frames
#     ppo_agent.load(model_path)

#     test_running_reward = 0

#     for ep in range(1, max_test_episodes+1):
#         ep_reward = 0
#         state = env.reset()

#         for t in range(1, max_ep_len+1):
#             action = ppo_agent.select_action(state)
#             state, reward, done, _ = env.step(action)
#             ep_reward += reward

#             if render:
#                 env.render()
#                 time.sleep(frame_delay)

#             if done:
#                 break

#         # clear buffer
#         ppo_agent.buffer.clear()

#         test_running_reward +=  ep_reward
#         print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
#         ep_reward = 0

#     print("============================================================================================")
#     avg_test_reward = test_running_reward / max_test_episodes
#     avg_test_reward = round(avg_test_reward, 2)
#     print("average test reward : " + str(avg_test_reward))
#     print("============================================================================================")


# if __name__ == '__main__':
def ppo_runner(env, args):
    K_epochs = args['K_epochs']               # update policy for K epochs in one PPO update
    eps_clip = args['eps_clip']          # clip parameter for PPO
    gamma = args['gamma']            # discount factor
    lr_actor = args['lr_actor']       # learning rate for actor network
    lr_critic = args['lr_critic']       # learning rate for critic network
    max_training_timesteps = int(args['episodes'])*int(args['max_steps'])   # break training loop if timeteps > max_training_timesteps
    max_test_timesteps = int(1e4)
    model_path = "./train_results/ppo/ppo_model_{}.pt".format(max_training_timesteps)
    ensure_directory_exists("./train_results/ppo")
    ensure_directory_exists("./test_results/ppo")
    state_dim = env.observation_space.n
    action_dim = env.action_space.n
    has_continuous_action_space = False
    action_std = 0.6
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
    if args['train'] == True:
        train(env, ppo_agent, args, model_path)
    else:
        ppo_agent.load(model_path)
        test(env, ppo_agent, args)