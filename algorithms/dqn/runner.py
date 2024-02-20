from algorithms.dqn.dqn import DQNAgent
from algorithms.dqn.utils import ensure_directory_exists
import numpy as np
import matplotlib.pyplot as plt
import csv

def smooth(input, window_size=10):
    # 使用窗口大小为 window_size 的移动平均平滑奖励
    smoothed_input = np.convolve(input, np.ones(window_size)/window_size, mode='valid')
    return smoothed_input

# 训练DQN模型
def train_dqn(env, agent,  model_path,args):
    episodes = int(args['episodes'])
    rewards = []  # List to store total rewards for each episode
    losses = []

    for episode in range(episodes):
        state = env.reset()
        # state_array = np.array(list(state.values()))  # 将字典的值转换为数组
        # state_array = np.reshape(state_array, [1, env.state_size])
        # state = np.reshape(state, [1, env.state_size])
        total_reward = 0
        done = False

        while not done:
            action_index = agent.select_action(state)
            next_state, reward, done, _ = env.step(action_index)
            # next_state = np.reshape(next_state, [1, env.state_size])
            agent.remember(state, action_index, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                agent.train()
                agent.target_train()
                agent.decay_epsilon()
                # 记录损失值
                losses.append(agent.get_last_loss())

                # Update the plot
                rewards.append(total_reward)
                plt.plot(rewards)
                plt.xlabel('Episode')
                plt.ylabel('Total Reward')
                plt.title('Reward over Episodes')
                plt.pause(0.01)  # Add a small pause to update the plot
                plt.clf()  # Clear the plot for the next update

        # print("Episode {}: Total Reward: {}; state:{}".format(episode + 1, total_reward, state))
        print("Episode {}: Total Reward: {}".format(episode + 1, total_reward))

    # save model
    agent.save_model(model_path)

    smoothed_rewards = smooth(rewards)
    plt.plot(smoothed_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward over Episodes')
    plt.savefig('./train_results/dqn/smoothed_reward_plot_{}.png'.format(episodes))  # Save the plot as an image
    plt.clf()  # Clear the plot for the next update

    # 绘制损失曲线
    smoothed_losses = smooth(losses)
    plt.plot(smoothed_losses)
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Loss over Episodes')
    plt.savefig('./train_results/dqn/smoothed_loss_plot_{}.png'.format(episodes))  # Save the plot as an image

    # filename = "./results/dqn_trainning_result.csv"

    # # 将 rewards 和 losses 写入 CSV 文件
    # with open(filename, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['Rewards', 'Losses'])  # 写入列名
    #     for i in range(len(rewards)):
    #         if (i+1) % 10 == 0:
    #             writer.writerow([rewards[i], losses[i]])  # 写入每一行数据

def test_dqn(env, agent,args):
    episodes = int(args['test_episodes'])
    total_rewards = []
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # During testing, choose actions greedily based on the learned Q-values
            action_index = agent.select_action(state)
            next_state, reward, done, _ = env.step(action_index)
            state = next_state
            total_reward += reward

            if done:
                total_rewards.append(total_reward)
                print("Test Episode {}: Total Reward: {}; ".format(episode+1, total_reward))

    plt.plot(total_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward over Test Episodes')
    plt.savefig('./test_results/dqn/smoothed_reward_plot_{}.png'.format(episodes))  # Save the plot as an image
    plt.clf()  # Clear the plot for the next update
    avg_reward = np.mean(total_rewards)
    print("Average Test Reward over {} episodes: {}".format(episodes, avg_reward))
    print('Completed The Test')

def dqn_runner(env,args):
    state_size = env.observation_space.n
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    ensure_directory_exists("./train_results/dqn")
    ensure_directory_exists("./test_results/dqn")
    model_path = "./train_results/dqn/dqn_model_{}.pt".format(args['episodes'])
    if args['train'] == True:
        train_dqn(env, agent,model_path,args)
    else:
        agent.load_model(model_path)
        test_dqn(env, agent,args)