from algorithms.ppo.ppo import Agent
from algorithms.ppo.utils import ensure_directory_exists
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
def smooth(input, window_size=10):
	# 使用窗口大小为 window_size 的移动平均平滑奖励
	smoothed_input = np.convolve(input, np.ones(window_size)/window_size, mode='valid')
	return smoothed_input

def ppo_runner(env,args):
    state_dim = env.observation_space.n
    action_dim = env.action_space.n
    agent = Agent(state_dim,action_dim,args)
    train_path = "./train_results/ppo/"
    test_path = "./test_results/ppo/"
    ensure_directory_exists(train_path)
    ensure_directory_exists(test_path)
    
    if args['train'] == True:
        train(env,agent,args)
    else:
        agent.load_models()
        test(env,agent,args)

def train(env,agent,args):
    rewards = []
    losses_actor = []
    losses_critic = []
    episodes = int(args['episodes'])
    for ep in tqdm(range(episodes)):
        state = env.reset()
        if args['render']:
            env.render()
        done = False
        totral_reward = 0
        loss_a = 0
        loss_c = 0
        while not done:
            action, action_prob = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            if args['render']:
                env.render()
            agent.memory(state, action, action_prob, reward, next_state)
            state = next_state
            totral_reward += reward
            if done :    
                break
        loss_a,loss_c = agent.update()
        rewards.append(totral_reward)
        losses_actor.append(loss_a)
        losses_critic.append(loss_c)
            
        # 创建一个1行3列的图表，其中第一个子图显示rewards，
        # 第二个子图显示losses_actor，第三个子图显示losses_critic
        if args['draw_show'] == True:
            plt.subplot(1, 3, 1)  # 第一个子图
            plt.plot(rewards)
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title('Reward over Episodes')

            plt.subplot(1, 3, 2)  # 第二个子图
            plt.plot(losses_actor)
            plt.xlabel('Episode')
            plt.ylabel('Actor Loss')
            plt.title('Actor Loss over Episodes')

            plt.subplot(1, 3, 3)  # 第三个子图
            plt.plot(losses_critic)
            plt.xlabel('Episode')
            plt.ylabel('Critic Loss')
            plt.title('Critic Loss over Episodes')

            # 调整子图的布局以防止重叠
            plt.tight_layout()  
            plt.pause(0.01)
            plt.clf()
    agent.save_models()
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward over Episodes')
    plt.savefig('./train_results/ppo/smoothed_reward_plot_{}.png'.format(episodes))  # Save the plot as an image
    plt.clf()  # Clear the plot for the next update

    smoothed_actor_loss = smooth(losses_actor)
    plt.plot(smoothed_actor_loss)
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Loss over Episodes')
    plt.savefig('./train_results/ppo/smoothed_actor_loss_plot_{}.png'.format(episodes))  # Save the plot as an image
    plt.clf()

    smoothed_critic_loss = smooth(losses_critic)
    plt.plot(smoothed_critic_loss)
    plt.xlabel('Episode')
    plt.ylabel('Critic Loss')
    plt.title('Critic Loss over Episodes')
    plt.savefig('./train_results/ppo/smoothed_critic_loss_plot_{}.png'.format(episodes))  # Save the plot as an image
    plt.clf()
    print('Completed The Train')

def test(env,agent,args):
    rewards = []
    episodes = int(args['test_episodes'])
    for ep in tqdm(range(episodes)):
        state = env.reset()
        if args['render']:
            env.render()
        done = False
        total_rewards = 0
        while not done:
            action, action_prob = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            if args['render']:
                env.render()
            state = next_state
            total_rewards += reward
            if done :
                break
        rewards.append(total_rewards)
        if args['draw_show'] == True:
            plt.plot(rewards)
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title('Reward over Episodes')
            plt.pause(0.01)
            plt.clf()
    
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward over Episodes')
    plt.savefig('./test_results/ppo/smoothed_reward_plot_{}.png'.format(episodes))  # Save the plot as an image
    plt.clf()  # Clear the plot for the next update