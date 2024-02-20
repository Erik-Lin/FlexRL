
import torch
import numpy as np
import matplotlib.pyplot as plt
from algorithms.ddpg.ddpg import Agent
from algorithms.ddpg.utils import MemoryBuffer,ensure_directory_exists

def smooth(input, window_size=10):
	# 使用窗口大小为 window_size 的移动平均平滑奖励
	smoothed_input = np.convolve(input, np.ones(window_size)/window_size, mode='valid')
	return smoothed_input

def train(env, trainer,model_path,args):
	episodes = int(args['episodes'])
	rewards = []
	losses_actor = []
	losses_critic = []
	for _ep in range(episodes):
		file_path = model_path + "ddpg_model_{}".format(_ep+1)
		observation = env.reset()
		totral_reward = 0
		for r in range(int(args['max_steps'])):
			# env.render()
			# print(observation)
			#action = trainer.get_exploration_action(state)
			if _ep%5 == 0:
				# validate every 5th episode
				action = trainer.get_exploitation_action(observation)
			else:
				# get action based on observation, use exploration policy here
				action = trainer.get_exploration_action(observation)
			action_probs = torch.softmax(torch.tensor(action), dim=-1)
			selected_action = torch.multinomial(action_probs.clone().detach(), 1).item()
			# print(action)
			# print(env.step(action))
			# new_observation, reward, terminal,truncted, info = env.step(action)
			new_observation, reward, done, info = env.step(selected_action)
			# done = terminal or truncted
			
			# # dont update if this is validation
			# if _ep%50 == 0 or _ep>450:
			# 	continue

			new_state = np.float32(new_observation)
			# push this exp in ram
			trainer.ram.add(observation, action, reward, new_state)
			totral_reward += reward
			observation = new_observation

			if done:
				break

		loss_a,loss_c = trainer.optimize()
		if _ep!=0 and _ep%500==0:
			trainer.save_models(file_path)
		# print('EPISODE := {}, total reward := {:.3f}, Loss_actor := {:.3f}, Loss_critic := {:.3f}'.format(_ep,totral_reward,loss_a,loss_c))
		rewards.append(totral_reward)
		losses_actor.append(loss_a)
		losses_critic.append(loss_c)
		
		# 创建一个1行3列的图表，其中第一个子图显示rewards，
		# 第二个子图显示losses_actor，第三个子图显示losses_critic
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

	plt.plot(rewards)
	plt.xlabel('Episode')
	plt.ylabel('Total Reward')
	plt.title('Reward over Episodes')
	plt.savefig('./train_results/ddpg/smoothed_reward_plot_{}.png'.format(episodes))  # Save the plot as an image
	plt.clf()  # Clear the plot for the next update

	smoothed_actor_loss = smooth(losses_actor)
	plt.plot(smoothed_actor_loss)
	plt.xlabel('Episode')
	plt.ylabel('Loss')
	plt.title('Loss over Episodes')
	plt.savefig('./train_results/ddpg/smoothed_actor_loss_plot_{}.png'.format(episodes))  # Save the plot as an image
	plt.clf()

	smoothed_critic_loss = smooth(losses_critic)
	plt.plot(smoothed_critic_loss)
	plt.xlabel('Episode')
	plt.ylabel('Critic Loss')
	plt.title('Critic Loss over Episodes')
	plt.savefig('./train_results/ddpg/smoothed_critic_loss_plot_{}.png'.format(episodes))  # Save the plot as an image
	plt.clf()

	trainer.save_models(file_path)
	print('Completed The Train')

def test(env, agent,args):
	episodes = int(args['test_episodes'])
	rewards = []
	for _ep in range(episodes):
		observation = env.reset()
		totral_reward = 0
		for r in range(int(args['max_steps'])):
			if _ep%5 == 0:
				# validate every 5th episode
				action = agent.get_exploitation_action(observation)
			else:
				# get action based on observation, use exploration policy here
				action = agent.get_exploration_action(observation)
			action_probs = torch.softmax(torch.tensor(action), dim=-1)
			selected_action = torch.multinomial(action_probs.clone().detach(), 1).item()

			new_observation, reward, done, info = env.step(selected_action)
			# done = terminal or truncted

			totral_reward += reward
			observation = new_observation

			if done:
				break

		rewards.append(totral_reward)

	plt.plot(rewards)
	plt.xlabel('Episode')
	plt.ylabel('Total Reward')
	plt.title('Reward over Episodes')
	plt.savefig('./test_results/ddpg/smoothed_reward_plot_{}.png'.format(episodes))  # Save the plot as an image
	plt.clf()  # Clear the plot for the next update
	print('Completed The Test')

def ddpg_runner(env, args):
	S_DIM = env.observation_space.n
	A_DIM = env.action_space.n
	A_MAX = env.action_space.n - 1
	ram = MemoryBuffer(args['capacity'])
	agent = Agent(S_DIM, A_DIM, A_MAX, ram)
	model_path = "./train_results/ddpg/"
	test_path = "./test_results/ddpg/"
	file_path = model_path + "ddpg_model_{}".format(args['episodes'])

	ensure_directory_exists(model_path)
	ensure_directory_exists(test_path)
	if args['train'] == True:
		train(env,agent, model_path,args)
	else:
		agent.load_models(file_path)
		test(env,agent,args)