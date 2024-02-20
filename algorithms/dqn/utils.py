import numpy as np
import torch
import shutil
import torch.autograd as Variable
import numpy as np
import random
from collections import deque
import os

def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created: {path}")
    else:
        print(f"Directory already exists: {path}")
		
class MemoryBuffer:
	def __init__(self, size):
		self.buffer = deque(maxlen=size)
		self.maxSize = size
		self.len = 0

	def sample(self, count):
		"""
		samples a random batch from the replay memory buffer
		:param count: batch size
		:return: batch (numpy array)
		"""
		batch = []
		count = min(count, self.len)
		batch = random.sample(self.buffer, count)
		#for i, arr in enumerate(batch):
		#	print(f"Sample {i}:")
		#	for j, val in enumerate(arr):
		#		print(f"  Element {j} - Type: {type(val)}, Value: {val}")

		s_arr = np.array([arr[0] for arr in batch], dtype=np.float32)
		a_arr = np.array([arr[1] for arr in batch], dtype=np.float32)
		r_arr = np.array([arr[2] for arr in batch], dtype=np.float32)
		s1_arr = np.array([arr[3] for arr in batch], dtype=np.float32)
		return s_arr, a_arr, r_arr, s1_arr

	def len(self):
		return self.len

	def add(self, s, a, r, s1):
		"""
		adds a particular transaction in the memory buffer
		:param s: current state
		:param a: action taken
		:param r: reward received
		:param s1: next state
		:return:
		"""
		s_arr = np.array(s, dtype=np.float32)
		a_arr = np.array(a, dtype=np.float32)
		r_arr = np.array(r, dtype=np.float32)
		s1_arr = np.array(s1, dtype=np.float32)
		transition = (s_arr, a_arr, r_arr, s1_arr)
		self.len += 1
		if self.len > self.maxSize:
			self.len = self.maxSize
		self.buffer.append(transition)

def soft_update(target, source, tau):
	"""
	Copies the parameters from source network (x) to target network (y) using the below update
	y = TAU*x + (1 - TAU)*y
	:param target: Target network (PyTorch)
	:param source: Source network (PyTorch)
	:return:
	"""
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(
			target_param.data * (1.0 - tau) + param.data * tau
		)

def hard_update(target, source):
	"""
	Copies the parameters from source network to target network
	:param target: Target network (PyTorch)
	:param source: Source network (PyTorch)
	:return:
	"""
	for target_param, param in zip(target.parameters(), source.parameters()):
			target_param.data.copy_(param.data)