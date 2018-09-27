import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import sys
#command-line arguments
assert len(sys.argv) >= 3, "Please specify 1.[dqn,double,dueling], 2.reward-fileName"
result_file=sys.argv[2]

import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
import gym
import itertools
import random

#####################################################################
# classes and methods
class ReplayBuffer():
	def __init__(self, min_size, max_size, recursive=False):
		self.episodes = []
		self.buffer = []
		self.min_size, self.max_size = min_size, max_size
		self.recursive = recursive

	def __len__(self):
		if self.recursive:
			return sum(map(len, self.episodes))
		else:
			return map(len, self.buffer)

	@property
	def full(self):
		if self.recursive:
			return len(self.episodes) >= self.max_size
		else:
			return len(self.buffer) >= self.max_size

	@property
	def initialized(self):
		if self.recursive:
			return len(self.episodes) >= self.min_size
		else:
			return len(self.buffer) >= self.min_size

	def add(self, episode=None, data_tuple=None):
		if self.recursive:
			assert episode is not None, "[ERROR] episode cannot be None for recursive replay buffer"
			if self.full:
			    self.episodes.pop(0)
			self.episodes.append(episode)
		else:
			assert data_tuple is not None, "[ERROR] data_tuple cannot be None for replay buffer"
			if self.full:
			    self.buffer.pop(0)
			self.buffer.append(data_tuple)

	def sample(self, batch_size, sequence_length=0):
		if self.recursive:
			assert sequence_length != 0, "Please specify sequence_length != 0"
			def take_seq():
			    episode = random.choice(self.episodes)
			    start = random.randint(0, len(episode)-sequence_length)
			    return [np.array(x) for x in zip(*episode[start:start+sequence_length])]
			return [take_seq() for b in range(batch_size)]
		else:
			batch_sample = []
			for idx in range(batch_size):
				sample = random.choice(self.buffer)
				batch_sample.append(sample)
			return batch_sample

class dqn():
	def __init__(self,obs_shape,num_actions,q_func_model,scope='dqn',reuse=False,gamma=1.0,grad_clip_value=1.0):
		with tf.variable_scope(scope, reuse=reuse):
			#placeholders for inputs
			self.obs_in = tf.placeholder(shape=obs_shape, dtype=tf.float32, name="observation")
			self.act_in = tf.placeholder(shape=[None], dtype=tf.int32, name="action")
			self.rew_in = tf.placeholder(shape=[None], dtype=tf.float32, name="reward")
			self.done_in = tf.placeholder(shape=[None], dtype=tf.float32, name="done")
			self.obs_p_in = tf.placeholder(shape=obs_shape, dtype=tf.float32, name="observation_p")		

			#compute prediction
			self.q_func = q_func_model(self.obs_in, num_actions, scope="q_func", reuse=False)
			self.q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "/q_func")
			self.q_func_target = q_func_model(self.obs_p_in, num_actions, scope="q_func_target", reuse=False)
			self.q_func_target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "/q_func_target")
			self.act_pred = tf.argmax(self.q_func, axis=1)			

			#training
			self.act_selected = tf.one_hot(self.act_in, num_actions, dtype=tf.float32)
			self.q_val = tf.reduce_sum(tf.multiply(self.q_func, self.act_selected), axis=1)
			self.q_val_target = (1.0 - self.done_in) * tf.reduce_max(self.q_func_target, axis=1)
			self.q_val_selected_target = self.rew_in + gamma * self.q_val_target
			self.td_error = tf.square(self.q_val - tf.stop_gradient(self.q_val_selected_target))
			self.loss = tf.reduce_mean(self.td_error)
			self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
			if not grad_clip_value:
				self.optimize_expr = self.optimizer.minimize(self.loss, var_list=self.q_func_vars)		
			else:
				gradients = self.optimizer.compute_gradients(self.loss, var_list=self.q_func_vars)
				for i, (grad, var) in enumerate(gradients):
					if grad is not None:
						gradients[i] = (tf.clip_by_norm(grad, grad_clip_value), var)  #clipping value = 1.0
				self.optimize_expr = self.optimizer.apply_gradients(gradients)

			#target network update
			self.update_target_expr = []
			for var, var_target in zip(sorted(self.q_func_vars, key=lambda v: v.name),
										sorted(self.q_func_target_vars, key=lambda v: v.name)):
				self.update_target_expr.append(var_target.assign(var))
			self.update_target_expr = tf.group(*self.update_target_expr)

#some global variables
num_cpu=4
max_buffer_size = 50000
batch_size = 64
target_net_upd_steps = 1000
recursive = False

def model(inpt, num_actions, scope, reuse=False):
	with tf.variable_scope(scope, reuse=reuse):
		out = inpt
		out = layers.fully_connected(out, num_outputs=128, activation_fn=tf.nn.tanh)
		out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
		return out

if __name__ == '__main__':
	tf.reset_default_graph()
	tf_config = tf.ConfigProto(
		inter_op_parallelism_threads=num_cpu,
		intra_op_parallelism_threads=num_cpu)
	# tf_config.gpu_options.allocator_type = 'BFC'

	with tf.Session(config=tf_config) as sess:
		env = gym.make('CartPole-v1')
		obs = env.reset()

		#create a model for the environment
		obs_shape = (None,env.observation_space.shape[0])
		print ("obs_shape:{}".format(obs_shape))
		num_actions = env.action_space.n
		dqn = dqn(obs_shape, num_actions, model)
		sess.run(tf.global_variables_initializer())

		#create a buffer for the environment
		replay_buffer = ReplayBuffer(batch_size*2, 50000, recursive=recursive)
		episode_reward = []
		episode = []
		episode_reward.append(0)
		exploration_steps = 50000

		for t in itertools.count():
			exploration = max(1.0-(float(t)/exploration_steps),0.05)
			# act = env.action_space.sample()
			act = sess.run(dqn.act_pred,feed_dict={dqn.obs_in:obs[None]})[0]
			if random.random() < exploration:
				act = env.action_space.sample()
			obs_p, rew, done, _ = env.step(act)
			episode_reward[-1] += rew
			# env.render()
			# add the sample to the buffer
			data_tuple = (obs, act, rew, obs_p, done)
			replay_buffer.add(data_tuple=data_tuple)
			obs = obs_p

			if t>50000 and np.mean(episode_reward[-101:-1]) > 500:
				env.render()

			if done:
				obs = env.reset()
				#episode finished
				if len(episode_reward) % 10 == 0:
					print ("#:{}\texploration:{}\t\tmean episode reward:{}".format(len(episode_reward), exploration, round(np.mean(episode_reward[-101:-1]), 1)))
					# print ("mean episode reward:{}".format(round(np.mean(episode_reward[-101:-1]), 1)))
				if len(episode_reward) % 100 == 0:
					np.savetxt(result_file, np.array(episode_reward))
					# exit(0)
				episode_reward.append(0)

			# training
			if t>250:
				before = sess.run(tf.trainable_variables())
				#sample from the buffer
				samples = replay_buffer.sample(batch_size)
				obs_t, act_t, rew_t, obs_p_t, done_t = map(np.array, zip(*samples))
				# before_loss = sess.run(dqn.loss,feed_dict={dqn.obs_in:obs_t, dqn.act_in:act_t, dqn.rew_in:rew_t, dqn.obs_p_in:obs_p_t, dqn.done_in:done_t})
				# before_loss2 = sess.run(dqn.loss,feed_dict={dqn.obs_in:obs_t, dqn.act_in:act_t, dqn.rew_in:rew_t, dqn.obs_p_in:obs_p_t, dqn.done_in:done_t})
				# assert before_loss == before_loss2, "loss isn't equal"
				#compute the optimizer expression
				sess.run(dqn.optimize_expr,feed_dict={dqn.obs_in:obs_t, dqn.act_in:act_t, dqn.rew_in:rew_t, dqn.obs_p_in:obs_p_t, dqn.done_in:done_t})
				# after_loss = sess.run(dqn.loss,feed_dict={dqn.obs_in:obs_t, dqn.act_in:act_t, dqn.rew_in:rew_t, dqn.obs_p_in:obs_p_t, dqn.done_in:done_t})
				# assert before_loss > after_loss, "loss didn't improve: {}, {}".format(before_loss, after_loss)
				after = sess.run(tf.trainable_variables())
				for idx in range(len(after)//2):
					a, b = after[idx], before[idx]
					# assert (b != a).any(), "variables are equal:{},{}".format(b, a)
					if not (b != a).any():
						print ("---------------------------- variables are equal:{},{}".format(b, a))
				for idx in range(len(after)//2,len(after)):
					a, b = after[idx], before[idx]
					# assert (b == a).all(), "variables are not equal:{},{}".format(b, a)
					if not (b == a).all():
						print ("---------------------------- variables are not equal:{},{}".format(b, a))

			#update target network
			if t>1000 and t % target_net_upd_steps == 0:
				before = sess.run(tf.trainable_variables())
				sess.run(dqn.update_target_expr, feed_dict={})
				after = sess.run(tf.trainable_variables())
				for idx in range(len(after)//2, len(after)):
					a, b = after[idx], before[idx]
					assert (b != a).any(), "variables are equal:{},{}".format(b, a)
				for idx in range(len(after)//2):
					a = after[idx]
					assert len(a) == len(after[idx+(len(after)//2)]), "arrays are not equal in length: {},{}".format(len(a), len(after[idx+(len(after)//2)]))
					assert (a == after[idx+(len(after)//2)]).all(), "arrays are not equal:{},{}".format(a, after[idx+(len(after)//2)])