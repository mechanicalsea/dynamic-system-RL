import numpy as np

class ac_Q_learning(object):
	"""
	--------
	@Example:
	--------
	qlearning = ac_Q_learning(n_episodes=1000, n_iterations=200, gamma=0.9,
	                          n_batch=32, agent=ppo, env=env)
	
	--------
	@Parameter
	--------
	n_episodes: int, default 1000
	n_iterations: int, default 200
	gamma: float, default 0.9
	n_batch: int, default 32
	agent: class agent, default None
	env: class environment, default None

	--------
	@Method
	--------
	main_loop(self): main interface
	deploy_learner(self, agent, env): deploy agent and environment
	plot_reward(self): plot reward curves

	"""

	def __init__(self, n_episodes=1000, n_iterations=200, gamma=0.9, 
				 n_batch=32, agent=None, env=None):
		self.n_episodes = n_episodes
		self.n_iterations = n_iterations
		self.gamma = gamma
		self.n_batch = n_batch
		self.agent = agent
		self.env = env

		self._init_transition()

	def deploy_learner(self, agent, env):
		self.agent = agent
		self.env = env
		self._init_transition()
		self.total_r = []

	def _store_transtion(self, s, a, r):
		"""store history (s, a, r)"""
		self.buffer_s.append(s)
		self.buffer_a.append(a)
		self.buffer_r.append(r)

	def _init_transition(self):
		"""clear former history"""
		self.buffer_s = []
		self.buffer_a = []
		self.buffer_r = []

	def main_loop(self):
		self.total_r = []
		for ep in range(self.n_episodes):
			s = self.env.reset()
			buffer_s, buffer_a, buffer_r = [], [], []
			ep_r = 0
			for t in range(self.n_iterations):
				#env.render()
				a = self.agent.choose_action(s)
				s_, r, done, _ = self.env.step(a)
				self._store_transtion(s, a, (r+8)/8)
				s = s_
				ep_r += r
				# update agent
				if (t+1) % self.n_batch == 0 or t == self.n_iterations - 1:
					v_s_ = self.agent.get_v(s_)
					discounted_r = [] 
					for r in self.buffer_r[::-1]:
						v_s_ = r + self.gamma * v_s_
						discounted_r.append(v_s_)
					discounted_r.reverse()

					ss = np.vstack(self.buffer_s)
					aa = np.vstack(self.buffer_a)
					rr = np.array(discounted_r)[:, np.newaxis]
					self._init_transition()
					self.agent.update(ss, aa, rr)

			if ep == 0: self.total_r.append(ep_r)
			else: self.total_r.append(self.total_r[-1] * 0.9 + ep_r * 0.1)
			#print('\rEp: %6d, Ep_r: %.0f' % (ep, ep_r), end='')
			print('Ep: %d, Ep_r: %.0f' % (ep, ep_r))

	def plot_reward(self):
		import matplotlib.pyplot as plt
		plt.plot(np.arange(len(self.total_r)), self.total_r)
		plt.xlabel('Episode')
		plt.ylabel('Moveing averaged episode reward')
		plt.show()


if __name__ == '__main__':
	import time
	import gym
	from actor_critic_agents import PPO

	env = gym.make('Pendulum-v0').unwrapped
	if hasattr(env, 'n_features'):
		print(env.n_features)
	env.n_features = 3
	ppo = PPO(3, 1)
	qlearning = ac_Q_learning(agent=ppo, env=env)
	start = time.time()
	qlearning.main_loop()
	end = time.time()
	print('It takes %.0f' % (end-start))
	qlearning.plot_reward()
		
