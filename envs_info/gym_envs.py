import gym


# print("Gym Envs:", gym.envs.registry.all())

"""
Breakout-v0
CartPole-v0
FrozenLake-v0
"""
env = gym.make('FrozenLake-v0')
print("action_space:", env.action_space)
print("observation_space:", env.observation_space)
# print("observation_space.low:", env.observation_space.low)
# print("observation_space.high:", env.observation_space.high)
# print(env.observation_space.sample(), env.action_space.sample())
# print(env.observation_space.n, env.action_space.n)

# for _ in range(1000):
#     env.render()
#     env.step(env.action_space.sample()) 
# env.close()

env.reset()
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        # print("observation:", observation)
        action = env.action_space.sample() # take a random action
        observation, reward, done, info = env.step(action)
        # End the episode if agent is dead
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            # break
env.close()
