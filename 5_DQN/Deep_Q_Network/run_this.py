from maze_env import Maze
from RL_brain import DeepQNetwork
# from DQN_modified import DeepQNetwork
import matplotlib.pyplot as plt
import numpy as np


def run_maze():
    step = 0
    rewards_in_episode = []
    episodes = 300
    for episode in range(episodes):
        # initial observation
        observation = env.reset()

        rewards = []
        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            rewards.append(reward)

            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

        rewards_mean = np.mean(np.array(rewards))
        rewards_in_episode.append(rewards_mean)

    # end of game
    print('game over')
    env.destroy()
    plt.plot(np.arange(episodes), rewards_in_episode)
    plt.ylabel('Rewards')
    plt.xlabel('training steps')
    plt.show()


if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      output_graph=True
                      )
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()