import gym
from gym import spaces
#from envs.robotics.fetch.slide import FetchSlideEnv
#from reach_and_throw import FetchReachAndThrowEnv
#from reach_and_throw import FetchReachAndThrowEnv
#from slide import FetchSlideEnv
from reach_with_obstacle import FetchReachWithObstacleEnv

#env = FetchReachEnv()
env = FetchReachWithObstacleEnv()
#env = FetchSlideEnv()

env.reset()

#print("sample action:", env.action_space.sample())

#print("observation space shape:", env.observation_space.shape)
#print("sample observation:", env.observation_space.sample())

episodes = 10

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
    #for step in range(500):
        env.render()
        obs, reward, done, info = env.step(env.action_space.sample())
        #print(reward)

env.close()