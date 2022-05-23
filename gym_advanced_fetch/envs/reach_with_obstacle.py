import os
import numpy as np
from gym import utils
from gym.envs.robotics import fetch_env

MODEL_XML_PATH = os.path.join("fetch", "reach_with_obstacle.xml")
MODEL_XML_PATH = os.path.join(os.path.dirname(__file__), "assets", MODEL_XML_PATH)

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    #print(goal_a.shape)
    return np.linalg.norm(goal_a - goal_b, axis=-1)
    #return np.linalg.norm(goal_a - goal_b)

class FetchReachWithObstacleEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.4049,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        }
        self.goal = np.zeros(shape=(3,))
        fetch_env.FetchEnv.__init__(
            self,
            MODEL_XML_PATH,
            has_object=False,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=True,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.02,
            initial_qpos=initial_qpos,
            reward_type=reward_type,)
        # go to fetch_env.py and fix _sample_goal()
        utils.EzPickle.__init__(self)
        
    def _sample_goal(self):
        #return super()._sample_goal()
        self.goal[0] = np.array(1.3)
        while np.linalg.norm(self.goal[0] - np.array(1.3)) < 0.1:
            self.goal[0] = np.array(1.3 + self.np_random.uniform(-0.125,0.125))
            
        self.goal[1] = np.array(0.75)
        while np.linalg.norm(self.goal[1] - np.array(0.75)) < 0.1:
            self.goal[1] = np.array(0.75 + self.np_random.uniform(-0.175,0.175))
            
        self.goal[2] = np.array(0.6 + self.np_random.uniform(
            -0.18,0.2))
          
        return self.goal.copy() 
    
    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        #print(d)
        #return (d < self.distance_threshold).astype(np.float32)
        
        
        if (d < self.distance_threshold).any():
            return (d < self.distance_threshold).astype(np.float32)
        else:
            return (d < self.distance_threshold).astype(np.float32)
        

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        #print(d)
        # print(d.shape)
        # print(type(d))
        #print(self.distance_threshold.shape)
        #print(type(d))
        
        #print(type(self.distance_threshold))
        #<no> print(self.distance_threshold.shape)
        if self.reward_type == "sparse":
            #print(d)
            #print(self.distance_threshold)
            # if (-(d > self.distance_threshold).astype(np.float32) != 0.0 and\
            #     -(d > self.distance_threshold).astype(np.float32) != -1.0):
            #     print(-(d > self.distance_threshold).astype(np.float32))
            # return -(d > self.distance_threshold).astype(np.float32)
            
            
            if (d > self.distance_threshold).all():
                return -(d > self.distance_threshold).astype(np.float32)
            else:
                return -(d > self.distance_threshold).astype(np.float32)
        else:
            print("else", -d)
            return -d