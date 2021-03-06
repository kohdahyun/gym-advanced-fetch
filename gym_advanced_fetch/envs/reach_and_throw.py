import os
from math import pi
import numpy as np
from gym import utils as ut
from gym.envs.robotics import fetch_env

from gym.envs.robotics import rotations, utils


MODEL_XML_PATH = os.path.join("fetch", "reach_and_throw.xml")
MODEL_XML_PATH = os.path.join(os.path.dirname(__file__), "assets", MODEL_XML_PATH)


class FetchReachAndThrowEnv(fetch_env.FetchEnv, ut.EzPickle):
    def __init__(self, reward_type="sparse"):
        self.success = 0
        self.moving_point = np.zeros(shape=(3,))
        self.slope = 0
        self.box_radius = 0.2
        self.del_x = 0
        self.del_y = 0
        initial_qpos = {
            "robot0:slide0": 0.4049,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "robot0:shoulder_lift_joint": - np.pi / 4,
            "robot0:upperarm_roll_joint": 0,
            "robot0:elbow_flex_joint": np.pi / 6,
            "robot0:wrist_roll_joint": 0,
            "robot0:wrist_flex_joint": np.pi / 3,
            "object0:joint": [1.3, 0.75018422, 0.62, 1., 0., 0., 0.],
        }
        self.goal = np.zeros(shape=(3,))
        self.object_qpos = np.zeros(shape=(7,))
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
            reward_type=reward_type,
        )
        ut.EzPickle.__init__(self, reward_type=reward_type)
        
    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos("robot0:grip")
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp("robot0:grip") * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

        object_pos = self.sim.data.get_site_xpos("object0")
        # rotations
        object_rot = rotations.mat2euler(self.sim.data.get_site_xmat("object0"))
        # velocities
        object_velp = self.sim.data.get_site_xvelp("object0") * dt
        object_velr = self.sim.data.get_site_xvelr("object0") * dt
        # gripper state
        object_rel_pos = object_pos - grip_pos
        object_velp -= grip_velp

        gripper_state = robot_qpos[-2:]
        gripper_vel = (
            robot_qvel[-2:] * dt
        )  # change to a scalar if the gripper is made symmetric

        #achieved_goal = grip_pos.copy()
        achieved_goal = np.squeeze(object_pos.copy())
        
        obs = np.concatenate(
            [
                grip_pos,
                object_pos.ravel(),
                object_rel_pos.ravel(),
                gripper_state,
                object_rot.ravel(),
                object_velp.ravel(),
                object_velr.ravel(),
                grip_velp,
                gripper_vel,
            ]
        )
        
        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
        }
        
    def _reset_sim(self):
        #return super()._reset_sim()
        self.sim.set_state(self.initial_state)

        self.object_qpos = self.sim.data.get_joint_qpos("object0:joint")
        assert self.object_qpos.shape == (7,)
        
        # table1 center: np.array([1.45 0.75018422 0.7]
        # table size: 0.1 0.45 0.35
        # constant height: 0.7 + 0.02
        
        self.object_qpos[0] = np.array(1.3 + self.np_random.uniform(-0.045,0.045))      
        self.object_qpos[1] = np.array(0.75018422 + self.np_random.uniform(-0.17,0.17))     
        self.object_qpos[2] = np.array(0.62)  
            
        self.sim.data.set_joint_qpos("object0:joint", self.object_qpos)
            
        self.sim.forward()
        return True
    
    def _sample_goal(self):
        #return super()._sample_goal()
        
        # self.goal[0] = self.object_qpos[0]
        # self.goal[1] = self.object_qpos[1]
        # self.goal[2] = self.object_qpos[2]
        
        # box center: 2.2 0.75018422 0.01
        
        self.goal[0] = np.array(2.2)
        self.goal[1] = np.array(0.75018422)
        self.goal[2] = np.array(0.4)
        
        #---------------------------------------------------------------------------------------------
        
        self.slope = (float)(self.goal[1]-self.object_qpos[1])/(self.goal[0]-self.object_qpos[0])
        
        self.moving_point[0] = np.array(1.3+0.1+self.box_radius)
        self.moving_point[1] = np.array(self.slope*(self.moving_point[0]-self.goal[0])+self.goal[1])
        self.moving_point[2] = self.goal[2]
        
        self.del_x = np.array((self.goal[0] - self.moving_point[0])/5)
        self.del_y = np.array((self.goal[1] - self.moving_point[1])/5)
        
        if ((self.success > 250) and (self.moving_point[0] != self.goal[0])):
            self.moving_point[0] += 5*self.del_x
            self.moving_point[1] += 5*self.del_y
        elif (self.success > 200):
            self.moving_point[0] += 4*self.del_x
            self.moving_point[1] += 4*self.del_y
        elif (self.success > 150):
            self.moving_point[0] += 3*self.del_x
            self.moving_point[1] += 3*self.del_y
        elif (self.success > 100):
            self.moving_point[0] += 2*self.del_x
            self.moving_point[1] += 2*self.del_y
        elif (self.success > 50):
            self.moving_point[0] += self.del_x
            self.moving_point[1] += self.del_y
                
        #----------------------------------------------------------------------------------------------
         
        return self.goal.copy() 

    def compute_reward(self, achieved_goal, goal, info):
        # # Compute distance between goal and the achieved goal.
        # # boxkdh center: 2.5 0.65 0
        # # high table center: 1.45 0.75018422 0.7
        # assert achieved_goal.shape == goal.shape
        # d = np.linalg.norm(achieved_goal - goal, axis=-1)
        # # if d < 0.7:
        # #     print(achieved_goal)
        # if self.reward_type == "sparse":
        #     #return -(d > self.distance_threshold).astype(np.float32)
        #     return -(d > 0.7).astype(np.float32)
        # else:
        #     return -d
        
        
        
        self.object_qpos = self.sim.data.get_joint_qpos("object0:joint")
        assert self.object_qpos.shape == (7,)
        
        # if self.reward_type == "sparse":
        #     return -(self.object_qpos[0] < 1.6).astype(np.float32)
        # else:
        #     return -self.object_qpos[0]    
        
        #-------------------------------------------------------------------
        if self.reward_type == "sparse":
            if ((self.object_qpos[2] < self.goal[2] + 0.1) and (self.object_qpos[2] > self.goal[2] - 0.1) and \
                (self.object_qpos[0] > np.array(self.moving_point[0] - self.box_radius)) and \
                    (self.object_qpos[0] < np.array(self.goal[0] + self.box_radius)) and \
                        (self.object_qpos[1] > np.array(self.slope*(self.object_qpos[0]-self.goal[0]) + self.goal[1] - self.box_radius)) and \
                            (self.object_qpos[1] < np.array(self.slope*(self.object_qpos[0]-self.goal[0]) + self.goal[1] + self.box_radius)) and \
                                (self.object_qpos[1] > np.array(self.goal[1] - self.box_radius)) and \
                                    (self.object_qpos[1] < np.array(self.moving_point[1] + self.box_radius))):
                                        return (np.array(0)).astype(np.float32)
            else:
                return (np.array(-1)).astype(np.float32)
        else:
            return -self.object_qpos[0] 
            
        #---------------------------------------------------------------------
        
        
        
        
    def _is_success(self, achieved_goal, desired_goal):
        # assert achieved_goal.shape == desired_goal.shape
        # d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        # if (d < 0.7).astype(np.float32):
        #     self.success += 1
        #     print(self.success)
        # #return (d < self.distance_threshold).astype(np.float32)
        # return (d < 0.7).astype(np.float32)
    
        self.object_qpos = self.sim.data.get_joint_qpos("object0:joint")
        assert self.object_qpos.shape == (7,)
        
        #if (self.object_qpos[0] > 1.6).astype(np.float32):

        if ((self.object_qpos[2] < self.goal[2] + 0.1) and (self.object_qpos[2] > self.goal[2] - 0.1) and \
            (self.object_qpos[0] > np.array(self.moving_point[0] - self.box_radius)) and \
                (self.object_qpos[0] < np.array(self.goal[0] + self.box_radius)) and \
                    (self.object_qpos[1] > np.array(self.slope*(self.object_qpos[0]-self.goal[0]) + self.goal[1] - self.box_radius)) and \
                        (self.object_qpos[1] < np.array(self.slope*(self.object_qpos[0]-self.goal[0]) + self.goal[1] + self.box_radius)) and \
                            (self.object_qpos[1] > np.array(self.goal[1] - self.box_radius)) and \
                                (self.object_qpos[1] < np.array(self.moving_point[1] + self.box_radius))):
                                        
            self.success += 1
            print(self.success)
        
        #return (self.object_qpos[0] > 1.6).astype(np.float32)
    #----------------------------------------------------------------------------           
        if ((self.object_qpos[2] < self.goal[2] + 0.1) and (self.object_qpos[2] > self.goal[2] - 0.1) and \
            (self.object_qpos[0] > np.array(self.moving_point[0] - self.box_radius)) and \
                (self.object_qpos[0] < np.array(self.goal[0] + self.box_radius)) and \
                    (self.object_qpos[1] > np.array(self.slope*(self.object_qpos[0]-self.goal[0]) + self.goal[1] - self.box_radius)) and \
                        (self.object_qpos[1] < np.array(self.slope*(self.object_qpos[0]-self.goal[0]) + self.goal[1] + self.box_radius)) and \
                            (self.object_qpos[1] > np.array(self.goal[1] - self.box_radius)) and \
                                (self.object_qpos[1] < np.array(self.moving_point[1] + self.box_radius))):
                                    return (np.array(1)).astype(np.float32)
        else:
            return (np.array(0)).astype(np.float32)

