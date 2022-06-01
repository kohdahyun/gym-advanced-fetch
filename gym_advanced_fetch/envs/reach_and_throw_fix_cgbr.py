import os
from math import pi
import csv
import datetime

import numpy as np
from gym import utils as ut
from gym.envs.robotics import fetch_env
from gym.envs.robotics import rotations, utils


MODEL_XML_PATH = os.path.join("fetch", "reach_and_throw_fix.xml")
MODEL_XML_PATH = os.path.join(os.path.dirname(__file__), "assets", MODEL_XML_PATH)

save_list_path = "./lists/path_cgbr/cgbr_large/" + str(datetime.datetime.now()) + "/"

def check_if_directory_exists():
    if not os.path.exists(save_list_path):
        os.makedirs(save_list_path)

class FetchReachAndThrowFixCgbrEnv(fetch_env.FetchEnv, ut.EzPickle):
    def __init__(self, reward_type="sparse"):
        self.success = 0
        #------------------------------------------------------
        #self.old_success = 0
        #self.trial_success = 0
        #self.trial = 0
        #self.num_trial = 0
        #self.save_list = []
        self.change_list = 0
        self.x_list = []
        self.y_list = []
        #------------------------------------------------------
        #change cgbr-------------------------------------------
        #self.moving_point = np.zeros(shape=(3,))
        #self.slope = 0
        self.box_radius = 0.1
        #self.del_x = 0
        #self.del_y = 0
        self.del_radius = 0
        self.moving_radius = 0
        #change cgbr-------------------------------------------
        initial_qpos = {
            "robot0:slide0": 0.4049,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            # "robot0:shoulder_lift_joint": - 38 * np.pi / 180,
            # "robot0:upperarm_roll_joint": 0,
            # "robot0:elbow_flex_joint": np.pi / 6,
            # "robot0:wrist_roll_joint": 0,
            # "robot0:wrist_flex_joint": 38 * np.pi / 180,
            "robot0:shoulder_lift_joint": 0,
            "robot0:upperarm_roll_joint": 0,
            "robot0:elbow_flex_joint": 0,
            "robot0:wrist_roll_joint": 0,
            "robot0:wrist_flex_joint": 0,
            "object0:joint": [1.4, 0.74910048, 0.52, 1., 0., 0., 0.],
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
            distance_threshold=0.035,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
        )
        ut.EzPickle.__init__(self, reward_type=reward_type)
        
    def _set_action(self, action):
        assert action.shape == (4,)
        action = (
            action.copy()
        )  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.15  # limit maximum change in position
        
        rot_ctrl = [
            1.0,
            0.0,
            1.0,
            0.0,
        ]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)
        
    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos("robot0:grip")
        #print(grip_pos)
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
        # 1.4 0.74910048 0.51
        
        self.object_qpos[0] = np.array(1.4)      
        self.object_qpos[1] = np.array(0.74910048)     
        self.object_qpos[2] = np.array(0.52)  
        
        #----------------------------------------------------------
        #self.trial += 1
        #print("trial", self.trial)
        
        #if (self.old_success != 0):
            #self.trial_success += 1
            #print("trial_success", self.trial_success)
            
        #if (self.trial > 14):
            #save success rate
            #self.save_list.append(self.trial_success/self.trial)
            #self.num_trial += 1
            #print("num_trial", self.num_trial)
        
            #self.trial = 0
            #self.trial_success = 0
            
        #--------------------------------------------------
            
        if ((self.success > 8 * 20 * 1 - 1) and (self.change_list == 0)):
            check_if_directory_exists()
            # with open(save_list_path + 'List_path_fix_cgbr_large_1.csv','w') as file:
                
            #     write = csv.writer(file)
            #     write.writerow(self.save_list)
            self.change_list = 1
                
            with open(save_list_path + 'List_path_fix_cgbr_large_1_pos_x.csv','w') as file_x:
                
                write_x = csv.writer(file_x)
                write_x.writerow(self.x_list)
                
            with open(save_list_path + 'List_path_fix_cgbr_large_1_pos_y.csv','w') as file_y:
                
                write_y = csv.writer(file_y)
                write_y.writerow(self.y_list)
                
        if ((self.success > 8 * 20 * 2 - 1) and (self.change_list == 1)):
            check_if_directory_exists()
            # with open(save_list_path + 'List_path_fix_cgbr_large_2.csv','w') as file:
                
            #     write = csv.writer(file)
            #     write.writerow(self.save_list)
            self.change_list = 2
                
            with open(save_list_path + 'List_path_fix_cgbr_large_2_pos_x.csv','w') as file_x:
                
                write_x = csv.writer(file_x)
                write_x.writerow(self.x_list)
                
            with open(save_list_path + 'List_path_fix_cgbr_large_2_pos_y.csv','w') as file_y:
                
                write_y = csv.writer(file_y)
                write_y.writerow(self.y_list)
                
        if ((self.success > 8 * 20 * 3 - 1) and (self.change_list == 2)):
            check_if_directory_exists()
            # with open(save_list_path + 'List_path_fix_cgbr_large_3.csv','w') as file:
                
            #     write = csv.writer(file)
            #     write.writerow(self.save_list)
            self.change_list = 3
                
            with open(save_list_path + 'List_path_fix_cgbr_large_3_pos_x.csv','w') as file_x:
                
                write_x = csv.writer(file_x)
                write_x.writerow(self.x_list)
                
            with open(save_list_path + 'List_path_fix_cgbr_large_3_pos_y.csv','w') as file_y:
                
                write_y = csv.writer(file_y)
                write_y.writerow(self.y_list)
                
        if ((self.success > 8 * 20 * 4 - 1) and (self.change_list == 3)):
            check_if_directory_exists()
            # with open(save_list_path + 'List_path_fix_cgbr_large_4.csv','w') as file:
                
            #     write = csv.writer(file)
            #     write.writerow(self.save_list)
            self.change_list = 4
                
            with open(save_list_path + 'List_path_fix_cgbr_large_4_pos_x.csv','w') as file_x:
                
                write_x = csv.writer(file_x)
                write_x.writerow(self.x_list)
                
            with open(save_list_path + 'List_path_fix_cgbr_large_4_pos_y.csv','w') as file_y:
                
                write_y = csv.writer(file_y)
                write_y.writerow(self.y_list)
                
        if ((self.success > 8 * 20 * 5 - 1) and (self.change_list == 4)):
            check_if_directory_exists()
            # with open(save_list_path + 'List_path_fix_cgbr_large_5.csv','w') as file:
                
            #     write = csv.writer(file)
            #     write.writerow(self.save_list)
            self.change_list = 5
                #self.num_trial = 0
                
            with open(save_list_path + 'List_path_fix_cgbr_large_5_pos_x.csv','w') as file_x:
                
                write_x = csv.writer(file_x)
                write_x.writerow(self.x_list)
                
            with open(save_list_path + 'List_path_fix_cgbr_large_5_pos_y.csv','w') as file_y:
                
                write_y = csv.writer(file_y)
                write_y.writerow(self.y_list)
                
        if ((self.success > 8 * 20 * 6 - 1) and (self.change_list == 5)):
            check_if_directory_exists()
            # with open(save_list_path + 'List_path_fix_cgbr_large_6.csv','w') as file:
                
            #     write = csv.writer(file)
            #     write.writerow(self.save_list)
            self.change_list = 6
                
            with open(save_list_path + 'List_path_fix_cgbr_large_6_pos_x.csv','w') as file_x:
                
                write_x = csv.writer(file_x)
                write_x.writerow(self.x_list)
                
            with open(save_list_path + 'List_path_fix_cgbr_large_6_pos_y.csv','w') as file_y:
                
                write_y = csv.writer(file_y)
                write_y.writerow(self.y_list)       
        
        #self.old_success = 0
        #----------------------------------------------------------
            
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
        self.goal[1] = np.array(0.74910048)
        self.goal[2] = np.array(0.23)
        
        #---------------------------------------------------------------------------------------------
        #change cgbr---------------------------------------------------------------------------------
        # self.slope = (float)(self.goal[1]-self.object_qpos[1])/(self.goal[0]-self.object_qpos[0])
        
        # self.moving_point[0] = np.array(1.4+0.1+self.box_radius)
        # self.moving_point[1] = np.array(self.slope*(self.moving_point[0]-self.goal[0])+self.goal[1])
        # self.moving_point[2] = self.goal[2]
        
        # self.del_x = np.array((self.goal[0] - self.moving_point[0])/5)
        # self.del_y = np.array((self.goal[1] - self.moving_point[1])/5)
        
        # if (self.success > 250):
        #     self.moving_point[0] += 5*self.del_x
        #     self.moving_point[1] += 5*self.del_y
        # elif (self.success > 200):
        #     self.moving_point[0] += 4*self.del_x
        #     self.moving_point[1] += 4*self.del_y
        # elif (self.success > 150):
        #     self.moving_point[0] += 3*self.del_x
        #     self.moving_point[1] += 3*self.del_y
        # elif (self.success > 100):
        #     self.moving_point[0] += 2*self.del_x
        #     self.moving_point[1] += 2*self.del_y
        # elif (self.success > 50):
        #     self.moving_point[0] += self.del_x
        #     self.moving_point[1] += self.del_y
        
        self.moving_radius = np.array(self.box_radius + 5 * self.del_radius)
        
        #change to 0.2!
        self.del_radius = np.array((0.7 - self.box_radius)/5)
        #print("del_radius",self.del_radius)
        
        if (self.success > 8 * 20 * 5 - 1):
            self.moving_radius = np.array(self.box_radius)
        elif (self.success > 8 * 20 * 4 - 1):
            self.moving_radius = np.array(self.box_radius + self.del_radius)
        elif (self.success > 8 * 20 * 3 - 1):
            self.moving_radius = np.array(self.box_radius + 2 * self.del_radius)
        elif (self.success > 8 * 20 * 2 - 1):
            self.moving_radius = np.array(self.box_radius + 3 * self.del_radius)
        elif (self.success > 8 * 20 * 1 - 1):
            self.moving_radius = np.array(self.box_radius + 4 * self.del_radius)
        #change cgbr----------------------------------------------------------------------------------
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
            #change cgbr---------------------------------------------------------
            # return -((self.object_qpos[2] > self.goal[2] + 0.05) |\
            #     # (self.object_qpos[2] > self.goal[2] - 0.1) &\
            #     (self.object_qpos[0] < self.moving_point[0] - self.box_radius) |\
            #         (self.object_qpos[0] > self.goal[0] + self.box_radius) |\
            #             (self.object_qpos[1] < self.slope*(self.object_qpos[0]-self.goal[0]) + self.goal[1] - self.box_radius) |\
            #                 (self.object_qpos[1] > self.slope*(self.object_qpos[0]-self.goal[0]) + self.goal[1] + self.box_radius) |\
            #                     (self.object_qpos[1] < self.goal[1] - self.box_radius) |\
            #                         (self.object_qpos[1] > self.moving_point[1] + self.box_radius)).astype(np.float32)
            return -((self.object_qpos[2] > self.goal[2] + 0.05) |\
                #(self.object_qpos[2] < self.goal[2] - 0.01) |\
                (self.object_qpos[0] < self.goal[0] - self.moving_radius) |\
                    (self.object_qpos[0] > self.goal[0] + self.moving_radius) |\
                        (self.object_qpos[1] < self.goal[1] - self.moving_radius) |\
                            (self.object_qpos[1] > self.goal[1] + self.moving_radius)).astype(np.float32)
            #change cgbr--------------------------------------------------------------

        else:
            return -self.object_qpos 
            
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
        #change cgbr--------------------------------------------------------------------------
        # if (((self.object_qpos[2] < self.goal[2] + 0.05) & (self.object_qpos[2] > self.goal[2] - 0.05) &\
        #     (self.object_qpos[0] > self.moving_point[0] - self.box_radius) &\
        #         (self.object_qpos[0] < self.goal[0] + self.box_radius) &\
        #             (self.object_qpos[1] > self.slope*(self.object_qpos[0]-self.goal[0]) + self.goal[1] - self.box_radius) &\
        #                 (self.object_qpos[1] < self.slope*(self.object_qpos[0]-self.goal[0]) + self.goal[1] + self.box_radius) &\
        #                     (self.object_qpos[1] > self.goal[1] - self.box_radius) &\
        #                         (self.object_qpos[1] < self.moving_point[1] + self.box_radius)).astype(np.float32).any()):
            
        #     # print(((self.object_qpos[2] < self.goal[2] + 0.1) & (self.object_qpos[2] > self.goal[2] - 0.1) &\
        #     # (self.object_qpos[0] > self.moving_point[0] - self.box_radius) &\
        #     #     (self.object_qpos[0] < self.goal[0] + self.box_radius) &\
        #     #         (self.object_qpos[1] > self.slope*(self.object_qpos[0]-self.goal[0]) + self.goal[1] - self.box_radius) &\
        #     #             (self.object_qpos[1] < self.slope*(self.object_qpos[0]-self.goal[0]) + self.goal[1] + self.box_radius) &\
        #     #                 (self.object_qpos[1] > self.goal[1] - self.box_radius) &\
        #     #                     (self.object_qpos[1] < self.moving_point[1] + self.box_radius)).astype(np.float32))
        if (((self.object_qpos[2] < self.goal[2] + 0.05) &\
            #(self.object_qpos[2] > self.goal[2] - 0.01) &\
            (self.object_qpos[0] > self.goal[0] - self.moving_radius) &\
                (self.object_qpos[0] < self.goal[0] +self.moving_radius) &\
                    (self.object_qpos[1] > self.goal[1] - self.moving_radius) &\
                        (self.object_qpos[1] < self.goal[1] + self.moving_radius)).astype(np.float32).any()):
        #change cgbr---------------------------------------------------------------------------
            
            print(self.object_qpos)
                        
            self.success += 1
            #-----------------------------------------
            #self.old_success += 1
            #-----------------------------------------
            print(self.success)
            
        if ((self.object_qpos[2] < self.goal[2] + 0.05).astype(np.float32).any()):
            self.x_list.append(self.object_qpos[0])
            self.y_list.append(self.object_qpos[1])
        
        #return (self.object_qpos[0] > 1.6).astype(np.float32)
        #----------------------------------------------------------------------------  
        #change cgbr--------------------------------------------------------------------        
        # return ((self.object_qpos[2] < self.goal[2]) &\
        #         # (self.object_qpos[2] > self.goal[2] - 0.1) &\
        #     (self.object_qpos[0] > self.moving_point[0] - self.box_radius) &\
        #         (self.object_qpos[0] < self.goal[0] + self.box_radius) &\
        #             (self.object_qpos[1] > self.slope*(self.object_qpos[0]-self.goal[0]) + self.goal[1] - self.box_radius) &\
        #                 (self.object_qpos[1] < self.slope*(self.object_qpos[0]-self.goal[0]) + self.goal[1] + self.box_radius) &\
        #                     (self.object_qpos[1] > self.goal[1] - self.box_radius) &\
        #                         (self.object_qpos[1] < self.moving_point[1] + self.box_radius)).astype(np.float32)
        return ((self.object_qpos[2] < self.goal[2] + 0.05) &\
            #(self.object_qpos[2] > self.goal[2] - 0.01) &\
            (self.object_qpos[0] > self.goal[0] - self.moving_radius) &\
                (self.object_qpos[0] < self.goal[0] + self.moving_radius) &\
                    (self.object_qpos[1] > self.goal[1] - self.moving_radius) &\
                        (self.object_qpos[1] < self.goal[1] + self.moving_radius)).astype(np.float32)
        #change cgbr----------------------------------------------------------------------------------
        