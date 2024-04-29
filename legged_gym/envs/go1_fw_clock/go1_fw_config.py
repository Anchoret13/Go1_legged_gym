# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Go1FwFlatClockCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env):
        num_envs = 4096
        num_actions = 12
        num_observations = 42 + 6
        num_privileged_obs = 45 + 6
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.35] # x,y,z [m]

        #    'FL_hip_joint': 0.1,   # [rad]
        #     'RL_hip_joint': 0.1,   # [rad]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.0,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.0 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.6,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.6,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.6,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.6,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]

            'FL_roller_foot_joint': 0,
            'FR_roller_foot_joint': 0
        }
    
    # FOR PLANE:
    class terrain( LeggedRobotCfg.terrain) :
        mesh_type = 'plane'
        curriculum = True
        measure_heights = True
        selected = True
        
    # class terrain( LeggedRobotCfg.terrain):
    #     mesh_type = 'trimesh'
        # horizontal_scale  = 0.1
        # vertical_scale = 0.001
        # border_size = 0
        # curriculum = True
        # static_friction = 1.0
        # dynamic_friction = 1.0
        # restitution = 0.0
        # TODO: terrain noise magnitude
        # TODO: terrain smoothness
        # measure_heights = True
        # measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        # measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        # selected = False
        # terrain_length = 20.
        # terrain_width = 20.
        # num_cols = 10
        # num_rows = 10
        # # NOTE: terrain_proportions: [smooth slope, rough slope, stairs up, stairs down, discrete]
        # terrain_proportions = [0.4, 0.4, 0.0, 0.0, 0.2]
        # # trimesh  
        # slope_treshold = 0.

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        gaits_type = 'fix_f'
        stiffness = {'hip_joint': 30.0, 'thigh_joint': 30.0, 'calf_joint': 30.0, 'roller': 0.0}  # [N*m/rad]
        damping = {'hip_joint': 0.5, 'thigh_joint': 0.5, 'calf_joint': 0.5, 'roller': 0.0}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1_fw/urdf/go1_fw3_contact.urdf'
        name = "go1"
        foot_name = "foot"
        roller_name = "roller"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
       
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        replace_cylinder_with_capsule = False # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False # Some .obj meshes must be flipped from y-up to z-up
        
    class commands(LeggedRobotCfg.commands):
        # num_commands = 1
        class ranges(LeggedRobotCfg.commands.ranges):
            heading = [0, 0]
            lin_vel_x = [0.5, 2.5] # min max [m/s]
            # old range 3/15
            lin_vel_y = [0.0, 0.0]
            # ang_vel_yaw = [-0.0, 0.]    # min max [rad/s]
            ang_vel_yaw = [-0.0, 0.0]


    class rewards( LeggedRobotCfg.rewards ):
        # soft_dof_pos_limit = 0.01 # NOTE: trying fully following wtw setting
        soft_dof_pos_limit = 0.9
        # self_dof_vel_limit = 0.01
        base_height_target = 0.35

        only_positive_rewards = False

        # max_contact_force = 300


        class scales:
            # dof_pos_limits = -0.4
            # torque_limits = -0.01
            # dof_vel_limits = -10.0
            # wheel_air_time = -2.                # commented 

            # add by xiaoyu
            # tracking_lin_vel = 4.0                # was 3.5
            # tracking_ang_vel = 0.5              # commented
            
            # lin_vel_x = 0.5
            
            # orientation = -3.0
            # collision = -1.0                    # commented   
            # base_height = -0.1
            # lin_vel_z = -0.05                     # was -0.1
            # action_rate = -0.01
            # roller_action_rate = -0.05
            # hip = -1.0
            # penalize_roll = -0.5                  # was -0.5
            # front_leg = -1.5                      # was -0.5
            # front_hip = -1.0

            # alive:
            # alive = 0.1

            # gait reward
            # tracking_contacts_binary = -0.1           # comment for contact
            # raibert_heuristic = -0.1
            # tracking_rear_swing_force = 0.5
            # tracking_rear_stance_vel = 0.5
            # tracking_swing_force = 1.0
            # tracking_stance_vel = 1.0


            #***********************************
            #    testing 3/15
            #***********************************  
            # torques = -0.001
            # masked_legs_energy = -1e-4
            # tracking_ang_vel = 1.0
            # lin_vel_x = 1.0
            # tracking_lin_vel_x = 2.5
            # orientation = -0.5
            # lin_vel_z = -0.05
            # action_rate = -0.01
            # roller_action_rate = -0.05
            # hip = -1.0
            # penalize_roll = -0.5                  # was -0.5
            # front_leg = -1.5                      # was -0.5
            # front_hip = -1.0
            # raibert_heuristic = -1.0
            # tracking_rear_swing_force = 0.5
            # tracking_rear_stance_vel = 0.5
            # tracking_swing_force = 1.0
            # tracking_stance_vel = 1.0

            #***********************************
            #    testing 3/22
            #*********************************** 
            # torques = -0.0001
            # masked_legs_energy = -5e-3
            # tracking_lin_vel_x = 2.0
            # tracking_ang_vel = 0.5
            # lin_vel_x = 1.0
            # lin_vel_z = -1.0
            # # ang_vel_xy = -0.05
            # # orientation = -1.
            # action_rate = -0.01
            # roller_action_rate = -0.05
            # hip = -1.0
            # penalize_roll = -0.5
            # front_leg = -1.5
            # front_hip = -1.0
            # raibert_heuristic = -1.0
            # rear_feet_air_time = 3.0

            #***********************************
            #    testing 4/14
            #*********************************** 
            torques = -0.001
            # masked_legs_energy = -5e-3
            masked_legs_energy = -1e-3
            tracking_ang_vel = 1.0
            lin_vel_x = 1.0
            tracking_lin_vel_x = 3.5
            orientation = -0.1
            lin_vel_z = -0.05
            action_rate = -0.01
            roller_action_rate = -0.1
            hip = -1.0
            penalize_roll = -0.5                  # was -0.5
            front_leg = -3.5                      # was -0.5
            front_hip = -1.0
            raibert_heuristic = -2.0
            rear_feet_air_time = 3.5
            # penalize_slow_x_vel = 1.0
            feet_clearance = -3.0
            roller_action_diff = -1.0
            alive = 0.5

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.75, 1.5]
        push_robots = False
        # push_interval_s = 15
        # max_push_vel_xy = 1.0
        randomize_base_mass = False

        # added_mass_range = [-1, 3]
        hip_friction_sim = False
        hip_action_noise = 0.1

class Go1FwFlatClockCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'roller_skating_asac'
        



'''
Reward notes:

3/16:
1. fast; base stable
            torques = -0.001
            masked_legs_energy = -1e-4
            lin_vel_z = -0.05                    
            action_rate = -0.01
            roller_action_rate = -0.05
            hip = -0.5
            penalize_roll = -1.0                 
            front_leg = -1.5                      
            front_hip = -1.0
            raibert_heuristic = -0.1
            tracking_swing_force = 1.0
            tracking_stance_vel = 1.0


            #***********************************
            #    testing 3/15
            #***********************************  
            tracking_ang_vel = 0.5
            lin_vel_x = 3.0
            orientation = -0.5


'''

  