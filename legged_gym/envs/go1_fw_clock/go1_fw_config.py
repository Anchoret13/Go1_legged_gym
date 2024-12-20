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
        num_observations = 42
        num_privileged_obs = 42 + 6
        num_adapt_input = 27
        num_adapt_output = 10

        # NOTE: For data collection:
        # episode_length_s = 80
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.35] # x,y,z [m]

        #    'FL_hip_joint': 0.1,   # [rad]
        #     'RL_hip_joint': 0.1,   # [rad]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.05,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.05 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]

            'FL_roller_foot_joint': 0,
            'FR_roller_foot_joint': 0
        }
    
    # FOR PLANE:
    class terrain( LeggedRobotCfg.terrain) :
        mesh_type = 'plane'
        # curriculum = True
        measure_heights = True
        # selected = True
        static_friction = 0.8
        dynamic_friction = 0.8
        
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
        # stiffness = {'hip_joint': 30.0, 'thigh_joint': 30.0, 'calf_joint': 30.0, 'roller': 0.0}  # [N*m/rad]
        stiffness = {
            'FL_hip_joint': 40.0,  
            'RL_hip_joint': 30.0,  
            'FR_hip_joint': 40.0, 
            'RR_hip_joint': 30.0, 

            'FL_thigh_joint': 35.0,
            'RL_thigh_joint': 35.0, 
            'FR_thigh_joint': 35.0,  
            'RR_thigh_joint': 35.0,   

            'FL_calf_joint': 35.0,  
            'RL_calf_joint': 35.0,  
            'FR_calf_joint': 35.0,  
            'RR_calf_joint': 35.0,  

            'FL_roller_foot_joint': 0,
            'FR_roller_foot_joint': 0
        }
        # damping = {'hip_joint': 0.5, 'thigh_joint': 0.5, 'calf_joint': 0.5, 'roller': 0.0}     # [N*m*s/rad]
        damping = {
            'FL_hip_joint': 0.5,  
            'RL_hip_joint': 0.5,  
            'FR_hip_joint': 0.5, 
            'RR_hip_joint': 0.5, 

            'FL_thigh_joint': 0.5,
            'RL_thigh_joint': 0.5, 
            'FR_thigh_joint': 0.5,  
            'RR_thigh_joint': 0.5,   

            'FL_calf_joint': 0.5,  
            'RL_calf_joint': 0.5,  
            'FR_calf_joint': 0.5,  
            'RR_calf_joint': 0.5,  

            'FL_roller_foot_joint': 0,
            'FR_roller_foot_joint': 0
        }
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1_fw/urdf/go1_fw3_contact.urdf'
        name = "go1"
        foot_name = "foot"
        roller_name = "roller"
        penalize_contacts_on = ["thigh", "calf", "hip"]
        terminate_after_contacts_on = ["base", "hip"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
       
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        replace_cylinder_with_capsule = False # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False # Some .obj meshes must be flipped from y-up to z-up
        
    class commands(LeggedRobotCfg.commands):
        # num_commands = 1
        class ranges(LeggedRobotCfg.commands.ranges):
            heading = [0, 0]
            lin_vel_x = [0.0, 4.0] # min max [m/s]
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
            # REMOVE HISTORY REWARD
            torques = -0.001
            # masked_legs_energy = -5e-3
            masked_legs_energy = -5e-4
            tracking_ang_vel = 1.0
            lin_vel_x = 0.1
            tracking_lin_vel_x = 5.5
            orientation = -1.0
            lin_vel_z = -1.0
            action_rate = -0.01
            roller_action_rate = -0.1
            hip = -1.0
            penalize_roll = -2.5                  
            front_leg = -3.5                      
            front_hip = -1.0
            raibert_heuristic = -5.0
            rear_feet_air_time = 3.5
            # penalize_slow_x_vel = 1.0
            feet_clearance = -6.0
            # tracking_contacts_binary = -0.1  
            roller_action_diff = -0.05
            # alive = 0.5

            base_height = -0.1
            collision = -1.5

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 0.75]
        push_robots = False
        # push_interval_s = 15
        # max_push_vel_xy = 1.0
        randomize_base_mass = True

        added_mass_range = [0, 3]
        randomize_com_displacement = False
        com_displacement_range = [-0.10, 0.10]

class Go1FwFlatClockCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'roller_skating_asac'
        



  