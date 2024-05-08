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
from legged_gym.envs.go1_fw_clock.go1_fw_config import Go1FwFlatClockCfg, Go1FwFlatClockCfgPPO

class Go1FwFlatTiltCfg( Go1FwFlatClockCfg):
    class env( Go1FwFlatClockCfg.env):
        num_envs = 4096
        num_actions = 12
        num_observations = 42  
        num_privileged_obs = 45 + 2
    class init_state( Go1FwFlatClockCfg.init_state ):
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
            'FR_roller_foot_joint': 0,

            'FL_tilt_joint': 0,
            'FR_tilt_joint': 0
        }
    
    # FOR PLANE:
    # class terrain( Go1FwFlatClockCfg.terrain) :
    #     mesh_type = 'plane'
    #     curriculum = True
    #     measure_heights = True
    #     selected = True
        
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

    class control( Go1FwFlatClockCfg.control ):
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
            'FR_roller_foot_joint': 0,

            'FR_tilt_joint':0.0,
            'FL_tilt_joint':0.0,
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
            'FR_roller_foot_joint': 0,

            'FR_tilt_joint':0.0,
            'FL_tilt_joint':0.0,
        }
        
    class asset( Go1FwFlatClockCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1_fw/urdf/go1_fw3_contact_tilt_wheel.urdf'
        roller_tilt_name = "tilt"


    class domain_rand(Go1FwFlatClockCfg.domain_rand):
        roller_tilt_rand_range = [-0.1, 0.1]

class Go1FwFlatTiltCfgPPO( Go1FwFlatClockCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'go1_roller_tilt'
        
