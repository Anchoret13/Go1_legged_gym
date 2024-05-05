
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym.envs.go1_fw_clock.go1_fw_config import Go1FwFlatClockCfg, Go1FwFlatClockCfgPPO

class Go1FwTerrainCfg(Go1FwFlatClockCfg):
    class env:
        num_envs = 4096
        num_actions = 12
        num_observations = 42 
        num_privileged_obs = 45 + 187
        episode_length_s = 20 # episode length in seconds
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
    # FOR PLANE:
    class terrain( Go1FwFlatClockCfg.terrain) :
        mesh_type = 'trimesh'
        measure_heights = True

        # NOTE: [plane, rough, slope, stair, discrete, stepping]
        curriculum = True
        terrain_proportions = [0.4, 0.15, 0.15, 0.15, 0.15]

        # customized_terrain = True
        # customized_terrain_set = {
        #     'plane': 0.5,
        #     'rough': 0.3,
        #     'pyramid_stairs':0.2
        # }
        # terrain_proportions = [0, 1.0, 0, 0, 0.0]
        # NOTE: varibale below when mesh_type is trimesh
        # if selected => terrain_kwargs is not None
        selected = False
'''
        terrain_kwargs = {
            'type': 'rough',
            'min_height': -0.01,
            'max_height': 0.02,
            'step': 0.01,
            'downsampled_scale': 0.2
        }
        terrain_kwargs = {
            'type': 'discrete',
            'max_height': 0.05,
            'min_size': 1.0,
            'max_size': 5.0,
            'num_rects': 15
        }

        terrain_kwargs = {
            'type': 'pyramid_stairs',
            'step_width': 0.31,
            'step_height': 0.05,
        }
        
        terrain_kwargs = {
            'type': 'slope',
            'slope': 0.3,
        }
'''
        
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
    # class rewards( Go1FwFlatClockCfg.rewards ):

    #     class scales:
    #         # REMOVE HISTORY REWARD
    #         torques = -0.001
    #         # masked_legs_energy = -5e-3
    #         masked_legs_energy = -5e-4
    #         tracking_ang_vel = 1.0
    #         lin_vel_x = 0.9
    #         tracking_lin_vel_x = 3.5
    #         orientation = -0.5
    #         lin_vel_z = -0.0
    #         action_rate = -0.01
    #         roller_action_rate = -0.1
    #         hip = -1.0
    #         penalize_roll = -2.5                  # was -0.5
    #         front_leg = -3.5                      # was -0.5
    #         front_hip = -1.0
    #         raibert_heuristic = -5.0
    #         rear_feet_air_time = 3.5
    #         # penalize_slow_x_vel = 1.0
    #         feet_clearance = -0.0
    #         # tracking_contacts_binary = -0.1  
    #         roller_action_diff = -1.0
    #         # alive = 0.5
    #         base_height = -0.0
    #         collision = -0.0

class Go1FwTerrainCfgPPO(Go1FwFlatClockCfgPPO ):
    class runner(Go1FwFlatClockCfgPPO.runner ):
        run_name = ''
        experiment_name = 'go1_terrain'
        