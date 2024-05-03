
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym.envs.go1_fw_clock.go1_fw_config import Go1FwFlatClockCfg, Go1FwFlatClockCfgPPO

class Go1FwTerrainCfg(Go1FwFlatClockCfg):
 
    # FOR PLANE:
    class terrain( Go1FwFlatClockCfg.terrain) :
        mesh_type = 'trimesh'
        curriculum = True
        measure_heights = True
        selected = False
        terrain_proportions = [0, 1.0, 0, 0, 0.0]
        
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
    class rewards( Go1FwFlatClockCfg.rewards ):

        class scales:
            # REMOVE HISTORY REWARD
            torques = -0.001
            # masked_legs_energy = -5e-3
            masked_legs_energy = -5e-4
            tracking_ang_vel = 1.0
            lin_vel_x = 0.9
            tracking_lin_vel_x = 3.5
            orientation = -0.5
            lin_vel_z = -0.0
            action_rate = -0.01
            roller_action_rate = -0.1
            hip = -1.0
            penalize_roll = -2.5                  # was -0.5
            front_leg = -3.5                      # was -0.5
            front_hip = -1.0
            raibert_heuristic = -5.0
            rear_feet_air_time = 3.5
            # penalize_slow_x_vel = 1.0
            feet_clearance = -0.0
            # tracking_contacts_binary = -0.1  
            roller_action_diff = -1.0
            # alive = 0.5
            base_height = -0.0
            collision = -0.0

class Go1FwTerrainCfgPPO(Go1FwFlatClockCfgPPO ):
    class runner(Go1FwFlatClockCfgPPO.runner ):
        run_name = ''
        experiment_name = 'go1_terrain'
        