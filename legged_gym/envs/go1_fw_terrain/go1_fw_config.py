
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym.envs.go1_fw_clock.go1_fw_config import Go1FwFlatClockCfg, Go1FwFlatClockCfgPPO

class Go1FwTerrainCfg(Go1FwFlatClockCfg):
 
    # FOR PLANE:
    class terrain( LeggedRobotCfg.terrain) :
        mesh_type = 'plane'
        curriculum = True
        measure_heights = False
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

class Go1FwTerrainCfgPPO(Go1FwFlatClockCfgPPO ):
    class runner(Go1FwFlatClockCfgPPO.runner ):
        run_name = ''
        experiment_name = 'go1_terrain'
        



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

  