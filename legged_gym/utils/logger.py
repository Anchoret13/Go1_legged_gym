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

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value
import pickle as pkl

class Logger:
    def __init__(self, dt):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if 'rew' in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()

    def plot_states(self):
        self.plot_process = Process(target=self._plot)
        self.plot_process.start()

    # def _plot(self):
    #     nb_rows = 4
    #     nb_cols = 3
    #     fig, axs = plt.subplots(nb_rows, nb_cols)
    #     for key, value in self.state_log.items():
    #         time = np.linspace(0, len(value)*self.dt, len(value))
    #         break
    #     log= self.state_log
    #     # plot joint targets and measured positions
    #     a = axs[1, 0]
    #     if log["dof_pos"]: a.plot(time, log["dof_pos"], label='measured')
    #     if log["dof_pos_target"]: a.plot(time, log["dof_pos_target"], label='target')
    #     a.set( ylabel='Position [rad]', title='DOF Position vs time [s]')
    #     a.legend()
    #     # plot joint velocity
    #     a = axs[1, 1]
    #     if log["dof_vel"]: a.plot(time, log["dof_vel"], label='measured')
    #     if log["dof_vel_target"]: a.plot(time, log["dof_vel_target"], label='target')
    #     a.set( ylabel='Velocity [rad/s]', title='Joint Velocity vs time [s]')
    #     a.legend()
    #     # plot base vel x
    #     a = axs[0, 0]
    #     if log["base_vel_x"]: a.plot(time, log["base_vel_x"], label='measured')
    #     if log["command_x"]: a.plot(time, log["command_x"], label='commanded')
    #     a.set(ylabel='base lin vel [m/s]', title='Base velocity x vs time [s]')
    #     a.legend()
    #     # plot base vel y
    #     a = axs[0, 1]
    #     if log["base_vel_y"]: a.plot(time, log["base_vel_y"], label='measured')
    #     if log["command_y"]: a.plot(time, log["command_y"], label='commanded')
    #     a.set( ylabel='base lin vel [m/s]', title='Base velocity y vs time [s]')
    #     a.legend()
    #     # plot base vel yaw
    #     a = axs[0, 2]
    #     if log["base_vel_yaw"]: a.plot(time, log["base_vel_yaw"], label='measured')
    #     if log["command_yaw"]: a.plot(time, log["command_yaw"], label='commanded')
    #     a.set(ylabel='base ang vel [rad/s]', title='Base velocity yaw vs time [s]')
    #     a.legend()
    #     # plot base vel z
    #     a = axs[1, 2]
    #     if log["base_vel_z"]: a.plot(time, log["base_vel_z"], label='measured')
    #     a.set(ylabel='base lin vel [m/s]', title='Base velocity z vs time [s]')
    #     a.legend()
    #     # plot contact forces
    #     a = axs[2, 0]
    #     if log["contact_forces_z"]:
    #         forces = np.array(log["contact_forces_z"])
    #         for i in range(forces.shape[1]):
    #             a.plot(time, forces[:, i], label=f'force {i}')
    #     a.set(ylabel='Forces z [N]', title='Vertical Contact forces vs time [s]')
    #     a.legend()
    #     # plot torque/vel curves
    #     a = axs[2, 1]
    #     if log["dof_vel"]!=[] and log["dof_torque"]!=[]: a.plot(log["dof_vel"], log["dof_torque"], 'x', label='measured')
    #     a.set( ylabel='Joint Torque [Nm]', title='Torque/velocity curves')
    #     a.legend()
    #     # plot torques
    #     a = axs[2, 2]
    #     if log["dof_torque"]!=[]: a.plot(time, log["dof_torque"], label='measured')
    #     a.set( ylabel='Joint Torque [Nm]', title='Torque vs tims s')
    #     a.legend()
    #     # *************************   added for debug  by xiaoyu
    #     a = axs[3,0]
    #     if log["roller_pos1"]: a.plot(time, log["roller_pos1"], label='roller_1')
    #     a.set(title='1-roller measure pos')
    #     a = axs[3,1]
    #     if log["roller_pos2"]: a.plot(time, log["roller_pos2"], label='roller_2')
    #     a.set(title='2-roller measure pos')
    #             # plot contact forces
    #     a = axs[3, 2]
    #     if log["contact_forces_rollers"]:
    #         forces = np.array(log["contact_forces_rollers"])
    #         for i in range(forces.shape[1]):
    #             a.plot(time, forces[:, i], label=f'force {i}')
    #     a.set(xlabel='time [s]', ylabel='Forces z [N]', title='Vertical Contact forces rollers')
    #     a.legend()

    #     plt.show()

    # def _plot(self):
    #     # THIS PLOT THE JOINT POSITION, NO LONGER NEEDED
    #     nb_rows = 4
    #     nb_cols = 4
    #     fig, axs = plt.subplots(nb_rows, nb_cols)

    #     axs = axs.flatten()
    #     for key, value in self.state_log.items():
    #         time = np.linspace(0, len(value)*self.dt, len(value))
    #         break
    #     log= self.state_log

    #     counter = 0
    #     for joint_idx in range(len(log['dof_pos_target'][0])):
    #     # for joint_idx in range(14):
    #         # if joint_idx == 3 or joint_idx == 7:
    #         #     counter += 0
    #         #     continue
    #         commanded_joint_pos = []
    #         measured_joint_pos = []
    #         for i in range(len(time)):
    #             commanded_joint_pos.append(log['dof_pos_target'][i][joint_idx])
    #             measured_joint_pos.append(log['dof_pos'][i][joint_idx])
    #         axs[joint_idx-counter].plot(time, commanded_joint_pos, color = 'red', label = "commanded")
    #         axs[joint_idx-counter].plot(time, measured_joint_pos, color = 'blue', label = 'measured')
    #         axs[joint_idx-counter].legend()
    #     plt.tight_layout(rect = [0, 0.03, 1, 0.95])
    #     plt.show()

    # def _plot(self):
    #     nb_rows = 4
    #     nb_cols = 1


    #     plt.figure()
    #     fig, axs = plt.subplots(nb_rows, nb_cols)
    #     axs = axs.flatten()
    #     for key, value in self.state_log.items():
    #         time = np.linspace(0, len(value) * self.dt, len(value))
    #         break
    #     log = self.state_log

    #     counter = 0
    #     for leg in range(len(log['desired_contact'][0])):
    #         desired_contact = []
    #         robot_contact = []
    #         for i in range(len(time)):
    #             desired_contact.append(log['desired_contact'][i][leg])
    #             robot_contact.append(log['actual_contact'][i][leg])
    #         axs[leg - counter].plot(time, desired_contact, color='red', label="desired")
    #         axs[leg - counter].plot(time, robot_contact, color='blue', label="actual")
    #         axs[leg - counter].legend()

    #     # Show the current figure
    #     plt.show()   

    def _plot(self):
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value) * self.dt, len(value))
            break
        log = self.state_log

        #***********************************
        #    plot 1 :    contact
        #*********************************** 
        nb_rows = 4
        nb_cols = 1

        fig1, axs1 = plt.subplots(nb_rows, nb_cols, num=1)
        counter = 0
        for leg in range(len(log['desired_contact'][0])):
            desired_contact = []
            robot_contact = []
            for i in range(len(time)):
                desired_contact.append(log['desired_contact'][i][leg])
                robot_contact.append(log['actual_contact'][i][leg])
            axs1[leg - counter].plot(time, desired_contact, color='red', label="desired")
            axs1[leg - counter].plot(time, robot_contact, color='blue', label="actual", linestyle='dashed')
            axs1[leg - counter].legend()


        #***********************************
        #    plot 2 :    tilt angle  pos
        #*********************************** 
        num_roller = 2
        fig2, axs2 = plt.subplots(num_roller, 1, num=2)
        if log["tilt angle"]:
            angle = np.array(log["tilt angle"])
            for i in range(angle.shape[1]):
                if i == 0:
                    axs2[i].plot(time, angle[:, i], color='blue', label="tilt angle FL")
                    # axs2[i].plot(time, (angle[:, i] + 2 * np.pi) % (2 * np.pi), color='blue', label="roller FL")
                    axs2[i].set_title(' FL tilt angle: q vs t')
                    axs2[i].set_xlabel('t')
                    axs2[i].set_ylabel('q')
                    axs2[i].legend()
                if i == 1:
                    # axs2[i].plot(time, (angle[:, i] + 2 * np.pi) % (2 * np.pi), color='blue', label="roller FR")
                    axs2[i].plot(time, angle[:, i], color='blue', label="tilt angle FR")
                    axs2[i].set_title(' FR tilt angle: q vs t')
                    axs2[i].set_xlabel('t')
                    axs2[i].set_ylabel('q')
                    axs2[i].legend()


        #***********************************
        #    plot 3 :    base state
        #***********************************                 
        base_vel_plot_num = 3
        fig3, axs3 = plt.subplots(base_vel_plot_num, 1, num=3)

        axs3[0].plot(time, log["base_vel_x"], color='blue', label="base_vel_x")
        axs3[0].plot(time, log["command_x"], color='red', label="command_x", linestyle='dashed')

        axs3[1].plot(time, log["base_vel_y"], color='blue', label="base_vel_y")
        axs3[1].plot(time, log["command_y"], color='red', label="command_y", linestyle='dashed')

        axs3[2].plot(time, log["base_vel_yaw"], color='blue', label="base_vel_yaw")
        axs3[2].plot(time, log["command_yaw"], color='red', label="command_yaw", linestyle='dashed')

        # axs3[3].plot(time, np.array(log["base_vel_z"]), color='blue', label='base_vel_z')
        # axs3[3].set_ylabel('base_vel_z')

        for i in range(base_vel_plot_num):
            axs3[i].set_ylabel('v')
            axs3[i].set_xlabel('t')
            axs3[i].legend()
        


        # Show the new figure
        plt.tight_layout()
        plt.show()



    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")

    
    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()

    def plot_pred_true(self):
        log = self.state_log
        prediction = log['prediction']
        target = log['target']
        timesteps = list(range(len(prediction)))

        fig, axes = plt.subplots(10, 1, figsize = (10, 15))
        axes = axes.flatten()

        titles = ["lin_x(m/s)", "lin_y(m/s)", "lin_z(m/s)", "roll(rad/s)", "pitch(rad/s)", "yaw(rad/s)", "mass", "com displacement_x", "com_displacement_y", "com_displacement_z"]

        for i in range(10):  # Since each tensor has 9 elements
            values_prediction = [tensor[0, i].item() for tensor in prediction]  # Extract all i-th elements from prediction
            values_target = [tensor[0, i].item() for tensor in target]  # Extract all i-th elements from target

            if i == 2:
                combined_values = values_prediction + values_target
                min_val = min(combined_values)
                max_val = max(combined_values)
                range_val = max_val - min_val

                if range_val != 0:
                    values_prediction = [0.015 + 0.02 * ((y - min_val) / range_val) for y in values_prediction]
                    values_target = [0.015 + 0.02 * ((y - min_val) / range_val) for y in values_target]
                else:
                    values_prediction = [0.0 for _ in values_prediction]
                    values_target = [0.0 for _ in values_target]


            axes[i].plot(timesteps, values_prediction, color='blue', label='Prediction' if i == 0 else "")
            axes[i].plot(timesteps, values_target, color='red', label='Target' if i == 0 else "")

            axes[i].set_title(titles[i])
            axes[i].set_xlabel('Timestep')
            axes[i].set_ylabel('Value')

            if i == 0:
                axes[i].legend()
        plt.tight_layout()
        plt.show()

    def save_log(self, name):
        path = f'{name}.pkl'
        print(path)
        with open(path, "wb") as file:
            pkl.dump(self.state_log, file)
            print("SAVED!")