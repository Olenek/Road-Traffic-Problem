import traci
import numpy as np
import timeit

import os
from shutil import copyfile

from generator import TrafficGenerator
from src.visualization import Visualization
from utils import import_test_configuration, set_sumo, set_test_path

# phase codes based on environment.net.xml
PHASE_NS_GREEN = 0  # action 0 code 00
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # action 1 code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action 2 code 10
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action 3 code 11
PHASE_EWL_YELLOW = 7


class Simulation:
    def __init__(self, TrafficGen, sumo_cmd, max_steps, green_duration, yellow_duration, num_states, num_actions):
        self._TrafficGen = TrafficGen
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_episode = []
        self._queue_length_episode = []
        self._poor_calibration = False

    def _set_poor_calibration(self):
        self._poor_calibration = True

    def run(self, episode):
        """
        Runs the testing simulation
        """
        start_time = timeit.default_timer()

        # first, generate the route file for this simulation and set up sumo
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # inits
        self._step = 0
        self._waiting_times = {}
        old_total_wait = 0
        old_action = -1 # dummy init

        while self._step < self._max_steps:


            # calculate reward of previous action: (change in cumulative waiting time between actions)
            # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
            current_total_wait = self._collect_waiting_times()
            reward = old_total_wait - current_total_wait

            # choose the light phase to activate, based on the current state of the intersection
            action = self._choose_action(self._step, old_action)

            # if the chosen phase is different from the last phase, activate the yellow phase

            # execute the phase selected before
            self._set_green_phase(action)
            self._simulate(self._green_duration)

            # saving variables for later & accumulate reward
            old_action = action
            old_total_wait = current_total_wait

            self._reward_episode.append(reward)

        total_reward = np.sum(self._reward_episode)
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        return total_reward, simulation_time


    def _simulate(self, steps_todo):
        """
        Proceed with the simulation in sumo
        """
        if (self._step + steps_todo) >= self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._step += 1 # update the step counter
            steps_todo -= 1
            queue_length = self._get_queue_length()
            self._queue_length_episode.append(queue_length)


    def _collect_waiting_times(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                self._waiting_times[car_id] = wait_time
            else:
                if car_id in self._waiting_times: # a car that was tracked has cleared the intersection
                    del self._waiting_times[car_id]
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time


    def _choose_action(self, current_step, old_action):
        """
        Pick the best action known based on the current state of the env
        """
        t = current_step%126
        if 0 <= t < 40:
            return 0
        elif 44 <= t < 59:
            return 1
        elif 63 <= t < 103:
            return 2
        elif 107 <= t < 122:
            return 3
        else:
            self._set_yellow_phase(old_action)
            self._simulate(self._yellow_duration)


    def _set_yellow_phase(self, old_action):
        """
        Activate the correct yellow light combination in sumo
        """
        yellow_phase_code = old_action * 2 + 1 # obtain the yellow phase code, based on the old action (ref on environment.net.xml)
        traci.trafficlight.setPhase("TL", yellow_phase_code)


    def _set_green_phase(self, action_number):
        """
        Activate the correct green light combination in sumo
        """
        if action_number == 0:
            traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)


    def _get_queue_length(self):
        """
        Retrieve the number of cars with speed = 0 in every incoming lane
        """
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        queue_length = halt_N + halt_S + halt_E + halt_W
        return queue_length


    @property
    def queue_length_episode(self):
        return self._queue_length_episode


    @property
    def reward_episode(self):
        return self._reward_episode


def make_benchmark(episode_count, n_cars):
    config = import_test_configuration(config_file='../settings/testing_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])


    traffic_gen = TrafficGenerator(
        config['max_steps'],
        n_cars
    )

    simulation = Simulation(
        traffic_gen,
        sumo_cmd,
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_states'],
        config['num_actions'],
    )
    raw_data = []
    avg_reward = 0
    for i in range(episode_count):
        print("Episode: {} of {}.".format(i + 1, episode_count))
        total_reward, simulation_time = simulation.run(config['episode_seed']+i)
        avg_reward += total_reward
        raw_data.append(simulation.queue_length_episode)
    avg_reward /= episode_count
    to_graph_raw = []
    for i in range(episode_count):
        s = 0
        tmp = []
        for j in range(len(raw_data[i]) - 100):
            s = sum(raw_data[i][j:j + 100]) / 100
            tmp.append(s)
        to_graph_raw.append(tmp)
    to_graph = [0] * 5400
    for i in range(episode_count):
        for j in range(len(to_graph_raw[i])):
            to_graph[j] += to_graph_raw[i][j] / episode_count

    visualization = Visualization(
        "test_results",
        dpi=96
    )
    visualization.save_data_and_plot(data=to_graph, filename='AQL_bench_' + str(n_cars), xlabel='Step',
                                     ylabel='avg queue length over 100 steps')

    print('Average reward:', avg_reward)
