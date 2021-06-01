import traci
import numpy as np
import timeit

import os
from shutil import copyfile

from src import visualization
from src.generator import TrafficGenerator
from src.visualization import Visualization
from src.utils import import_test_configuration, set_sumo, set_test_path

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
        self._waiting_times = {}
        self._total_wait_time = 0

    def run(self, episode):
        timeit.default_timer()

        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)

        self._step = 0
        self._waiting_times = {}
        old_action = -1
        self._queue_length_episode = []
        self._reward_episode = []
        self._total_wait_time = 0
        current_total_wait = 0
        while self._step < self._max_steps:
            current_total_wait = self._collect_waiting_times()

            action = self._choose_action(self._step, old_action)

            if self._step != 0 and old_action != action:
                if old_action == 4:
                    self._set_yellow_phase(3)
                else:
                    self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            self._set_green_phase(action)
            self._simulate(self._green_duration)

            old_action = action

        self._total_wait_time = current_total_wait
        traci.close()

        return 0

    def _simulate(self, steps_todo):
        if (self._step + steps_todo) >= self._max_steps:
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._step += 1  # update the step counter
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
                if car_id in self._waiting_times:  # a car that was tracked has cleared the intersection
                    del self._waiting_times[car_id]
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time

    def _choose_action(self, current_step, old_action):
        t = current_step % 126
        if 0 <= t < 40:
            return 0
        elif 40 <= t < 63:
            return 1
        elif 63 <= t < 103:
            return 2
        elif 103 <= t < 126:
            return 3


    def _set_yellow_phase(self, old_action):
        yellow_phase_code = old_action * 2 + 1  # obtain the yellow phase code, based on the old action (ref on environment.net.xml)
        traci.trafficlight.setPhase("TL", yellow_phase_code)

    def _set_green_phase(self, action_number):
        if action_number == 0:
            traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)

    def _get_queue_length(self):
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        queue_length = halt_N + halt_S + halt_E + halt_W
        return queue_length

    def cumulative_total_wait(self):
        return np.sum(self._queue_length_episode)

    @property
    def queue_length_episode(self):
        return self._queue_length_episode

    @property
    def reward_episode(self):
        return self._reward_episode


def make_benchmark(n_cars, episode_count, seed_shift):
    config = import_test_configuration(config_file='settings/testing_settings.ini')
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
    plot_path = "benchmark"

    visualization = Visualization(
        plot_path,
        dpi=96
    )
    raw_data = []
    avg_delay = 0
    for i in range(episode_count):
        simulation.run(config['episode_seed'] + i + seed_shift)
        raw_data.append(simulation.queue_length_episode)
        delay = simulation.cumulative_total_wait()
        avg_delay += delay / episode_count

    to_graph_raw = []
    for i in range(episode_count):
        s = 0
        tmp = []
        for j in range(len(raw_data[i]) - 100):
            s = sum(raw_data[i][j:j + 100]) / 100
            tmp.append(s)
        to_graph_raw.append(tmp)
    to_graph = [0.0] * (len(to_graph_raw[0]) - 100)
    for i in range(episode_count):
        for j in range(len(to_graph)):
            to_graph[j] += (to_graph_raw[i][j] / episode_count)
    visualization.save_data_and_plot(data=to_graph, filename='AQL_stl' + str(n_cars), xlabel='Step',
                                     ylabel='avg queue length over 100 steps')

    with open("benchmark/total_delay.txt", 'a') as f:
        f.write("{}, {}, {}, resulted: {}\n".format(n_cars, episode_count, seed_shift, avg_delay))
    print('Average delay:', avg_delay)
