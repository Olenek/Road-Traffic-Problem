import traci
import numpy as np
import random
import timeit
import os

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
    def __init__(self, Model, TrafficGen, sumo_cmd, max_steps, green_duration, yellow_duration, num_states,
                 num_actions):
        self._Model = Model
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
        """
        Runs a single episode of simulation
        """
        start_time = timeit.default_timer()

        # generate the routefile for the simulation and set up sumo
        car_timings = self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        # print("Simulating...")

        self._step = 0
        self._waiting_times = {}
        self._queue_length_episode = []
        old_total_wait = 0
        current_total_wait = 0
        old_action = -1  # dummy inits
        self._reward_episode = []
        threshold = 0.75  # threshold for initiating STL cycle
        counter = 0
        while self._step < self._max_steps:
            current_state = self._get_state()
            current_total_wait = self._collect_waiting_times()

            allow_stl = sum(current_state) / len(current_state) >= threshold  # decide if to allow an STL cycle
            action = self._choose_action(current_state, allow_stl=allow_stl)
            if action != 4:  # if not STL cycle
                # if the chosen phase is different from the last phase, activate the yellow phase
                if self._step != 0 and old_action != action:
                    if old_action == 4:
                        self._set_yellow_phase(3)
                    else:
                        self._set_yellow_phase(old_action)
                    self._simulate(self._yellow_duration)

                # execute the phase selected before
                self._set_green_phase(action)
                self._simulate(self._green_duration)

                # saving variables for later & accumulate reward
                old_state = current_state
                old_action = action
                old_total_wait = current_total_wait
            else:
                counter += 1
                initial_step = self._step
                old_total_wait = current_total_wait

                while self._step < (initial_step + 126) and self._step < self._max_steps:
                    # choose the light phase to activate, based on the current state of the intersection
                    action = self._choose_stl_action(self._step - initial_step, old_action)
                    if self._step != 0 and old_action != action:
                        if old_action == 4:
                            self._set_yellow_phase(3)
                        else:
                            self._set_yellow_phase(old_action)
                        self._simulate(self._yellow_duration)
                    # execute the phase selected before
                    self._set_green_phase(action)
                    self._simulate(self._green_duration)

                    # saving variables for later & accumulate reward
                    old_state = current_state
                    old_action = action
                old_action = 4

        total_reward = np.sum(self._reward_episode)
        self._total_wait_time = current_total_wait
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)
        # print("Made {} stl cycles".format(counter))

        return total_reward, simulation_time, car_timings

    def _simulate(self, steps_todo):
        """
        Simulates 'steps_todo' steps
        """
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
        Return current total waiting time of all incoming cars
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

    def _choose_action(self, state, allow_stl):
        """
        Chooses best q-value action
        """
        prediction = self._Model.predict_one(state)
        if not allow_stl and np.argmax(prediction) == 4:
            try:
                np.argsort(prediction[0])[::-1][1]
            except:
                print(prediction)
            return np.argsort(prediction[0])[::-1][1]  # the second best if not allowed to stl
        return np.argmax(prediction)  # the best action given the current state

    def _choose_stl_action(self, current_step, old_action):
        """
        Chooses action according to STL policy
        """
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
        """
        Sets a yellow phase for the traffic light
        """
        yellow_phase_code = old_action * 2 + 1  # obtain the yellow phase code, based on the old action (ref on environment.net.xml)
        traci.trafficlight.setPhase("TL", yellow_phase_code)

    def _set_green_phase(self, action_number):
        """
        Sets a green phase for the traffic light
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
        Returns the current total queue length in the simulation
        """
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        queue_length = halt_N + halt_S + halt_E + halt_W
        return queue_length

    def _get_state(self):
        """
        Returns 1-d array-state according to uneven discretisation policy
        """
        state = np.zeros(self._num_states)
        car_list = traci.vehicle.getIDList()

        for car_id in car_list:
            lane_pos = traci.vehicle.getLanePosition(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)
            lane_pos = 750 - lane_pos
            if lane_pos < 7:
                lane_cell = 0
            elif lane_pos < 14:
                lane_cell = 1
            elif lane_pos < 21:
                lane_cell = 2
            elif lane_pos < 28:
                lane_cell = 3
            elif lane_pos < 40:
                lane_cell = 4
            elif lane_pos < 60:
                lane_cell = 5
            elif lane_pos < 100:
                lane_cell = 6
            elif lane_pos < 160:
                lane_cell = 7
            elif lane_pos < 400:
                lane_cell = 8
            elif lane_pos <= 750:
                lane_cell = 9

            # finding the lane where the car is located 
            # x2TL_3 are the "turn left only" lanes
            if lane_id == "W2TL_0" or lane_id == "W2TL_1" or lane_id == "W2TL_2":
                lane_group = 0
            elif lane_id == "W2TL_3":
                lane_group = 1
            elif lane_id == "N2TL_0" or lane_id == "N2TL_1" or lane_id == "N2TL_2":
                lane_group = 2
            elif lane_id == "N2TL_3":
                lane_group = 3
            elif lane_id == "E2TL_0" or lane_id == "E2TL_1" or lane_id == "E2TL_2":
                lane_group = 4
            elif lane_id == "E2TL_3":
                lane_group = 5
            elif lane_id == "S2TL_0" or lane_id == "S2TL_1" or lane_id == "S2TL_2":
                lane_group = 6
            elif lane_id == "S2TL_3":
                lane_group = 7
            else:
                lane_group = -1

            if 1 <= lane_group <= 7:
                car_position = int(str(lane_group) + str(lane_cell))  # composition a number in interval 0-79
                valid_car = True
            elif lane_group == 0:
                car_position = lane_cell
                valid_car = True
            else:
                valid_car = False  # flag for not detecting cars crossing the intersection or driving away from it

            if valid_car:
                state[car_position] = 1

        return state

    def cumulative_total_wait(self):
        """
        Returns the sum of all waiting times throughout the episode
        car in a queue = car is waiting -> queue_length = increment in waiting time per timestep.
        """
        return np.sum(self._queue_length_episode)

    @property
    def queue_length_episode(self):
        return self._queue_length_episode

    @property
    def reward_episode(self):
        return self._reward_episode
