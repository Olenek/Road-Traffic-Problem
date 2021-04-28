from __future__ import absolute_import
from __future__ import print_function

import os
import re
import datetime
from shutil import copyfile

from src.training_simulation import Simulation
from src.generator import TrafficGenerator
from src.memory import Memory
from src.model import TrainModel
from src.visualization import Visualization
from src.utils import import_train_configuration, set_sumo, set_train_path

if __name__ == "__main__":

    for file in os.listdir("training_batch"):
        print(file)
        config = import_train_configuration(config_file="training_batch/" + file)
        sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
        path = set_train_path(config['models_path_name'])

        model = TrainModel(
            config['num_layers'],
            config['width_layers'],
            config['batch_size'],
            config['learning_rate'],
            config['num_states'],
            config['num_actions'],
            config['optimizer']
        )

        memory = Memory(
            config['memory_size_max'],
            config['memory_size_min']
        )

        traffic_gen = TrafficGenerator(
            config['max_steps'],
            config['n_cars_generated']
        )

        visualization = Visualization(
            path,
            dpi=96
        )

        simulation = Simulation(
            model,
            memory,
            traffic_gen,
            sumo_cmd,
            config['gamma'],
            config['max_steps'],
            config['green_duration'],
            config['yellow_duration'],
            config['num_states'],
            config['num_actions'],
            config['training_epochs'],
            config['is_greedy']
        )

        episode = 0
        timestamp_start = datetime.datetime.now()
        file_postfix = re.match(r'\w+_\w+_(.*)\.\w+', file).groups()[0]
        with open("batch_results/_" + file_postfix + ".txt", 'a') as f:
            if config['is_greedy']:
                while episode < config['total_episodes']:
                    f.write('\n----- Episode {} of {}'.format(str(episode + 1), str(config['total_episodes'])))
                    epsilon = 1.0 - (episode / config[
                        'total_episodes'])  # set the epsilon for this episode according to epsilon-greedy policy
                    simulation_time, training_time = simulation.run(episode, epsilon)  # run the simulation
                    f.write(
                        'Simulation time: {}s - Training time: '
                        '{}s - Total: {}s'.format(simulation_time, training_time,
                                                  round(simulation_time + training_time, 1)))
                    episode += 1
            else:
                while episode < config['total_episodes']:
                    f.write('\n----- Episode {} of {}'.format(str(episode + 1), str(config['total_episodes'])))

                    simulation_time, training_time = simulation.run(episode, 0)  # run the simulation
                    f.write(
                        'Simulation time: {}s - Training time: '
                        '{}s - Total: {}s'.format(simulation_time, training_time,
                                                  round(simulation_time + training_time, 1)))
                    episode += 1

            model.save_model(path)

            copyfile(src="training_batch/" + file, dst=os.path.join(path, 'settings/training_settings.ini'))

            visualization.save_data_and_plot(data=simulation.reward_store, filename='reward', xlabel='Episode',
                                             ylabel='Cumulative negative reward')
            visualization.save_data_and_plot(data=simulation.cumulative_wait_store, filename='delay', xlabel='Episode',
                                             ylabel='Cumulative delay (s)')
            visualization.save_data_and_plot(data=simulation.avg_queue_length_store, filename='queue', xlabel='Episode',
                                             ylabel='Average queue length (vehicles)')
