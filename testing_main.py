from __future__ import absolute_import
from __future__ import print_function

import os
from shutil import copyfile

from src.testing_simulation import Simulation
from src.generator import TrafficGenerator
from src.model import TestModel
from src.visualization import Visualization
from src.utils import import_test_configuration, set_sumo, set_test_path


if __name__ == "__main__":
    config = import_test_configuration(config_file='settings/testing_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    model_path, plot_path = set_test_path(config['models_path_name'], config['model_to_test'])

    Model = TestModel(
        input_dim=config['num_states'],
        model_path=model_path
    )

    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated']
    )

    Visualization = Visualization(
        plot_path, 
        dpi=96
    )
        
    Simulation = Simulation(
        Model,
        TrafficGen,
        sumo_cmd,
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_states'],
        config['num_actions'],
    )

    print('\n----- Test episode')
    total_reward, simulation_time, car_timings = Simulation.run(config['episode_seed'])  # run the simulation
    print('Simulation time:', simulation_time, 's')

    print('Total_delay:', Simulation.cumulative_total_wait())

    print("----- Testing info saved at:", plot_path)

    copyfile(src='settings/testing_settings.ini', dst=os.path.join(plot_path, 'testing_settings.ini'))

    Visualization.plot_timings(timings=car_timings)
