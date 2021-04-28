from __future__ import absolute_import
from __future__ import print_function

from src.testing_simulation import Simulation
from src.generator import TrafficGenerator
from src.model import TestModel
from src.visualization import Visualization
from src.utils import import_test_configuration, set_sumo


def test_td(models_to_test_str, n_cars, episode_count):
    config = import_test_configuration(config_file='settings/testing_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])

    traffic_gen = TrafficGenerator(
        config['max_steps'],
        n_cars
    )

    models_to_test = models_to_test_str.split()
    models_tr = {}
    for model_id in models_to_test:
        model_path = "models/model_" + model_id
        plot_path = "models/model_" + model_id

        visualization = Visualization(
            plot_path,
            dpi=96
        )
        model = TestModel(
            input_dim=config['num_states'],
            model_path=model_path
        )

        simulation = Simulation(
            model,
            traffic_gen,
            sumo_cmd,
            config['max_steps'],
            config['green_duration'],
            config['yellow_duration'],
            config['num_states'],
            config['num_actions'],
        )

        avg_reward = 0
        for i in range(episode_count):
            print('\n----- Test episode: {}, model to test: {}'.format(i + 1, model_id))
            total_reward, simulation_time, car_timings = simulation.run(config['episode_seed']+i)  # run the simulation
            avg_reward += total_reward
        avg_reward /= episode_count
        models_tr[model_id] = avg_reward
    print("-" * 250)
    print("Test results: {}\n\n".format(models_tr))

    # if i == n-1:
    #     print("----- Testing info saved at:", plot_path)
    #
    #     copyfile(src='testing_settings.ini', dst=os.path.join(plot_path, 'testing_settings.ini'))
    #
    #     visualization.save_data_and_plot(data=simulation.reward_episode, filename='reward', xlabel='Action step',
    #                                      ylabel='Reward')
    #     visualization.save_data_and_plot(data=simulation.queue_length_episode, filename='queue', xlabel='Step',
    #                                      ylabel='Queue length (vehicles)')


def test_aql(models_to_test_str, n_cars, episode_count):
    config = import_test_configuration(config_file='settings/testing_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])

    traffic_gen = TrafficGenerator(
        config['max_steps'],
        n_cars
    )

    models_to_test = models_to_test_str.split()
    for model_id in models_to_test:
        model_path = "models/model_" + model_id
        plot_path = "models/model_" + model_id

        visualization = Visualization(
            plot_path,
            dpi=96
        )
        model = TestModel(
            input_dim=config['num_states'],
            model_path=model_path
        )

        simulation = Simulation(
            model,
            traffic_gen,
            sumo_cmd,
            config['max_steps'],
            config['green_duration'],
            config['yellow_duration'],
            config['num_states'],
            config['num_actions'],
        )

        raw_data = []
        for i in range(episode_count):
            print("Episode: {} of {}. Model id: {}".format(i + 1, episode_count, model_id))
            total_reward, simulation_time, car_timings = simulation.run(config['episode_seed'])
            raw_data.append(simulation.queue_length_episode)
        to_graph_raw = []
        for i in range(episode_count):
            s = 0
            tmp = []
            for j in range(len(raw_data[i]) - 100):
                s = sum(raw_data[i][j:j + 100]) / 100
                tmp.append(s)
            to_graph_raw.append(tmp)
        to_graph = [0] * (len(to_graph_raw[0]) - 100)
        for i in range(episode_count):
            for j in range(len(to_graph_raw[i]) - 100):
                to_graph[j] += to_graph_raw[i][j] / episode_count
        visualization.save_data_and_plot(data=to_graph, filename='AQL_' + str(n_cars), xlabel='Step',
                                         ylabel='avg queue length over 100 steps')
    print("-" * 250)


def group_aql(models_to_test_str, n_cars):
    visualization = Visualization(
        "test_results",
        dpi=96
    )
    visualization.plot_together_aql(models_to_test_str, n_cars, "together" + str(n_cars), "step", "AQL")


if __name__ == "__main__":
    # test_td("01 02 03 04 05 06 07", 1000, 5)
    # test_td("01 02 03 04 05 06 07", 2500, 5)
    #
    # test_aql("01 04 07", 1000, 1)
    # group_aql("01 04 07", 1000)
    #
    # test_aql("01 04 07", 2500, 1)
    # group_aql("01 04 07", 2500)

    test_td("01 04 07 9 10 11", 1000, 5)
    test_td("01 04 07 9 10 11", 2500, 5)


