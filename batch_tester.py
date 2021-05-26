from __future__ import absolute_import
from __future__ import print_function

from src.testing_simulation import Simulation
from src.generator import TrafficGenerator
from src.model import TestModel
from src.visualization import Visualization
from src.utils import import_test_configuration, set_sumo
from src.benchmark_stl import make_benchmark


def test(models_to_test_str, n_cars, episode_count, seed_shift):
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
        avg_delay = 0
        raw_data = []
        for i in range(episode_count):
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

            # print("Episode: {} of {}. Model id: {}".format(i + 1, episode_count, model_id))
            total_reward, simulation_time, car_timings = simulation.run(config['episode_seed']+i+seed_shift)
            raw_data.append(simulation.queue_length_episode)
            delay = simulation.cumulative_total_wait()
            print(delay)
            avg_delay += delay / episode_count
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
        with open(model_path+"/total_delay.txt", 'a') as f:
            f.write("{}, {}, {}, resulted: {} \n".format(n_cars, episode_count, seed_shift, avg_delay))
        print('Model: {};Average total delay: {}'.format(model_id, avg_delay))
        # if i == n-1:
        #     print("----- Testing info saved at:", plot_path)
        #
        #     copyfile(src='testing_settings.ini', dst=os.path.join(plot_path, 'testing_settings.ini'))
        #
        #     visualization.save_data_and_plot(data=simulation.reward_episode, filename='reward', xlabel='Action step',
        #                                      ylabel='Reward')
        #     visualization.save_data_and_plot(data=simulation.queue_length_episode, filename='queue', xlabel='Step',
        #                                      ylabel='Queue length (vehicles)')
    print("-" * 250)


def group_aql(models_to_test_str, n_cars, with_benchmark):
    visualization = Visualization(
        "test_results",
        dpi=96
    )
    visualization.plot_together_aql(models_to_test_str, n_cars, models_to_test_str + "_together_" + str(n_cars), "step", "AQL", with_benchmark)


if __name__ == "__main__":
    test("1 2 3 4 5 6 7", 1000, 5, 0)
    test("1 2 3 4 5 6 7", 2500, 5, 0)
    #
    # test("1 4 7 8 9 10", 1000, 5, 5)
    # test("1 4 7 8 9 10", 2500, 5, 5)
    #
    # group_aql("1 4 7", 1000, 0)
    # group_aql("1 4 7", 2500, 0)
    #
    # make_benchmark(1000, 10, 10)

    # test("4 9", 2500, 5, 10)
    # make_benchmark(2500, 5, 10)
    #
    # test("1 4", 1000, 5, 10)
    # test("1 4", 2500, 5, 10)
    # test("12", 2500, 10, 10)
    # test("12", 1000, 10, 10)
    # group_aql("1 4 7 9", 1000, 0)
    # group_aql("4 9", 2500, 1)
    # test("4 9 12", 3500, 5, 10)
    # make_benchmark(3500, 5, 10)
    # group_aql("4 9 12", 3500, 1)
