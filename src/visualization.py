import matplotlib.pyplot as plt
import os
import numpy as np


class Visualization:
    def __init__(self, path, dpi):
        self._path = path
        self._dpi = dpi


    def save_data_and_plot(self, data, filename, xlabel, ylabel):
        """
        Produce a plot of performance of the agent over the session and save the related data to txt
        """
        min_val = min(data)
        max_val = max(data)

        plt.rcParams.update({'font.size': 24})  # set bigger font size

        plt.plot(data)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.margins(0)
        plt.ylim(min_val - 0.05 * abs(min_val), max_val + 0.05 * abs(max_val))
        fig = plt.gcf()
        fig.set_size_inches(20, 11.25)
        fig.savefig(os.path.join(self._path, 'plot_'+filename+'.png'), dpi=self._dpi)
        plt.close("all")

        with open(os.path.join(self._path, 'plot_'+filename + '_data.txt'), "w") as file:
            for value in data:
                file.write("%s\n" % value)

    def plot_together_aql(self, models_to_test_str, n_cars, filename, xlabel, ylabel):
        models_to_test = models_to_test_str.split()
        data = []
        for model_id in models_to_test:
            with open("models/model_"+model_id+"/plot_AQL_"+str(n_cars)+"_data.txt", 'r') as f:
                tmp = []
                datum = f.readline()
                while datum:
                    tmp.append(datum)
                    datum = f.readline()
                data.append(tmp)
        data = np.array(data).astype(float)
        min_val = np.amin(data)
        max_val = np.amax(data)

        plt.rcParams.update({'font.size': 24})  # set bigger font size
        for i in range(len(data)):
            plt.plot(data[i], label=models_to_test[i])
        plt.legend()
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.margins(0)
        plt.ylim(min_val - 0.05 * abs(min_val), max_val + 0.05 * abs(max_val))
        fig = plt.gcf()
        fig.set_size_inches(20, 11.25)
        fig.savefig(os.path.join(self._path, 'plot_'+filename+'.png'), dpi=self._dpi)
        plt.close("all")


    def plot_timings(self, timings):
        count, bins, ignored = plt.hist(timings, bins=50)
        plt.savefig(os.path.join(self._path, 'plot_timings.png'), dpi=self._dpi)


def weib(x, n, a):
    return (a / n) * (x / n) ** (a - 1) * np.exp(-(x / n) ** a)
