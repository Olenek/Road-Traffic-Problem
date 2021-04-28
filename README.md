# Road Traffic Problem
A possible solution to the road traffic problem with deep Q-learning. 

This project is done as a part of Higher School of Economics DSBA 2nd year course.

## Requirements
The code requires a number of Python modules, including but not limited to numpy, tensorflow and pyplot. One may choose to install them separately, however an easier way is to use conda's tf_gpu environment.

```
conda create --name tf_gpu
activate tf_gpu
conda install tensorflow-gpu
```

Apart from Python3 and its modules, SUMO was used to simulate traffic on an intersection. It can be downloaded from the [official site](https://sumo.dlr.de/docs/Downloads.php).

This project uses Python 3.8, tensorflow-gpu 2.4, SUMO 1.9.0

## Conducting training procedure.

There are two ways of training agents:
1. Train agents one-by-one.
2. Train agents in batches.

The first way can be done by changing config file `training_settings.ini` in `settings/` directory, followed by running `training_main.py` script.

Alternatively, one may create numerous config files by the name `training_settings_x.ini`, where `x` is an integer number and place these files in `training_batch/` directory. Script `batch_trainer.py` does the rest.

## Conducting testing procedure.

Similarly, it is possible to test agents one-by-one by running `testing_main.py` and editing the corresponding config file in the `settings/` directory.

If one wishes to test many agents at once, it is advisory to run `batch_tester.py` and following the command prompt.

## Results.
Results and details of this project can be observed in the report as soon as it will be published online, or as soon as you get a copy.

### Related projects.
https://github.com/rdulmina/Traffic-Control-Reinforcement-Learing-Agent \
https://github.com/raymelon/TrafficLightNeuralNetwork \
https://github.com/AndreaVidali/Deep-QLearning-Agent-for-Traffic-Signal-Control 
