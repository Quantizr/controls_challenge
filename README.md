# Comma Controls Challenge!

## Submission
Tried a whole lot of things ranging from LSTMs to MPC and weird combinations of everything. Some had much lower lateral acceleraton cost, some had much lower jerk costs, but in terms of total cost, PID (which I was using as a baseline) remained on top out of the methods which would run in a reasonable amount of time. Report is in the repository as `pid_report.html`.

Essentially brute forced optimal PID values using the code in `pid_tuner.py`. 

`k_p = 0.044`, `k_i = 0.10`, `k_d = -0.035` was found to be near optimal.

| Controller | LatAccel Cost | Jerk Cost | Total Cost |
|------------|---------------|-----------|------------|
| Baseline   | 9.412         | 32.167    | 79.225     |
| Test (PID) | 1.564         | 26.422    | 34.241     |

I'll try to organize and upload some of my other attempts in another branch at some point soon.

## Rest of the original README
![Car](./imgs/car.jpg)

Machine learning models can drive cars, paint beautiful pictures and write passable rap. But they famously suck at doing low level controls. Your goal is to write a good controller. This repo contains a model that simulates the lateral movement of a car, given steering commands. The goal is to drive this "car" well for a given desired trajectory.


## Geting Started
We'll be using a synthetic dataset based on the [comma-steering-control](https://github.com/commaai/comma-steering-control) dataset for this challenge. These are actual routes with actual car and road states.

```
# download necessary dataset (~0.6G)
bash ./download_dataset.sh

# install required packages
pip install -r requirements.txt

# test this works
python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data/00000.csv --debug --controller simple
```

There are some other scripts to help you get aggregate metrics: 
```
# batch Metrics of a controller on lots of routes
python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --controller simple

# generate a report comparing two controllers
python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --test_controller simple --baseline_controller open

```
You can also use the notebook at [`experiment.ipynb`](https://github.com/commaai/controls_challenge/blob/master/experiment.ipynb) for exploration.

## TinyPhysics
This is a "simulated car" that has been trained to mimic a very simple physics model (bicycle model) based simulator, given realistic driving noise. It is an autoregressive model similar to [ML Controls Sim](https://blog.comma.ai/096release/#ml-controls-sim) in architecture. It's inputs are the car velocity (`v_ego`), forward acceleration (`a_ego`), lateral acceleration due to road roll (`road_lataccel`), current car lateral acceleration (`current_lataccel`) and a steer input (`steer_action`) and predicts the resultant lateral acceleration fo the car.


## Controllers
Your controller should implement an [update function](https://github.com/commaai/controls_challenge/blob/1a25ee200f5466cb7dc1ab0bf6b7d0c67a2481db/controllers.py#L2) that returns the `steer_action`. This controller is then run in-loop, in the simulator to autoregressively predict the car's response.


## Evaluation
Each rollout will result in 2 costs:
- `lataccel_cost`: $\dfrac{\Sigma(actual\\_lat\\_accel - target\\_lat\\_accel)^2}{steps} * 100$

- `jerk_cost`: $\dfrac{\Sigma((actual\\_lat\\_accel\_t - actual\\_lat\\_accel\_{t-1}) / \Delta t)^2}{steps - 1} * 100$

It is important to minimize both costs. `total_cost`: $(lataccel\\_cost *5) + jerk\\_cost$

## Submission
Run the following command, and send us a link to your fork of this repo, and the `report.html` this script generates.
```
python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 5000 --test_controller <insert your controller name> --baseline_controller simple
```

## Work at comma
Like this sort of stuff? You might want to work at comma!
https://www.comma.ai/jobs
