import itertools
import numpy as np
from tqdm import tqdm
from pathlib import Path
import pandas as pd

from controllers import *
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator

# Simple brute force PID tuner
class PIDTuner:
    def __init__(self, k_p_range, k_i_range, k_d_range):
        self.k_p_range = k_p_range
        self.k_i_range = k_i_range
        self.k_d_range = k_d_range
        self.best_params = None
        self.best_cost = float('inf')

    def tune(self, data_path, num_segs=100):
        tinyphysicsmodel = TinyPhysicsModel("./models/tinyphysics.onnx", debug=False)
        param_combinations = itertools.product(self.k_p_range, self.k_i_range, self.k_d_range)

        for k_p, k_i, k_d in tqdm(param_combinations, total=len(self.k_p_range) * len(self.k_i_range) * len(self.k_d_range)):
            controller = PIDController(k_p=k_p, k_i=k_i, k_d=k_d)
            total_cost = self.evaluate_controller(tinyphysicsmodel, controller, data_path, num_segs)
            print(f"\ncurrent: {(k_p, k_i, k_d)} : {total_cost}")
            print(f"best: {self.best_params} : {self.best_cost}")

            if total_cost < self.best_cost:
                self.best_params = (k_p, k_i, k_d)
                self.best_cost = total_cost

        print(f"Best parameters: k_p={self.best_params[0]}, k_i={self.best_params[1]}, k_d={self.best_params[2]}")
        print(f"Best total cost: {self.best_cost}")

    def evaluate_controller(self, tinyphysicsmodel, controller, data_path, num_segs):
        data_path = Path(data_path)
        assert data_path.is_dir(), "data_path should be a directory"

        costs = []
        files = sorted(data_path.iterdir())[:num_segs]
        for data_file in tqdm(files, total=len(files)):
            sim = TinyPhysicsSimulator(tinyphysicsmodel, str(data_file), controller=controller, debug=False)
            cost = sim.rollout()
            costs.append(cost)

        costs_df = pd.DataFrame(costs)
        total_cost = np.mean(costs_df['total_cost'])
        return total_cost

if __name__ == "__main__":
    # start with wide range, large intervals low segment count
    # decrease range and intervals, increase segment count as you get closer to an optimal value
    k_p_range = np.arange(0.00, 0.1, 0.01)
    k_i_range = np.arange(0.00, 0.12, 0.01)
    k_d_range = np.arange(-0.04, 0.01, 0.02)

    tuner = PIDTuner(k_p_range, k_i_range, k_d_range)
    tuner.tune("./data/", 10)