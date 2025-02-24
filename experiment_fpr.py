from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import pickle
from concurrent.futures import ProcessPoolExecutor
from time import time
from typing import Literal, cast
from source.tree_afe_si import TreeAFE_si
from sicore import SelectiveInferenceResult
from dataclasses import dataclass, field
from typing_extensions import Self
import numpy as np
import pandas as pd
import sympy as sp

def create_cov_matrix(size, rho):
    matrix = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(abs(i - j))
        matrix.append(row)
    cov = np.power(rho, matrix)
    return cov

def run_experiment(seed, n, d, noise_type):

    rng = np.random.default_rng(seed)

    # generate design matrix
    X = rng.normal(size=(n, d))

    # convert to pandas dataframe
    symbol_cols = [sp.symbols(f'x{i}', real=True) for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=symbol_cols)

    # generate response variable
    if noise_type == 'iid':
        y = rng.normal(size=(n,))
        cov = np.eye(n)

    elif noise_type == 'corr':
        rho = 0.5
        cov = create_cov_matrix(n, rho)
        y = rng.multivariate_normal(mean=np.zeros(n), cov=cov)

    # setup hyperparameters for tree based AFE
    max_depth = 5
    num_generate_node = 2
    num_extract_node = 1
    gamma = 2
    transformation_list = ['sin', 'exp', 'mul', 'sqrt']

    # generate tree based AFE object
    afe_si = TreeAFE_si(
        seed=seed,
        max_depth=max_depth,
        num_generate_node=num_generate_node,
        num_extract_node=num_extract_node,
        gamma=gamma,
        transformation_list=transformation_list
    )

    # run the tree based AFE
    df_fe_obs = afe_si.apply_afe(df, y, cov)

    # For DS
    df_fe_alg = afe_si.apply_afe_ds(df, y, cov)

    # select stastical test index (si and ds)
    index_si = rng.choice(list(range(df_fe_obs.shape[1])))
    index_ds = rng.choice(list(range(df_fe_alg.shape[1])))

    try:

        start_proposed = time()
        result = afe_si.inference(
            index_si,
            inference_mode = 'parametric'
        )
        result = cast(SelectiveInferenceResult, result)
        elapsed_proposed = time() - start_proposed

        start_oc = time()
        result_oc = afe_si.inference(
            index_si,
            inference_mode = 'over_conditioning'
        )
        # over-conditioning p-value
        oc_p_value = cast(float, result_oc.p_value)
        elapsed_oc = time() - start_oc

        ds_p_value = afe_si.inference_ds(index_ds)

    except Exception as e:  # noqa: BLE001
        print(e)
        return None

    else:
        return result, oc_p_value, ds_p_value, elapsed_proposed

@dataclass
class Results:
    """Results dataclass for the data analysis pipeline."""

    results: list[SelectiveInferenceResult] = field(default_factory=list)
    oc_p_values: list[float] = field(default_factory=list)
    times: list[float] = field(default_factory=list)

    def __add__(self, other: Results) -> Results:
        """Take union of two results."""
        return Results(
            results=self.results + other.results,
            oc_p_values=self.oc_p_values + other.oc_p_values,
            times=self.times + other.times,
        )

    def __iadd__(self, other: Self) -> Self:
        """Take union of two results in place."""
        self.results += other.results
        self.oc_p_values += other.oc_p_values
        self.times += other.times
        return self

    def __len__(self) -> int:
        """Return the length of the results."""
        return len(self.results)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument("--noise_type", type=str, default="iid")
    args = parser.parse_args()

    start_time = time()
    with ProcessPoolExecutor(max_workers=96) as executor:
        res = executor.map(run_experiment, list(range(0,10000)), [args.n]*10000, [args.d]*10000, [args.noise_type]*10000)
    full_results = [p for p in list(res) if p is not None]
    process_time = time() - start_time

    experiment_results_si = Results(
        results=[result[0] for result in full_results],
        oc_p_values=[result[1] for result in full_results],
        times=[result[3] for result in full_results]
    )

    experiment_results_ds = [result[2] for result in full_results]

    log_message = f"n_{args.n}_d_{args.d}, 処理時間: {process_time:.2f} 秒"
    with open("process_time_log.txt", "a") as log_file:
        log_file.write(log_message + "\n")

    result_path = "summary_pkl/artificial"

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    file_name = f"n_{args.n}_d_{args.d}_{args.noise_type}_si.pkl"
    file_path = os.path.join(result_path, file_name)
    with open(file_path, 'wb') as f:
        pickle.dump(experiment_results_si, f)

    file_name = f"n_{args.n}_d_{args.d}_{args.noise_type}_ds.pkl"
    file_path = os.path.join(result_path, file_name)
    with open(file_path, 'wb') as f:
        pickle.dump(experiment_results_ds, f)
