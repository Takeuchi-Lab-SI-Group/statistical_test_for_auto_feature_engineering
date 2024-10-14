import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concurrent.futures import ProcessPoolExecutor
from source.tree_autofe import Tree_autofe, create_cov_matrix
import argparse
import time
import pickle
import numpy as np
import sympy as sp
import pandas as pd

def run_experiment(seed, n, d, noise_type, option):

    try:
        start_time = time.time()

        # setup hyperparameters
        transformation_list = ['sin', 'exp', 'mul', 'sqrt']
        max_depth = 6 
        generate_node = 3 
        extract_node = 1
        gamma = 4 

        # generate data
        rng = np.random.default_rng(seed)
        X = rng.standard_normal(size=(n, d))
        # dataframe
        symbol_cols = [sp.symbols(f'x{i}', real=True) for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=symbol_cols)

        if noise_type == 'iid':
            y = rng.standard_normal(size=(n,))
            cov = np.eye(n)
        
        elif noise_type == 'corr':
            cov = create_cov_matrix(n)
            y = rng.multivariate_normal(mean=np.zeros(n), cov=cov)
        
        else:
            raise ValueError(f"Invalid noise_type: {noise_type}. Choose from 'iid', 'corr'.")
        
        tree_inference = Tree_autofe(seed, rng, df, y, cov, max_depth, generate_node, extract_node, gamma, transformation_list, None, noise_type)

        if hasattr(tree_inference, option):
            result = getattr(tree_inference, option)()
            if result is not None:
                p_value = result[0]
            else:
                return None
        else:
            raise ValueError(f"Invalid option: {option}. Choose from 'naive', 'oc', 'DS', 'parametric'.")

        return p_value, time.time() - start_time
    
    except Exception as e:
        print(e)
        return None


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument("--noise_type", type=str, default="iid")
    parser.add_argument("--option", type=str, default="DS")
    args = parser.parse_args()

    with ProcessPoolExecutor(max_workers=8) as executor:
        res = executor.map(run_experiment, list(range(0,10000)), [args.n]*10000, [args.d]*10000, [args.noise_type]*10000, [args.option]*10000)
    p_value_list = []
    for p in list(res):
        if p is not None and p[0] is not None:
            p_value_list.append(p)
   
    result_path = f"results/fpr"

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    file_name = f"n_{args.n}_d_{args.d}_{args.noise_type}_{args.option}.pkl"

    file_path = os.path.join(result_path,file_name)

    with open(file_path, 'wb') as f:
        pickle.dump(p_value_list, f)