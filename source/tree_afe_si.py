import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import sympy as sp
import pandas as pd
import time
from scipy.stats import norm
from typing import Literal
from sicore import SelectiveInferenceNorm, SelectiveInferenceResult
from sklearn.model_selection import train_test_split
from source.tree_afe_func import TreeAFE, SafeMath



class TreeAFE_si:

    def __init__(
        self,
        seed: int,
        max_depth: int,
        num_generate_node: int,
        num_extract_node: int,
        gamma: int,
        transformation_list: list[str]
    ):
        
        self.seed = seed
        self.max_depth = max_depth
        self.num_generate_node = num_generate_node
        self.num_extract_node = num_extract_node
        self.gamma = gamma
        self.transformation_list = transformation_list

        self.generator_AFE = TreeAFE(
            seed=self.seed,
            max_depth=self.max_depth,
            num_generate_node=self.num_generate_node,
            num_extract_node=self.num_extract_node,
            gamma=self.gamma,
            transformation_list=self.transformation_list
        )

    def apply_afe(
        self,
        df: pd.DataFrame,
        y: np.ndarray,
        cov: np.ndarray
    ) -> pd.DataFrame:
        
        self.df = df    
        self.y = y
        self.cov = cov
        
        df_fe_obs = self.generator_AFE.generate_features(df, y, cov)
        self.df_fe_obs = df_fe_obs
        self.M = df_fe_obs.columns

        return self.df_fe_obs
    
    def apply_afe_ds(
        self,
        df: pd.DataFrame,
        y: np.ndarray,
        cov: np.ndarray
    ) -> pd.DataFrame:
        
        self.df = df
        self.y = y
        
        # split data into algorithm and inference
        idx_alg, idx_si = train_test_split(range(self.df.shape[0]), test_size=0.5, random_state=self.seed)
        df_alg, df_si = self.df.iloc[idx_alg, :], self.df.iloc[idx_si, :]
        y_alg, y_si = y[idx_alg], y[idx_si]

        # reset index
        df_alg, df_si = df_alg.reset_index(drop=True), df_si.reset_index(drop=True)

        self.y_si = y_si
        self.df_si = df_si

        # splitting cov
        cov_alg = cov[np.ix_(idx_alg, idx_alg)]
        cov_si = cov[np.ix_(idx_si, idx_si)]

        self.cov_alg = cov_alg
        self.cov_si = cov_si

        # apply AFE to algorithm dataset
        df_fe_alg = self.generator_AFE.generate_features(df_alg, y_alg, cov_alg)

        self.df_fe_alg = df_fe_alg

        return self.df_fe_alg
        
    def model_selector(self, M: list[str]) -> bool:
        """Conduct selective inference.

        It takes a list of indices M as input and returns True if the generated features
        are the same as the features generated from the obsearved dataset (self.X, self.y).

        Parameters
        ----------
        M : list[str]
            A list of indices of the generated features columns.

        Returns
        -------
        bool
            True if the generated features columns are the same as the features generated from the obsearved dataset (self.X, self.y).
        """
        return set(self.M) == set(M)
    
    def algorithm(
        self,
        a: np.ndarray,
        b: np.ndarray,
        z: float
    ) -> tuple[list[str],list[list[float]]]:
        
        df_fe_z, l, u = self.generator_AFE.feature_generation_SI(a, b, z, self.df, self.cov)
        return df_fe_z.columns, [[l, u]]
        
    def inference(
        self,
        index: int,
        inference_mode: Literal["parametric", "over_conditioning"] = "parametric"
    ) -> SelectiveInferenceResult:
        
        X_fe = self.df_fe_obs.values
        ej = np.zeros(X_fe.shape[1])
        ej[index] = 1
        etaj = X_fe @ np.linalg.pinv(X_fe.T @ X_fe) @ ej
        si = SelectiveInferenceNorm(self.y, self.cov, etaj)
        return si.inference(
            self.algorithm,
            self.model_selector,
            inference_mode=inference_mode
        )
     
    def inference_ds(
        self,
        index: int,
    ) -> float:
        
         # Based on df_fe_alg, apply AFE to inference dataset
        df_fe_si = pd.DataFrame()
        safe_math = SafeMath()
        custom_modules = {
            'exp': safe_math.exp,
            'sqrt': safe_math.sqrt,
            'inv': safe_math.inv,
            'log': safe_math.log,
            'div': safe_math.div
        }

        for expr in self.df_fe_alg.columns:
            lambda_func = sp.lambdify(self.df_si.columns, expr, modules=[custom_modules, 'numpy'])
            result = lambda_func(*[self.df_si[col] for col in self.df_si.columns])
            df_fe_si[expr] = result
        
        X_fe = df_fe_si.values 
        ej = np.zeros(X_fe.shape[1])
        ej[index] = 1
        etaj = X_fe @ np.linalg.pinv(X_fe.T @ X_fe) @ ej
        z_obs = etaj.T @ self.y_si
        var = etaj.T @ self.cov_si @ etaj
        z_obs = z_obs / np.sqrt(var)

        return 2 * norm.cdf(-np.abs(z_obs))


def return_generated_features(
        df: pd.DataFrame,
        df_fe: pd.DataFrame
    ) -> pd.DataFrame:
        
    # Based on df_fe_alg, apply AFE to inference dataset
    df_fe_si = pd.DataFrame()
    safe_math = SafeMath()
    custom_modules = {
        'exp': safe_math.exp,
        'sqrt': safe_math.sqrt,
        'inv': safe_math.inv,
        'log': safe_math.log,
        'div': safe_math.div
    }

    for expr in df_fe.columns:
        lambda_func = sp.lambdify(df.columns, expr, modules=[custom_modules, 'numpy'])
        result = lambda_func(*[df[col] for col in df.columns])
        df_fe_si[expr] = result

    return df_fe_si
