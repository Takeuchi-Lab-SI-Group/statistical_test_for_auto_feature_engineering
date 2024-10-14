import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import sympy as sp
import pandas as pd
from scipy.stats import norm
from sicore import tn_cdf_mpmath
from sklearn.model_selection import train_test_split
from source.tree_autofe_func import feature_generation, feature_generation_SI, apply_features
from source.basic_func import compute_AIC, compute_teststatistics

eps = 1e-6

def create_cov_matrix(size, rho=0.5):
    matrix = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(abs(i - j))
        matrix.append(row)
    cov = np.power(rho, matrix)
    return cov

class Tree_autofe:

    def __init__(
        self, 
        seed: int, 
        rng: np.random.Generator,
        df: pd.DataFrame,
        y: np.ndarray, 
        cov: np.ndarray,
        max_depth: int, 
        generate_node: int, 
        extract_node: int, 
        gamma: int, 
        transformation_list: list[str],
        feature_candidate_sympy: list[sp.Symbol],
        exp_type: str
    ) -> None:
        
        self.seed = seed
        self.rng = rng
        self.df = df
        self.y = y
        self.cov = cov
        # autofe hyperparameters
        self.max_depth = max_depth
        self.generate_node = generate_node
        self.extract_node = extract_node
        self.gamma = gamma
        self.transformation_list = transformation_list
        # candidate generate feature
        self.feature_candidate_sympy = feature_candidate_sympy
        # experiment type
        self.exp_type = exp_type
        # original AIC
        self.aic = compute_AIC(self.df.values, self.y, self.cov)

    def _generate_feature(self, df, y, cov):

        # delete categorical variables
        cat_features = {col for col in df.columns if len(df[col].unique()) <= 2}
        df_part = df.drop(cat_features, axis=1)

        df_fe = feature_generation(
            df_part, y, cov, self.max_depth, self.generate_node, 
            self.extract_node, self.gamma, self.seed, self.transformation_list
        )
        
        if len(cat_features) != 0:
            for col in cat_features:
                df_fe = pd.concat([df_fe, df[col]], axis=1)

        return df_fe
        
    def _select_test_index(self, df_fe):

        if self.feature_candidate_sympy is not None:

            generate_index = list(range(self.df.shape[1], df_fe.shape[1]))

            if len(generate_index) == 0:
                return None
            
            j_selected = self.rng.choice(generate_index)

            if df_fe.columns[j_selected] in self.feature_candidate_sympy:
                return j_selected
            else:
                return None
            
        elif self.exp_type == 'real':
            j_selected = self.rng.choice(list(range(self.df.shape[1], df_fe.shape[1])))
            return j_selected
        
        else:
            j_selected = self.rng.choice(list(range(df_fe.shape[1])))
            return j_selected
    
    def naive(self):
    
        df_fe_obs = self._generate_feature(self.df, self.y, self.cov)
        # compute AIC
        aic_fe = compute_AIC(df_fe_obs.values, self.y, self.cov)

        j_selected = self._select_test_index(df_fe_obs)

        if j_selected is None:
            return None
        
        _, _, z_obs, var = compute_teststatistics(df_fe_obs.values, self.y, j_selected, self.cov)
        z_obs = z_obs / np.sqrt(var)
        return 2 * norm.cdf(-np.abs(z_obs)), self.aic, aic_fe
    
    def DS(self):
        # split data for algorithm, inference
        idx_alg, idx_si = train_test_split(range(self.df.shape[0]), test_size=0.5, random_state=self.seed)
        df_alg, df_si = self.df.iloc[idx_alg, :], self.df.iloc[idx_si, :]
        y_alg, y_si = self.y[idx_alg], self.y[idx_si]

        # reset index
        df_alg, df_si = df_alg.reset_index(drop=True), df_si.reset_index(drop=True)
        
        if self.exp_type == "estimated":
            # for algorithm
            residuals = y_alg - df_alg @ np.linalg.pinv(df_alg.T @ df_alg) @ df_alg.T @ y_alg
            sigma_alg = np.std(residuals, ddof=df_alg.shape[1])
            cov_alg = np.eye(len(y_alg)) * sigma_alg ** 2

            # for inference
            residuals = y_si - df_si @ np.linalg.pinv(df_si.T @ df_si) @ df_si.T @ y_si
            sigma_si = np.std(residuals, ddof=df_si.shape[1])
            cov_si = np.eye(len(y_si)) * sigma_si ** 2

        elif self.exp_type in ['skewnorm', 'exponnorm', 'gennormsteep', 'gennormflat', 't', None, 'iid', 'real']:
            cov_alg = np.eye(len(y_alg))
            cov_si = np.eye(len(y_si))

        elif self.exp_type == 'corr':
            cov_alg = create_cov_matrix(len(y_alg))
            cov_si = create_cov_matrix(len(y_si))

        else:
            raise ValueError(f"Invalid exp_type: {self.exp_type}")

        df_alg_fe = self._generate_feature(df_alg, y_alg, cov_alg)
        
        # compute AIC
        df_new = apply_features(self.df, df_alg_fe.columns)
        aic_fe = None
        if df_new is not None:
            aic_fe = compute_AIC(df_new.values, self.y, self.cov)

        # apply features to df_si (reproduce the same features as df_alg, so use df_alg_fe)
        df_si_fe = apply_features(df_si, df_alg_fe.columns)
        
        if df_si_fe is not None:
            
            # select test index
            j_selected = self._select_test_index(df_si_fe)
            
            if j_selected is None:
                return None
            
            _, _, z_obs, var = compute_teststatistics(df_si_fe.values, y_si, j_selected, cov_si)
            z_obs = z_obs / np.sqrt(var)
            return 2 * norm.cdf(-np.abs(z_obs)), self.aic, aic_fe
        
    def oc(self):

        df_fe_obs = self._generate_feature(self.df, self.y, self.cov)
        # compute AIC
        aic_fe = compute_AIC(df_fe_obs.values, self.y, self.cov)

        j_selected = self._select_test_index(df_fe_obs)
        if j_selected is None:
            return None
        
        a, b, z_obs, var = compute_teststatistics(df_fe_obs.values, self.y, j_selected, self.cov)

        l, u = -np.inf, np.inf
        _, l, u = feature_generation_SI(
            a, b, z_obs, self.df, self.cov, l, u, self.max_depth, 
            self.generate_node, self.extract_node, self.gamma, self.seed, self.transformation_list
        )
        
        z_interval = [[l, u]]
        interval = z_interval / np.sqrt(var)
        z_obs = z_obs / np.sqrt(var)

        pivot = tn_cdf_mpmath(z_obs, interval, absolute=True)
        return 1 - pivot, self.aic, aic_fe
    
    def parametric(self):

        df_fe_obs = self._generate_feature(self.df, self.y, self.cov)
        # compute AIC
        aic_fe = compute_AIC(df_fe_obs.values, self.y, self.cov)

        j_selected = self._select_test_index(df_fe_obs)
        if j_selected is None:
            return None
        
        a, b, z_obs, var = compute_teststatistics(df_fe_obs.values, self.y, j_selected, self.cov)
        
        std = np.sqrt(var)
        z_min, z_max = -10 * std - np.abs(z_obs), 10 * std + np.abs(z_obs)

        z = z_min
        z_interval = []

        while z < z_max:
            l, u = -np.inf, np.inf
            df_fe_z, l, u = feature_generation_SI(
                a, b, z, self.df, self.cov, l, u, self.max_depth, 
                self.generate_node, self.extract_node, self.gamma, self.seed, self.transformation_list
            )
            if set(df_fe_z.columns) == set(df_fe_obs.columns):
                z_interval.append([l, u])
            z = u + eps

        z_obs = z_obs / std
        interval = z_interval / std

        pivot = tn_cdf_mpmath(z_obs, interval, absolute=True)
        return 1 - pivot, self.aic, aic_fe

