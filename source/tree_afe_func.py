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
from sicore import polytope_below_zero
from typing import Literal

# Node for tree based AFE 
class Node:

    def __init__(
        self, 
        dataset, 
        accuracy, 
        no_improve
    ):
        """
        Initialize a Node in the feature generation tree.

        Parameters:
        -----------
        dataset : pd.DataFrame
            Input dataset with features
        accuracy : float
            AIC value of the dataset
        no_improve : int
            Number of iterations with no improvement
        applied_transformations : dict
            Dictionary of applied transformations
        done : bool
            Whether the node has finished generating features
        exhausted_transformations : set
            Set of transformations that cannot be applied
        """

        self.dataset = dataset 
        self.accuracy = accuracy 
        self.no_improve = no_improve
        self.applied_transformations = {}
        self.done = False 
        self.exhausted_transformations = set()

    def apply_transformation(self, feature_col, transformation: str):
        """
        Record the features after applying a transformation.

        Parameters:
        -----------
        feature_col : Union[str, Tuple[str, str], pd.Index]
            Feature column or pair of columns

        transformation : str
            Name of the transformation

        """

        # Add a new empty set() as the value for the transformation key in the dictionary
        # to avoid errors with the add method in later implementations
        if transformation not in self.applied_transformations:
            self.applied_transformations[transformation] = set()
        
        # If the feature is a pair, convert it to a frozenset for the set
        if isinstance(feature_col, tuple):
            self.applied_transformations[transformation].add(frozenset(feature_col))

        # If the feature is a pd.Index object, convert it to a frozenset
        elif isinstance(feature_col, pd.Index):
            self.applied_transformations[transformation].add(frozenset(feature_col.tolist()))
        
        # If the feature is single, add it to the set as is
        else:
            self.applied_transformations[transformation].add(feature_col)

    def transformation_applied(self, transformation: str) -> bool:
        """
        Check if all features have been used in a transformation. (min, max, mean, etc.)

        Parameters:
        -----------
        transformation : str
            Name of the transformation
            
        Returns:
        --------
        bool
            Whether all features have been used in the transformation
        """
        return self.applied_transformations.get(transformation, False)
    
    def is_transformation_exhausted(self, transformation: str) -> bool:
        """
        Check if a transformation cannot be applied.

        Parameters:
        -----------
        transformation : str
            Name of the transformation
        
        Returns:
        --------
        bool
            Whether the transformation cannot be applied
        """
        return transformation in self.exhausted_transformations

    def check_if_done(self, transformation_list: list):
        """
        Check if all transformations have been exhausted.

        Parameters:
        -----------
        transformation_list : List[str]
            List of transformations
        """

        if len(self.exhausted_transformations) == len(transformation_list):
            self.done = True

class SafeMath:
    def __init__(self, threshold=5, epsilon=1e-6):
        self.threshold = threshold
        self.epsilon = epsilon

    def exp(self, x):
        x_array = np.asarray(x)
        result = np.zeros_like(x_array, dtype=float)

        mask_normal = x_array <= self.threshold
        result[mask_normal] = np.exp(x_array[mask_normal])

        mask_exceeded = x_array > self.threshold
        result[mask_exceeded] = x_array[mask_exceeded] # exp at the threshold
        return result

    def sqrt(self, x):
        x_array = np.asarray(x)
        return np.sqrt(np.abs(x_array))

    def inv(self, x):
        x_array = np.asarray(x)
        return np.where(x_array != 0, 1 / x_array, x_array)

    def log(self, x):
        x_array = np.asarray(x)
        return np.log(np.abs(x_array))
        # if np.any(x_array < 0):
        #     min_value = np.min(x_array)
        #     col_shifted = x_array + abs(min_value) + self.epsilon
        # elif np.any(x_array == 0):
        #     col_shifted = x_array + self.epsilon
        # else:
        #     col_shifted = x_array
        # return np.log(col_shifted)

    def div(self, x, y):
        x_array = np.asarray(x)
        y_array = np.asarray(y)
        return np.where(y_array != 0, x_array / y_array, x_array / (y_array + self.epsilon))
        

class TreeAFE:

    def __init__(
        self, 
        seed: int = 42, 
        max_depth: int = 4, 
        num_generate_node: int = 10, 
        num_extract_node: int = 5, 
        gamma: int = 2, 
        transformation_list: list = ['abs', 'log', 'add']
    ):
        """
        Initialize the AutoFeatureGenerator with configuration parameters.
        
        Parameters:
        -----------
        seed : int, optional
            Random seed for reproducibility
        max_depth : int, optional
            Maximum depth of feature generation tree
        num_generate_node : int, optional
            Number of candidate nodes to generate at each depth
        num_extract_node : int, optional
            Number of nodes to extract and proceed with at each depth
        gamma : int, optional
            Patience parameter for stopping feature generation
        transformation_list : List[str], optional
            List of transformations to apply
        """

        self.seed = seed
        self.max_depth = max_depth
        self.num_generate_node = num_generate_node
        self.num_extract_node = num_extract_node
        self.gamma = gamma
        self.transformation_list = transformation_list

        # Predefined operations dictionary with NumPy and SymPy functions
        self.operations = self._define_operations()

    def _safe_min_shift(self, col: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """
        Safely shift values to ensure non-negative and non-zero inputs for transformations.
        
        Parameters:
        -----------
        col : np.ndarray
            Input array to be shifted
        eps : float, optional
            Small epsilon value to add for stability
        
        Returns:
        --------
        np.ndarray
            Shifted array
        """
        if np.any(col < 0):
            min_value = np.min(col)
            col_shifted = col + abs(min_value) + eps
        elif np.any(col == 0):
            col_shifted = col + eps
        else:
            col_shifted = col
        return col_shifted
    
    def _define_operations(self) -> dict:
        """
        Define operations with both NumPy and SymPy implementations.
        
        Returns:
        --------
        dict
            Dictionary of operations with NumPy and SymPy functions
        """

        return {
            # unary
            'abs': (np.abs, sp.Abs),
            '^2': (np.square,  lambda x: sp.Pow(x, 2)),
            'inv': (
                lambda x: np.where(x != 0, 1/x, x),  
                lambda x: sp.Pow(x, -1)
            ),  
            'log': (
                # lambda x: np.log(self._safe_min_shift(x)), 
                lambda x: np.log(np.abs(x)),
                lambda x: sp.log(sp.Abs(x))
            ),
            'sqrt': (
                lambda x: np.sqrt(np.abs(x)), 
                lambda x: sp.sqrt(sp.Abs(x))
            ),
            'power3': (
                lambda x:np.power(x, 3), 
                lambda x: sp.Pow(x, 3)
            ),
            'sin': (np.sin, sp.sin),
            'cos': (np.cos, sp.cos),
            'tan': (np.tan, sp.tan),
            'exp': (
                lambda x: np.where(x >= 5, x, np.exp(x)), 
                sp.exp
            ),
            # binary
            'add': (
                lambda x,y: np.add(x, y), 
                lambda x,y: sp.Add(x,y)
            ),
            'sub': (
                lambda x,y: np.subtract(x, y), 
                lambda x,y: sp.simplify(x-y)
            ),
            'mul': (
                lambda x,y: np.multiply(x, y), 
                lambda x,y: sp.Mul(x, y)
            ),
            'div': (
                lambda x,y: np.where(y != 0, np.divide(x, y), np.divide(x, y + 1e-6)), 
                lambda x,y: sp.simplify(x / y)
            ),
            # aggregation
            'mean': (
                lambda x: np.mean(x, axis=1), 
                lambda x: sp.simplify(sum(x) / len(x))
            ),
            'max': (
                lambda x: np.max(x, axis=1), 
                lambda x: sp.Max(*x)
            ),
            'min': (
                lambda x: np.min(x, axis=1), 
                lambda x: sp.Min(*x)
            )
        }
    
    def _compute_aic_component(self, X: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """
        Compute the AIC component for a given dataset.
        
        Parameters:
        -----------
        X : np.ndarray
            Input dataset with features
        cov : np.ndarray
            Covariance matrix
        
        Returns:
        --------
        np.ndarray
            AIC component
        """
        cov_inv = np.linalg.pinv(cov)
        return cov_inv - cov_inv @ X @ np.linalg.pinv(X.T @ cov_inv @ X) @ X.T @ cov_inv
    
    def _compute_aic(self, df: pd.DataFrame, y: np.ndarray, cov: np.ndarray) -> float:
        """
        Compute the Akaike Information Criterion (AIC) for a given dataset.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataset with features
        y : np.ndarray
            Target variable
        cov : np.ndarray
            Covariance matrix
        
        Returns:
        --------
        float
            AIC value
        """
        X = df.values
        A_M = self._compute_aic_component(X, cov)
        aic = y.T @ A_M @ y + 2 * X.shape[1]
        return aic
    
    def _select_node_and_transformation(self, current_nodes: list[Node], transformation_list: list[str], rng: np.random.Generator) -> tuple:
        """
        Select an applicable node and transformation for an unfinished node.

        Parameters:
        -----------
        current_nodes : List[Node]
            List of current nodes
        transformation_list : List[str]
            List of transformations
        rng : np.random.Generator
            Random number generator
        
        Returns:
        --------
        tuple
            Tuple of selected node and transformation
        """

        available_nodes = [node for node in current_nodes if not node.done]
        
        if not available_nodes:
            return None, None

        while True:
            node = rng.choice(available_nodes)
            available_transformations = [t for t in transformation_list if not node.is_transformation_exhausted(t)]
            if not available_transformations:
                node.check_if_done(transformation_list)
                available_nodes.remove(node)
                if not available_nodes:
                    return None, None
            else:
                return node, rng.choice(available_transformations)
            
    def _get_unapplied_features(self, node: Node, transformation: str) -> list:
        """
        Get the features that have not been applied for a transformation.

        Parameters:
        -----------
        node : Node
            Current node
        transformation : str
            Name of the transformation
        
        Returns:
        --------
        list
            List of unapplied features
        """

        # Return an empty set() if the key does not exist
        applied_features = node.applied_transformations.get(transformation, set())
        
        # Get the unapplied features based on the transformation
        if transformation in ['abs', '^2', 'inv', 'log', 'sqrt', 'power3', 'sin', 'cos', 'tan', 'exp']:
            # Get the unapplied features
            return [f for f in node.dataset.columns if f not in applied_features]

        elif transformation in ['add', 'sub', 'mul', 'div']:
            features = list(node.dataset.columns)
            unapplied_pairs = []
            for i in range(len(features)):
                for j in range(i + 1, len(features)):
                    pair = frozenset([features[i], features[j]])
                    if pair not in applied_features:
                        unapplied_pairs.append((features[i], features[j]))

            # Return the unapplied feature pairs
            return unapplied_pairs
        
    def _reducing_node(self, candidate_nodes: list[Node], current_min_aic: float, gamma: int, extract_node: int) -> list[Node]:
        """
        Reduce the number of candidate nodes based on the AIC value.

        Parameters:
        -----------
        candidate_nodes : List[Node]
            List of candidate nodes
        current_min_aic : float
            Current minimum AIC value
        gamma : int
            Patience parameter for stopping feature generation
        extract_node : int
            Number of nodes to extract
        
        Returns:
        --------
        List[Node]
            Reduced list of candidate nodes
        """

        next_node_set = []
        sort_candidate_nodes = sorted(candidate_nodes, key=lambda x: x.accuracy)
        
        for node in sort_candidate_nodes:
            if node.accuracy < current_min_aic:
                node.no_improve = 0
            else:
                node.no_improve += 1
            
            # Add the node to the next step set if the improvement is less than gamma
            if node.no_improve < gamma:
                next_node_set.append(node)

            if len(next_node_set) == extract_node:
                break

        if len(next_node_set) == 0:
            return next_node_set
        
        return next_node_set

    def generate_features(self, df: pd.DataFrame, y: np.ndarray, cov: np.ndarray) -> pd.DataFrame:
        """
        Generate new features using the AutoFeatureGenerator.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataset with features
        y : np.ndarray
            Target variable
        cov : np.ndarray
            Covariance matrix

        Returns:
        --------
        pd.DataFrame
            Dataset with generated features
        """

        # Initialize the root node
        root_dataset = df
        root_accuracy = self._compute_aic(df, y, cov)
        no_improve = 0
        root = Node(root_dataset, root_accuracy, no_improve)

        current_nodes = [root]

        # Random number generator
        rng = np.random.default_rng(self.seed)

        for _ in range(self.max_depth):
            candidate_nodes = []
            while len(candidate_nodes) < self.num_generate_node:
                # Select an applicable node and transformation
                node, operation = self._select_node_and_transformation(current_nodes, self.transformation_list, rng)
                if node is None:
                    next_node_set = current_nodes
                    break # Exit if all nodes are finished

                # Get the corresponding functions (numpy and sympy) from the dictionary
                np_func, sp_func = self.operations[operation]

                # In the case of transformations that use all features
                if operation in ['min', 'max', 'mean']:
                    # Apply the transformation if it has not been applied
                    if node.transformation_applied(operation): 
                        node.exhausted_transformations.add(operation) # Record that it has already been applied
                        continue
                    else:
                        feature = np_func(node.dataset.values)
                        expr = sp_func(node.dataset.columns)
                        new_feature = pd.DataFrame(feature, columns=[expr])
                        node.apply_transformation(node.dataset.columns, operation)

                # In the case of binary transformations
                if operation in ['add', 'sub', 'mul', 'div']:
                    feature_pairs = self._get_unapplied_features(node, operation)
                    
                    if not feature_pairs:
                        node.exhausted_transformations.add(operation)
                        continue  # Skip if all pairs have been applied
                    
                    cols = rng.choice(feature_pairs)
                    col1, col2 = cols
                    # Swap col1 and col2 randomly for div
                    if operation == 'div':
                        if rng.choice([0, 1]) == 1:
                            col1, col2 = col2, col1 
                    # Generate the feature
                    feature = np_func(node.dataset[col1].values, node.dataset[col2].values)
                    expr = sp_func(col1, col2)
                    new_feature = pd.DataFrame(feature, columns=[expr])
                    # Record that the transformation and features has been applied
                    node.apply_transformation((col1, col2), operation)

                # In the case of unary transformations
                if operation in ['abs', '^2', 'inv', 'log', 'sqrt', 'power3', 'sin', 'cos', 'tan', 'exp']:
                    unapplied_features = self._get_unapplied_features(node, operation)
                    if not unapplied_features:
                        node.exhausted_transformations.add(operation)
                        continue  # Skip if there are no unapplied features
                    
                    # Randomly select a feature
                    col = rng.choice(unapplied_features)
                    # Generate the feature
                    feature = np_func(node.dataset[col].values)

                    # Skip if the feature contains inf
                    if np.any(np.isinf(feature)):
                        # Record that the transformation cannot be applied if inf is included
                        node.apply_transformation(col, operation)
                        continue

                    expr = sp_func(col)
                    new_feature = pd.DataFrame(feature, columns=[expr])
                    node.apply_transformation(col, operation)
                
                # Concatenate the new feature with the original dataset
                df_new = pd.concat([node.dataset, new_feature], axis=1)
                df_new = df_new.loc[:, ~df_new.columns.duplicated()]
                
                # Skip if the new dataset is the same as the original
                if set(df_new.columns) == set(node.dataset.columns):
                    continue

                # Add the new feature if the correlation coefficient is less than the threshold
                max_corr = max(abs(np.corrcoef(node.dataset[col], new_feature.values.reshape(-1,))[0,1]) for col in node.dataset.columns)
                
                if max_corr < 0.9:
                    new_accuracy = self._compute_aic(df_new, y, cov)
                    new_node = Node(df_new, new_accuracy, node.no_improve)
                    # Add the node if it is not in the candidate nodes
                    if len(candidate_nodes) != 0:
                        if not any(set(new_node.dataset.columns) == set(select_node.dataset.columns) for select_node in candidate_nodes):
                            candidate_nodes.append(new_node)

                    else:
                        candidate_nodes.append(new_node)
                    
            # Current minimum AIC at this depth
            current_min_aic = current_nodes[0].accuracy
            # Reduce the number of candidate nodes
            next_node_set = self._reducing_node(candidate_nodes, current_min_aic, self.gamma, self.num_extract_node)
            # Select the previous node if there is no next_node_set
            if len(next_node_set) == 0:
                next_node_set = current_nodes
                break

            current_nodes = next_node_set

        # Select the best node
        best_node = next_node_set[0]
        
        # Return the dataset with the best features
        return best_node.dataset
    
    def _reducing_node_SI(
        self, 
        a: np.ndarray, 
        b: np.ndarray, 
        z: float, 
        cov: np.ndarray,
        candidate_nodes: list[Node], 
        current_min_node: Node, 
        l_list: list[float], 
        u_list: list[float]
    ) -> tuple:
        
        """
        Reduce the number of candidate nodes based on the AIC value and compute truncated interval.

        Returns:
        --------
        tuple
            Tuple of reduced candidate nodes, truncated interval [l, u]
        """

        next_node_set = []
        current_min_aic = current_min_node.accuracy

        sort_candidate_nodes = sorted(candidate_nodes, key=lambda x: x.accuracy)
        # condition for sorting candidate_nodes
        for i in range(len(sort_candidate_nodes)-1):
            first_node_component = self._compute_aic_component(sort_candidate_nodes[i].dataset.values, cov)
            second_node_component = self._compute_aic_component(sort_candidate_nodes[i+1].dataset.values, cov)
            intervals = polytope_below_zero(a, b, first_node_component-second_node_component, np.zeros(a.shape[0]), 0)
            for left ,right in intervals:
                if left < z < right:
                    l_list.append(left)
                    u_list.append(right)
                    break
        
        unselect_node_component = self._compute_aic_component(current_min_node.dataset.values, cov)
        for node in sort_candidate_nodes:
            select_node_component = self._compute_aic_component(node.dataset.values, cov)
            
            if node.accuracy < current_min_aic:
                node.no_improve = 0

                # condition for selecting improved node
                intervals = polytope_below_zero(a, b, select_node_component-unselect_node_component, np.zeros(a.shape[0]), 2)
                for left, right in intervals:
                    if left < z < right:
                        l_list.append(left)
                        u_list.append(right)
                        break

            else:
                node.no_improve += 1
                
                # condition for selecting unimproved node
                intervals = polytope_below_zero(a, b, unselect_node_component-select_node_component, np.zeros(a.shape[0]), -2)
                for left, right in intervals:
                    if left < z < right:
                        l_list.append(left)
                        u_list.append(right)
                        break

            # Add the node to the next step set if the improvement is less than gamma
            if node.no_improve < self.gamma:
                next_node_set.append(node)

            if len(next_node_set) == self.num_extract_node:
                break

        if len(next_node_set) == 0:
            return next_node_set, l_list, u_list

        return next_node_set, l_list, u_list

    def feature_generation_SI(
        self, 
        a: np.ndarray, 
        b: np.ndarray, 
        z: float, 
        df: pd.DataFrame,
        cov: np.ndarray
     ) -> pd.DataFrame:
        
        """
        Generate new features and compute the truncated interval [l, u] for the SI method.

        Returns:
        --------
        pd.DataFrame
            Dataset with generated features
        """
        # yz 
        yz_flatten = a + b * z
        y = yz_flatten.reshape(-1, )

        l, u = -np.inf, np.inf
        l_list, u_list = [l], [u]
        
        root_dataset = df
        root_accuracy = self._compute_aic(df, y, cov)
        no_improve = 0
        root = Node(root_dataset, root_accuracy, no_improve)

        current_nodes = [root]

        rng = np.random.default_rng(self.seed)

        for _ in range(self.max_depth):
            candidate_nodes = []
            while len(candidate_nodes) < self.num_generate_node:
                # Select an applicable node and transformation
                node, operation = self._select_node_and_transformation(current_nodes, self.transformation_list, rng)
                if node is None:
                    next_node_set = current_nodes
                    break # Exit if all nodes are finished
                
                # Get the corresponding functions (numpy and sympy) from the dictionary
                np_func, sp_func = self.operations[operation]

                # In the case of transformations that use all features
                if operation in ['min', 'max', 'mean']:
                    # Apply the transformation if it has not been applied
                    if node.transformation_applied(operation):
                        node.exhausted_transformations.add(operation) # Record that it has already been applied
                        continue
                    else:
                        feature = np_func(node.dataset.values)
                        expr = sp_func(node.dataset.columns)
                        new_feature = pd.DataFrame(feature, columns=[expr])
                        node.apply_transformation(node.dataset.columns, operation)

                # In the case of binary transformations
                if operation in ['add', 'sub', 'mul', 'div']:
                    feature_pairs = self._get_unapplied_features(node, operation)
                    
                    if not feature_pairs:
                        node.exhausted_transformations.add(operation)
                        continue  # Skip if all pairs have been applied
                    
                    cols = rng.choice(feature_pairs)
                    col1, col2 = cols
                    # Swap col1 and col2 randomly for div
                    if operation == 'div':
                        if rng.choice([0, 1]) == 1:
                            col1, col2 = col2, col1    
                    # Generate the feature
                    feature = np_func(node.dataset[col1].values, node.dataset[col2].values)
                    expr = sp_func(col1, col2)
                    new_feature = pd.DataFrame(feature, columns=[expr])
                    # Record that the transformation and features has been applied
                    node.apply_transformation((col1, col2), operation)

                # In the case of unary transformations
                if operation in ['abs', '^2', 'inv', 'log', 'sqrt', 'power3', 'sin', 'cos', 'tan', 'exp']:
                    unapplied_features = self._get_unapplied_features(node, operation)
                    if not unapplied_features:
                        node.exhausted_transformations.add(operation)
                        continue  # Skip if there are no unapplied features

                    # Randomly select a feature
                    col = rng.choice(unapplied_features)
                    # Generate the feature
                    feature = np_func(node.dataset[col].values)

                    # Skip if the feature contains inf
                    if np.any(np.isinf(feature)):
                        # Record that the transformation cannot be applied if inf is included
                        node.apply_transformation(col, operation)
                        continue

                    expr = sp_func(col)
                    new_feature = pd.DataFrame(feature, columns=[expr])
                    node.apply_transformation(col, operation)
                
                # Concatenate the new feature with the original dataset
                df_new = pd.concat([node.dataset, new_feature], axis=1)
                df_new = df_new.loc[:, ~df_new.columns.duplicated()]

                # Skip if the new dataset is the same as the original
                if set(df_new.columns) == set(node.dataset.columns):
                    continue

                # Add the new feature if the correlation coefficient is less than the threshold
                max_corr = max(abs(np.corrcoef(node.dataset[col], new_feature.values.reshape(-1, ))[0, 1]) for col in node.dataset.columns)
                
                if max_corr < 0.9:
                    new_accuracy = self._compute_aic(df_new, y, cov)
                    new_node = Node(df_new, new_accuracy, node.no_improve)
                    # Add the node if it is not in the candidate nodes
                    if len(candidate_nodes) != 0:
                        if not any(set(new_node.dataset.columns) == set(select_node.dataset.columns) for select_node in candidate_nodes):
                            candidate_nodes.append(new_node)

                    else:
                        candidate_nodes.append(new_node)
                    
            # Current minimum AIC at this depth
            current_min_node = current_nodes[0]
            # Reduce the number of candidate nodes
            next_node_set, l_list, u_list = self._reducing_node_SI(a, b, z, cov, candidate_nodes, current_min_node, l_list, u_list)
            # Select the previous node if there is no next_node_set
            if len(next_node_set) == 0:
                next_node_set = current_nodes
                break

            current_nodes = next_node_set

        # Select the best node
        best_node = next_node_set[0]
        
        # Extract the truncated interval [l, u]
        l = np.max(l_list)
        u = np.min(u_list)

        assert l < z < u

        # Return the dataset with the best features and the truncated interval
        return best_node.dataset, l, u