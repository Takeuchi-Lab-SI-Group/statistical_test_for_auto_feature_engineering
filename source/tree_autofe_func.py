import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import sympy as sp
import pandas as pd
from sicore import polytope_to_interval
from source.basic_func import compute_AIC, compute_AIC_component
from functools import reduce

operations = {
    # unary
    'abs': (np.abs, sp.Abs),
    '^2': (np.square,  lambda x: sp.Pow(x, 2)),
    'inv': (lambda x: np.where(x != 0, 1/x, x),  lambda x: sp.Pow(x, -1)),  
    'log': (lambda x: np.log(np.abs(x)), sp.log),
    'sqrt': (lambda x: np.sqrt(np.abs(x)), sp.sqrt),
    'power3': (lambda x:np.power(x,3), lambda x: sp.Pow(x, 3)),
    'sin': (np.sin, sp.sin),
    'cos': (np.cos, sp.cos),
    'tan': (np.tan, sp.tan),
    'exp': (np.exp, sp.exp),
    # binary
    'add': (lambda x,y: np.add(x, y), lambda x,y: sp.Add(x,y)),
    'sub': (lambda x,y: np.subtract(x, y), lambda x,y: sp.simplify(x-y)),
    'mul': (lambda x,y: np.multiply(x, y), lambda x,y: sp.Mul(x, y)),
    'div': (lambda x,y: np.where(y != 0, np.divide(x, y), np.divide(x, y + 1e-6)), lambda x,y: sp.simplify(x / y))
}

class Node:
    def __init__(self, dataset, accuracy, no_improve):
        self.dataset = dataset  
        self.accuracy = accuracy  # AIC
        self.no_improve = no_improve  
        self.applied_transformations = {}  
        self.done = False 
        self.exhausted_transformations = set() 

    def apply_transformation(self, feature_col, transformation):
        
        if transformation not in self.applied_transformations:
            self.applied_transformations[transformation] = set()
        
        if isinstance(feature_col, tuple):
            self.applied_transformations[transformation].add(frozenset(feature_col))
        
        elif isinstance(feature_col, pd.Index):
            self.applied_transformations[transformation].add(frozenset(feature_col.tolist()))
        
        else:
            self.applied_transformations[transformation].add(feature_col)

    def transformation_applied(self, transformation):
        return self.applied_transformations.get(transformation, False)
    
    def mark_transformation_exhausted(self, transformation):
        self.exhausted_transformations.add(transformation)

    def is_transformation_exhausted(self, transformation):
        return transformation in self.exhausted_transformations

    def check_if_done(self, transformation_list):
        if len(self.exhausted_transformations) == len(transformation_list):
            self.done = True
        

def get_unapplied_feature_pairs(node, transformation):
    applied_pairs = node.applied_transformations.get(transformation, set())
    features = list(node.dataset.columns)
    unapplied_pairs = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            pair = frozenset([features[i], features[j]])
            if pair not in applied_pairs:
                unapplied_pairs.append((features[i], features[j]))
    
    return unapplied_pairs

def get_unapplied_features(node, transformation):
    applied_features = node.applied_transformations.get(transformation, set())
    return [f for f in node.dataset.columns if f not in applied_features]

def select_node_and_transformation(current_nodes, transformation_list, rng):
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
        

def reducing_node(candidate_nodes, current_min_aic, gamma, extract_node):

    next_node_set = []
    sort_candidate_nodes = sorted(candidate_nodes, key=lambda x: x.accuracy)
      
    for node in sort_candidate_nodes:
        if node.accuracy < current_min_aic:
            node.no_improve = 0
        else:
            node.no_improve += 1
        
        if node.no_improve < gamma:
            next_node_set.append(node)

        if len(next_node_set) == extract_node:
            break

    if len(next_node_set) == 0:
        return next_node_set
    
    return next_node_set

def feature_generation(df, y, cov, max_depth, generate_node, extract_node, gamma, seed, transformation_list):

    root_dataset = df
    root_accuracy = compute_AIC(df.values, y, cov)
    no_improve = 0
    root = Node(root_dataset, root_accuracy, no_improve)

    current_nodes = [root]

    rng = np.random.default_rng(seed)

    for _ in range(max_depth):
        candidate_nodes = []
        while len(candidate_nodes) < generate_node:
            
            node, operation = select_node_and_transformation(current_nodes, transformation_list, rng)
            if node is None:
                next_node_set = current_nodes
                break 
            
            np_func, sp_func = operations[operation]

            if operation in ['add', 'sub', 'mul', 'div']:
                feature_pairs = get_unapplied_feature_pairs(node, operation)
                
                if not feature_pairs:
                    node.exhausted_transformations.add(operation)
                    continue 
                
                cols = rng.choice(feature_pairs)
                col1, col2 = cols                
                feature = np_func(node.dataset[col1].values, node.dataset[col2].values)
                expr = sp_func(col1, col2)
                new_feature = pd.DataFrame(feature, columns=[expr])
                node.apply_transformation((col1, col2), operation)

            if operation in ['abs', '^2', 'inv', 'log', 'sqrt', 'power3', 'sin', 'cos', 'tan', 'exp']:
                unapplied_features = get_unapplied_features(node, operation)
                
                if not unapplied_features:
                    node.exhausted_transformations.add(operation)
                    continue 

                col = rng.choice(unapplied_features)
                col_copy = node.dataset[col].copy()

                if operation == 'exp' and np.any(col_copy.values >= 5):
                    continue

                feature = np_func(col_copy.values)

                if np.any(np.isinf(feature)):
                    continue

                expr = sp_func(col)
                new_feature = pd.DataFrame(feature, columns=[expr])
                node.apply_transformation(col, operation)
            
            df_new = pd.concat([node.dataset, new_feature], axis=1)
            df_new = df_new.loc[:, ~df_new.columns.duplicated()]
            
            if set(df_new.columns) == set(node.dataset.columns):
                continue

            max_corr = max(abs(np.corrcoef(node.dataset[col], new_feature.values.reshape(-1, ))[0, 1]) for col in node.dataset.columns)
            
            if max_corr < 0.9:
                new_accuracy = compute_AIC(df_new.values, y, cov)
                new_node = Node(df_new, new_accuracy, node.no_improve)

                if len(candidate_nodes) != 0:
                    if not any(set(new_node.dataset.columns) == set(select_node.dataset.columns) for select_node in candidate_nodes):
                        candidate_nodes.append(new_node)

                else:
                    candidate_nodes.append(new_node)
                
        current_min_aic = current_nodes[0].accuracy
        next_node_set = reducing_node(candidate_nodes, current_min_aic, gamma, extract_node)
        if len(next_node_set) == 0:
            next_node_set = current_nodes
            break

        current_nodes = next_node_set

    best_node = next_node_set[0]
    
    return best_node.dataset

def reducing_node_SI(a, b, z, candidate_nodes, current_min_node, gamma, extract_node, l_list, u_list, cov):

    next_node_set = []
    current_min_aic = current_min_node.accuracy

    sort_candidate_nodes = sorted(candidate_nodes, key=lambda x: x.accuracy)
    for i in range(len(sort_candidate_nodes)-1):
        first_node_component = compute_AIC_component(sort_candidate_nodes[i].dataset.values, cov)
        second_node_component = compute_AIC_component(sort_candidate_nodes[i+1].dataset.values, cov)
        intervals = polytope_to_interval(a, b, first_node_component-second_node_component, np.zeros(a.shape[0]), 0)
        for left ,right in intervals:
            if left < z < right:
                l_list.append(left)
                u_list.append(right)
                break
    
    unselect_node_component = compute_AIC_component(current_min_node.dataset.values, cov)
    for node in sort_candidate_nodes:
        select_node_component = compute_AIC_component(node.dataset.values, cov)
        
        if node.accuracy < current_min_aic:
            node.no_improve = 0

            intervals = polytope_to_interval(a, b, select_node_component-unselect_node_component, np.zeros(a.shape[0]), 2)
            for left, right in intervals:
                if left < z < right:
                    l_list.append(left)
                    u_list.append(right)
                    break

        else:
            node.no_improve += 1
            
            intervals = polytope_to_interval(a, b, unselect_node_component-select_node_component, np.zeros(a.shape[0]), -2)
            for left, right in intervals:
                if left < z < right:
                    l_list.append(left)
                    u_list.append(right)
                    break

        if node.no_improve < gamma:
            next_node_set.append(node)

        if len(next_node_set) == extract_node:
            break

    if len(next_node_set) == 0:
        return next_node_set, l_list, u_list
    
    for select_node in next_node_set:
        select_node_component = compute_AIC_component(select_node.dataset.values, cov)
        for node in candidate_nodes:
            if node not in next_node_set:
                unselect_node_component = compute_AIC_component(node.dataset.values, cov)
                if select_node.accuracy < node.accuracy:
                    intervals = polytope_to_interval(a, b, select_node_component-unselect_node_component, np.zeros(a.shape[0]), 0)
                    for left, right in intervals:
                        if left < z < right:
                            l_list.append(left)
                            u_list.append(right)
                            break
                    
                if select_node.accuracy > node.accuracy:
                    intervals = polytope_to_interval(a, b, unselect_node_component-select_node_component, np.zeros(a.shape[0]), 0)
                    for left, right in intervals:
                        if left < z < right:
                            l_list.append(left)
                            u_list.append(right)
                            break
    
    return next_node_set, l_list, u_list

def feature_generation_SI(a, b, z, df, cov, l, u, max_depth, generate_node, extract_node, gamma, seed, transformation_list):

    yz_flatten = a + b * z
    y = yz_flatten.reshape(-1, )

    l_list, u_list = [l], [u]
    
    root_dataset = df
    root_accuracy = compute_AIC(df.values, y, cov)
    no_improve = 0
    root = Node(root_dataset, root_accuracy, no_improve)

    current_nodes = [root]

    rng = np.random.default_rng(seed)

    for _ in range(max_depth):
        candidate_nodes = []
        while len(candidate_nodes) < generate_node:
            node, operation = select_node_and_transformation(current_nodes, transformation_list, rng)
            if node is None:
                next_node_set = current_nodes
                break
            
            np_func, sp_func = operations[operation]

            if operation in ['add', 'sub', 'mul', 'div']:
                feature_pairs = get_unapplied_feature_pairs(node, operation)
                
                if not feature_pairs:
                    node.exhausted_transformations.add(operation)
                    continue
                
                cols = rng.choice(feature_pairs)
                col1, col2 = cols                
                feature = np_func(node.dataset[col1].values, node.dataset[col2].values)
                expr = sp_func(col1, col2)
                new_feature = pd.DataFrame(feature, columns=[expr])
                node.apply_transformation((col1, col2), operation)

            if operation in ['abs', '^2', 'inv', 'log', 'sqrt', 'power3', 'sin', 'cos', 'tan', 'exp']:
                unapplied_features = get_unapplied_features(node, operation)
                
                if not unapplied_features:
                    node.exhausted_transformations.add(operation)
                    continue

                col = rng.choice(unapplied_features)
                col_copy = node.dataset[col].copy()

                if operation == 'exp' and np.any(col_copy.values >= 5):
                    continue

                feature = np_func(col_copy.values)

                if np.any(np.isinf(feature)):
                    continue

                expr = sp_func(col)
                new_feature = pd.DataFrame(feature, columns=[expr])
                node.apply_transformation(col, operation)
            
            df_new = pd.concat([node.dataset, new_feature], axis=1)
            df_new = df_new.loc[:, ~df_new.columns.duplicated()]

            if set(df_new.columns) == set(node.dataset.columns):
                continue

            max_corr = max(abs(np.corrcoef(node.dataset[col], new_feature.values.reshape(-1, ))[0, 1]) for col in node.dataset.columns)
            
            if max_corr < 0.9:
                new_accuracy = compute_AIC(df_new.values, y, cov)
                new_node = Node(df_new, new_accuracy, node.no_improve)

                if len(candidate_nodes) != 0:
                    if not any(set(new_node.dataset.columns) == set(select_node.dataset.columns) for select_node in candidate_nodes):
                        candidate_nodes.append(new_node)

                else:
                    candidate_nodes.append(new_node)
                
        current_min_node = current_nodes[0]
        next_node_set, l_list, u_list = reducing_node_SI(a, b, z, candidate_nodes, current_min_node, gamma, extract_node, l_list, u_list, cov)
        
        if len(next_node_set) == 0:
            next_node_set = current_nodes
            break

        current_nodes = next_node_set

    best_node = next_node_set[0]
    
    select_node_component = compute_AIC_component(best_node.dataset.values, cov)
    for node in next_node_set:
        if node != best_node:
            unselect_node_component = compute_AIC_component(node.dataset.values, cov)
            intervals = polytope_to_interval(a, b, select_node_component-unselect_node_component, np.zeros(a.shape[0]), 0)
            for left, right in intervals:
                if left < z < right:
                    l_list.append(left)
                    u_list.append(right)
                    break

    l = np.max(l_list)
    u = np.min(u_list)

    assert l < z < u

    return best_node.dataset, l, u


# for Data Splitting(DS)
operations_DS = {
    'Abs': (np.abs, sp.Abs),
    '^2': (np.square, lambda x: sp.Pow(x, 2)),
    'Integer': (int, int),
    'inv': (lambda x: np.where(x != 0, 1/x, x), lambda x: sp.Pow(x, -1)),
    'log': (lambda x: np.log(np.abs(x)), sp.log),
    'sqrt': (lambda x: np.sqrt(np.abs(x)), sp.sqrt),
    'Pow': (lambda x,n: np.power(x, n), lambda x,n: sp.Pow(x, n)),
    'power3': (lambda x: np.power(x, 3), lambda x: sp.Pow(x, 3)),
    'sin': (np.sin, sp.sin),
    'cos': (np.cos, sp.cos),
    'tan': (np.tan, sp.tan),
    'exp': (np.exp, sp.exp),
    'Add': (lambda x,y: np.add(x, y), lambda x,y: sp.Add(x,y)),
    'sub': (lambda x,y: np.subtract(x, y), lambda x,y: sp.simplify(x-y)),
    'Mul': (lambda x,y: np.multiply(x, y), lambda x,y: sp.Mul(x, y)),
    'div': (lambda x,y: np.where(y != 0, np.divide(x, y), np.divide(x, y + 1e-6)), lambda x,y: sp.simplify(x / y))
}

def generate_feature(data, expr):

    if isinstance(expr, sp.Symbol):
        return data[expr].values
        
    elif isinstance(expr, (sp.Integer, sp.Float)):
        return float(expr)

    # half
    elif expr == sp.S.Half:
        return 1/2

    # Rational
    elif isinstance(expr, sp.Rational):
        return float(expr)
    
    op = expr.func.__name__
    
    if op in operations_DS:
        np_func, _ = operations_DS[op]
        args = [generate_feature(data, arg) for arg in expr.args]
        args = [np.asarray(arg, dtype=float) for arg in args]

        if op == 'exp' and np.any(args[0] >= 5):
            return np.nan
        
        if op in ['Add', 'sub', 'Mul', 'div']:
            return reduce(np_func, args)
        
        elif op == 'Pow':
            base, exp = args
            if exp == 1/2:
                return np.sqrt(np.abs(base))

            elif exp % 1 == 0.5:
                return np_func(np.abs(base), exp)
            
            else:
                return np_func(base, exp)

        else:
            return np_func(args[0])
    
    else:
        raise ValueError(f"Unsupported operation: {op}")

def apply_features(data, feature_expressions):
    new_features = {}

    for expr in feature_expressions:
        col_name = expr
        feature = generate_feature(data, expr)
        if not np.isnan(feature).any():
            new_features[col_name] = feature

        else:
            return None

    return pd.DataFrame(new_features)