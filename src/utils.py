# Author: Kshitij Kayastha
# Date: Feb 3, 2025


import lime
import lime.lime_tabular
import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import Callable
from copy import deepcopy


def recourse_needed(predict_fn: Callable, X: np.ndarray, y_target: float|int = 1):
    indices = np.where(predict_fn(X) == 1-y_target)
    return X[indices]

def recourse_validity(predict_fn: Callable, recourses: np.ndarray, y_target: float|int = 1):
    return sum(predict_fn(recourses) == y_target) / len(recourses)

def recourse_expectation(predict_proba_fn: Callable, recourses: np.ndarray):
    return sum(predict_proba_fn(recourses)[:,1]) / len(recourses)


def lime_explanation(predict_proba_fn: Callable, X: np.ndarray, x: np.ndarray):
    explainer = lime.lime_tabular.LimeTabularExplainer(training_data=X, mode='regression', discretize_continuous=False, feature_selection='none')
    exp = explainer.explain_instance(x, predict_proba_fn, num_features=X.shape[1], model_regressor=LogisticRegression())
    weights = exp.local_exp[1][0][1]
    bias = exp.intercept[1]
    return weights, bias

def sigmoid(x, weights, bias):
    return 1 / (1 + np.exp(-(np.exp(x, weights) + bias))) 

def l1_cost(x1, x2):
    return np.linalg.norm(x1-x2, 1, -1)

def l2_cost(x1, x2):
    return np.linalg.norm(x1-x2, 2, -1)

def linf_cost(x1, x2):
    return np.linalg.norm(x1-x2, np.inf, -1)


def generate_grid(center, delta, n, ord=None):
    linspaces = [np.linspace(center[i]-delta, center[i]+delta, n) for i in range(len(center))]
    grids = np.meshgrid(*linspaces)
    points = np.stack([grid.reshape(-1) for grid in grids]).T
    if ord != None:
        mask = np.linalg.norm(points-center, ord=ord, axis=1) <= delta
        points = points[mask]
    return points    

def hex2rgba(color, a=1):
    h = color.lstrip('#')
    c = f'rgba{tuple(int(h[i:i+2], 16) for i in (0, 2, 4))}'[:-1] + f', {a})'
    return c

def pareto_frontier(A, B):
    temp = np.column_stack((A, B))
    is_dominated = np.ones(temp.shape[0], dtype=bool)
    
    for i, c in enumerate(temp):
        if is_dominated[i]:
            is_dominated[is_dominated] = np.any(temp[is_dominated] < c, axis=1)
            is_dominated[i] = True
    return is_dominated

def find_pareto(x, y, return_index=False):
    a = list(zip(x, y))
    a = sorted(a, key=lambda x: (x[0], -x[1]))
    best = -1
    pareto = []
    mask = []
    for ie, e in enumerate(a):
        if e[1] >= best:
            pareto.append(e)
            mask.append(ie)
            best = e[1]
    if return_index:
        return [e[0] for e in pareto], [e[1] for e in pareto], mask
    return [e[0] for e in pareto], [e[1] for e in pareto]

def generate_nn_predictions(dataset, theta_0: np.ndarray, theta_r: np.ndarray, alpha: float):
    if dataset.name == 'sba':
        theta_r1 = deepcopy(theta_r) * 0.3
        theta_r2 = deepcopy(theta_r) * 0.5
        
        alphas1 = theta_r1 - theta_0
        theta_p1 = theta_0 - alphas1
        
        alphas2 = theta_r2 - theta_0
        theta_p2 = theta_0 - alphas2
    else:
        theta_r1 = deepcopy(theta_r)
        theta_r1[0] = theta_0[0]
        theta_r2 = deepcopy(theta_r)
        
        alphas1 = theta_r1 - theta_0
        theta_p1 = theta_0 - alphas1
        
        alphas2 = theta_r2 - theta_0
        theta_p2 = theta_0 - alphas2
        
    predictions = []
    for pred in [theta_0, theta_r1, theta_r2, theta_p1, theta_p2]:
        predictions.append(np.clip(pred, theta_0-alpha-1e-9, theta_0+alpha+1e-9).round(4))
    
    return predictions

def generate_lr_predictions(dataset, theta_0: np.ndarray, alpha: float):
    theta_preds = np.load(f'../results/theta_preds/lr_{dataset.name}.npy')
    predictions = [theta_0]
    for theta_p in theta_preds:
        alphas = theta_p - theta_0
        theta = theta_0 - alphas
        predictions.append(np.clip(theta_p, theta_0-alpha-1e-9, theta_0+alpha+1e-9).round(4))
        predictions.append(np.clip(theta, theta_0-alpha-1e-9, theta_0+alpha+1e-9).round(4))
    return predictions

def generate_nn_smoothness_predictions(theta_0: np.ndarray, theta_s: np.ndarray, alpha: float):
    dist = np.min(np.abs(theta_0-theta_s))
    eps = np.round(np.abs(alpha - dist)/10, 2) 
    
    theta_p_minus_eps = np.clip(theta_s - 4.5*eps, theta_0-alpha, theta_0+alpha)
    theta_p_plus_eps = np.clip(theta_s + 4.5*eps, theta_0-alpha, theta_0+alpha)
    
    theta_p_minus_2eps = np.clip(theta_s - 9*eps, theta_0-alpha, theta_0+alpha)
    theta_p_plus_2eps = np.clip(theta_s + 9*eps, theta_0-alpha, theta_0+alpha)

    predictions = [
        (f'theta_s-2\epsilon', (theta_p_minus_2eps[:-1], theta_p_minus_2eps[[-1]])),
        (f'theta_s-\epsilon', (theta_p_minus_eps[:-1], theta_p_minus_eps[[-1]])), 
        ('theta_s', (theta_s[:-1], theta_s[[-1]])), 
        (f'theta_s+\epsilon', (theta_p_plus_eps[:-1], theta_p_plus_eps[[-1]])), 
        (f'theta_s+2\epsilon', (theta_p_plus_2eps[:-1], theta_p_plus_2eps[[-1]])),
    ]
    return predictions

def generate_lr_smoothness_predictions(theta_0, theta_s, alpha):
    dist = np.min(np.abs(theta_0-theta_s))
    eps = np.round(np.abs(alpha - dist)/10, 2) 
    
    theta_p_minus_eps = np.clip(theta_s - 4.5*eps, theta_0-alpha, theta_0+alpha)
    theta_p_plus_eps = np.clip(theta_s + 4.5*eps, theta_0-alpha, theta_0+alpha)
    
    theta_p_minus_2eps = np.clip(theta_s - 9*eps, theta_0-alpha, theta_0+alpha)
    theta_p_plus_2eps = np.clip(theta_s + 9*eps, theta_0-alpha, theta_0+alpha)

    predictions = [
        (f'theta_s-2\epsilon', (theta_p_minus_2eps[:-1], theta_p_minus_2eps[[-1]])),
        (f'theta_s-\epsilon', (theta_p_minus_eps[:-1], theta_p_minus_eps[[-1]])), 
        ('theta_s', (theta_s[:-1], theta_s[[-1]])), 
        (f'theta_s+\epsilon', (theta_p_plus_eps[:-1], theta_p_plus_eps[[-1]])), 
        (f'theta_s+2\epsilon', (theta_p_plus_2eps[:-1], theta_p_plus_2eps[[-1]])),
    ]
    return predictions

def hardmax(x, cat_indices):
    if cat_indices:
        max_i = cat_indices[0]
        max_val = x[max_i]
        for i in cat_indices:
            if x[i] > max_val:
                max_i = i
                max_val = x[i]
            x[i] = 0
        x[max_i] = 1
    return x


