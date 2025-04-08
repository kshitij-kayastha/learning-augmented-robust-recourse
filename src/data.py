# Author: Kshitij Kayastha
# Date: Feb 3, 2025


import numpy as np
import pandas as pd
from typing import List
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler

class Dataset(ABC):
    def __init__(self, seed: int = 0, n_folds: int = 5):
        super().__init__()
        self.seed = seed
        self.n_folds = n_folds
        self.X = None
        self.y = None
        self.X_shift = None
        self.y_shift = None
        self.name = None
        self.scaler = None
    
    def get_feature_types(self, df: pd.DataFrame):
        cat_features, num_features = [], []
        for feature in df.columns:
            if df[feature].dtype == object:
                cat_features.append(feature)
            elif len(set(df[feature])) > 2:
                num_features.append(feature)
        return cat_features, num_features
    
    def scale_num_features(self, df: pd.DataFrame, num_features: List[str]):
        self.scaler = StandardScaler()
        df[num_features] = self.scaler.fit_transform(df[num_features].values)
        return df
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, fold: int):
        X_chunks, y_chunks = [], []
        for i in range(self.n_folds):
            start = int(i/self.n_folds * len(X))
            end = int((i+1)/self.n_folds * len(X))
            X_chunks.append(X.iloc[start:end])
            y_chunks.append(y.iloc[start:end])
            
        X_test, y_test = X_chunks.pop(fold), y_chunks.pop(fold)
        X_train, y_train = pd.concat(X_chunks), pd.concat(y_chunks)
        
        return (X_train, y_train), (X_test, y_test)
    
    def get_data(self, fold: int, shift: bool = False):
        fold = fold % self.n_folds
        if shift:
            return self.split_data(self.X, self.y, fold), self.split_data(self.X_shift, self.y_shift, fold)
        else:
            return self.split_data(self.X, self.y, fold)
            
            
            
class SyntheticDataset(Dataset):
    def __init__(self, n: int = 1000, d: int = 2, seed = 0, n_folds = 5, alpha: float = 0.1, beta: float = 0.):
        super(SyntheticDataset, self).__init__(seed, n_folds)
        self.n = n
        self.d = d
        c0_means = -2*np.ones(d)
        c1_means = 2*np.ones(d)
        c0_cov = 0.5*np.eye(d)
        c1_cov = 0.5*np.eye(d)
        
        self.X, self.y = self.create(c0_means, c1_means, c0_cov, c1_cov)
        self.X_shift, self.y_shift = self.create(c0_means+alpha, c1_means+alpha, (1+beta)*c0_cov, (1+beta)*c1_cov)
        self.name = 'synthetic'
        self.cat_features, self.num_features = [], [0,1]
        self.imm_features = []
    
    def create(self, c0_means, c1_means, c0_cov, c1_cov):
        rng = np.random.default_rng(self.seed)
        
        X0 = rng.multivariate_normal(c0_means, c0_cov, int(self.n/2))
        X1 = rng.multivariate_normal(c1_means, c1_cov, int(self.n/2))
        X = np.vstack((X0,X1))
        y = np.array([0]*int(self.n/2) + [1]*int(self.n/2)).reshape(-1,1)
        data = np.hstack((X, y))
        rng.shuffle(data)
        
        df = pd.DataFrame(data, columns=['X0', 'X1', 'y'])
        X, y = df.drop(columns=['y']), df['y']
        return X, y
        
        
class GermanDataset(Dataset):
    def __init__(self, seed = 0, n_folds = 5):
        super(GermanDataset, self).__init__(seed, n_folds)
        self.X, self.y = self.create('../datasets/german.csv')
        self.X_shift, self.y_shift = self.create('../datasets/corrected_german.csv')
        self.name = 'german'
        self.cat_features, self.num_features = list(range(3,7)), list(range(2))
        self.imm_features = [2]
        
    def create(self, filepath):
        df = pd.read_csv(filepath, sep=',').sample(frac=1, random_state=self.seed)
        
        cat_features, num_features = ['personal_status_sex'], ['duration', 'amount', 'age']
        
        target = 'credit_risk'
        
        df = df.drop(columns=[c for c in list(df) if c not in num_features+cat_features+[target]])
        df = self.scale_num_features(df, num_features)
        df = pd.get_dummies(df, columns=cat_features, dtype=float)
        
        X, y = df.drop(columns=[target]), df[target]
        return X, y
        


class SBADataset(Dataset):
    def __init__(self, seed = 0, n_folds = 5):
        super(SBADataset, self).__init__(seed, n_folds)
        
        df = pd.read_csv('../datasets/SBAcase.11.13.17.csv', sep=',').sample(frac=1, random_state=self.seed).fillna(-1)
        df['NoDefault'] = 1-df['Default'].values
        df = df.drop(columns=['Selected', 'State','Name', 'BalanceGross', 'LowDoc', 'BankState', 'LoanNr_ChkDgt', 'MIS_Status', 'Default', 'Bank', 'City'])
        
        cat_features, num_features = self.get_feature_types(df)
        target = 'NoDefault'
        
        df = df[num_features+cat_features+[target]]
        df = pd.get_dummies(df, columns=cat_features, dtype=float)
        
        df_shift = df.copy()
        df = df[df['ApprovalFY'] < 2006]
        
        df = self.scale_num_features(df, num_features)
        df_shift = self.scale_num_features(df_shift, num_features)
        
        self.X, self.y = df.drop(columns=[target]), df[target]
        self.X_shift, self.y_shift = df_shift.drop(columns=[target]), df_shift[target]
        self.name = 'sba'
        
        self.cat_features, self.num_features = list(range(20,25)), list(range(20))
        self.imm_features = []
        