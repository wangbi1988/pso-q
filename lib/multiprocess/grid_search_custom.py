# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 14:21:24 2020

@author: bb
"""
import numpy as np;
from sklearn.model_selection import GridSearchCV, PredefinedSplit;
from sklearn.linear_model.base import BaseEstimator;
from sklearn.metrics import accuracy_score;

class abstractGSBE(BaseEstimator):
    def fit(self, x, y):
        raise NotImplementedError('estimator\'s fit function is not implemented');

    def predict(self, x):
        raise NotImplementedError('estimator\'s predict function is not implemented');
    
    def score(self, x, y):
        raise NotImplementedError('estimator\'s score function is not implemented');
    
    @staticmethod
    def search(elm, param_grid={"c":np.linspace(0.0001,1,10)}, n_jobs = -1, verbose = 1):
        X = np.arange(1);
        Y = np.zeros(1);
        cv = PredefinedSplit([-1] * (X.shape[0] - 1) + [0]);
        gs = GridSearchCV(elm, param_grid = param_grid, cv = cv, n_jobs = n_jobs, verbose = verbose);
        gs.fit(X, Y);
        return gs;
    
    def set_params(self, **params):
        from collections import defaultdict
        if not params:
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))
            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self
    
    @classmethod
    def _get_param_names(cls):
        import inspect
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            return []

        init_signature = inspect.signature(init)
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("scikit-learn estimators should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key, None)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out
    
class exampleGSBE(abstractGSBE):
    def __init__(self, job, c = 0):
        self.c = c;
        self.job = job;
        return None;
    
    def fit(self, x, y):
        return None;

    def predict(self, x):
        return np.random.randint(3, size = len(x))
    
    def score(self, x, y):
        return accuracy_score(y, self.predict(x)) * self.c

if __name__ == '__main__':
#    实际使用时，传入参数和任务即可
    gs = exampleGSBE.search(exampleGSBE(None));
    print(gs.best_params_)