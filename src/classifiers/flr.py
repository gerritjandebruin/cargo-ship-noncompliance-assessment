import pprint

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid


class CovarianceConstraintLogisticRegression:
    """
    Covariance-Constraint Logistic Regression
    """
    def __init__(self, **args):
        
        """
        base_covariance: {None, dict(str: float)}
            dictionary of sens-attr-val (key) to abs unconstrained-covariance-value
            unconstrained-covariance-value is the measured between logitraw prediciton
            and the binarised sens-attr-value vector from a normal Logistic Regression
            
            covariance(logitraw, bin-sens-attr-value) = 1/n * sum (logitraw * z),
            where z = bin-sens-attr-value - bin-sens-attr-value.mean()
                
            if None, computes the dictionary from additional  unconstrained fitting.
            * default: None
        
        cov_trehshold: {False, float, dict(str: float)}
            the constrained covariance value to apply during fitting
            if float, apply the same threshold to all sens-attr-values
            if dict, apply value-specific threshold
            if False, apply cov_coefficient instead
            * default: False
            
        cov_coefficient: {False, float, dict(str: float)}
            the coefficient value to apply to the base_covariance during fitting
            if float [0,1] -> apply the same coefficient to all sens-attr-values
            if dict, apply value-specific coefficient
            if False, apply cov_trehshold instead
            * default: 1e-1
        
        lambd: float
            constant that multiplies the penalty terms in elastic net
            * default: 0
        
        alpha: float
            controls the ratio between l1 and l2 as
                alpha*norm1(w) + (1-alpha)*norm2(w)
            must be in betwen [0,1]
        add_intercept: bool
            if the classifier should add a 1s column to serve as intercept
            * default: True
        """
        
        keys = ["base_covariance", "add_intercept", "cov_trehshold", "cov_coefficient"]
        args_keys = list(args.keys())
        
        if "base_covariance" not in args_keys:
            #self.base_covariance = None
            args["base_covariance"] = None
        
        if "add_intercept" not in args_keys:
            #self.add_intercept = True
            args["add_intercept"] = True
            
        if "cov_trehshold" not in args_keys:
            #self.cov_trehshold = False
            args["cov_trehshold"] = False
            
        if "cov_coefficient" not in args_keys:
            #self.cov_coefficient = 1e-1
            args["cov_coefficient"] = 1e-1
        
        if "lambd" not in args_keys:
            #self.alpha = 1
            args["lambd"] = 0
        
        if "alpha" not in args_keys:
            #self.alpha = 1
            args["alpha"] = 1
        
        self.args = args
        self.is_fit = False
    
    def fit(self, X, y, s, weights=None):
        """
        X: np.array.astype(float) shape(n,m)
        y: np.array.astype(int)   shape(n,)
        s: np.array.astype(str)   shape(n,)
        """
       
        if type(weights) == type(None):
            weights = np.ones_like(y)
        X = np.array(X).astype(float)
        y = np.array(y).astype(int)
        s = pd.get_dummies(np.array(s).astype(str))
        z = s - s.mean() #type: ignore
        
        if self.args["add_intercept"] == True:
            X = np.concatenate((X, np.ones(shape=(X.shape[0],1))), axis=1)

        if self.args["base_covariance"] == None:
            base_covariance = {}
            w = cp.Variable(X.shape[1])
            # logistic loss
            loss = -1 * cp.sum(
                cp.multiply(
                    weights,
                    cp.multiply(y, X @ w) - cp.logistic(X @ w)
                )
            ) / X.shape[0]
            # regularisation
            loss = loss + self.args["lambd"] * (
                self.args["alpha"]*cp.norm1(w[:-1]) + (1-self.args["alpha"])*cp.norm2(w[:-1])
            )
            obj = cp.Minimize(loss)
            prob = cp.Problem(obj)
            prob.solve(
                abstol=1e-2,
                reltol=1e-2,
                feastol=1e-2,                
                abstol_inacc=1e-2,
                reltol_inacc=1e-2,
                feastol_inacc=1e-2,                
                max_iters=10_000
            )
            pred = X @ w.value
            for attr_value in z.columns:
                base_covariance[attr_value] = abs(sum(pred * z[attr_value]) / X.shape[0])
            self.args["base_covariance"] = base_covariance
        
        # make constraints sens-atr-value specific
        constraints = []
        w = cp.Variable(X.shape[1])
        if self.args["cov_trehshold"] == False:
            cov_coefficient = self.args["cov_coefficient"]
            base_covariance = self.args["base_covariance"]
            if type(cov_coefficient) == type({}):
                for attr_value in z.columns:
                    cov_coef = cov_coefficient[attr_value]
                    cov_trehshold = base_covariance[attr_value] * cov_coef
                    covariance = cp.sum(cp.multiply(z[attr_value], X @ w)) / X.shape[0]
                    constraints.append(covariance >= -cov_trehshold)
                    constraints.append(covariance <= cov_trehshold)
            else:
                if cov_coefficient < 1: # if not, then we don't need constraints
                    for attr_value in z.columns:
                        cov_trehshold = base_covariance[attr_value] * cov_coefficient
                        covariance = cp.sum(cp.multiply(z[attr_value], X @ w)) / X.shape[0]
                        constraints.append(covariance >= -cov_trehshold)
                        constraints.append(covariance <= cov_trehshold)
        # logistic loss
        loss = -1 * cp.sum(
            cp.multiply(
                weights,
                cp.multiply(y, X @ w) - cp.logistic(X @ w)
            )
        ) / X.shape[0]
        # regularisation
        loss = loss + self.args["lambd"] * (
            self.args["alpha"]*cp.norm1(w[:-1]) + (1-self.args["alpha"])*cp.norm2(w[:-1])
        )
        obj = cp.Minimize(loss)
        prob = cp.Problem(obj, constraints=constraints)
        fun_value = prob.solve(
            # abstol=1e-2,
            # reltol=1e-2,
            # feastol=1e-2,                
            # abstol_inacc=1e-2,
            # reltol_inacc=1e-2,
            # feastol_inacc=1e-2,                
            max_iters=int(1e5)
        )
        self.is_fit = True
        self.coefs = w.value
        self.fun_value = fun_value
        
    def predict_raw(self, X):
        if self.args["add_intercept"] == True:
            X = np.concatenate((X, np.ones(shape=(X.shape[0],1))), axis=1)
        pred = np.array([-1 * X @ self.coefs, X @ self.coefs]).T
        return pred
    
    def predict_proba(self, X):
        return sigmoid(self.predict_raw(X))
    
    def predict(self, X):
        return np.argmax(self.predict_raw(X), axis=1)
    
    def __str__(self):
        args_str = pprint.pformat(self.args, indent=4)
        return("Covariance-Constraint Logistic Regression\n" + args_str)
    
    def __repr__(self):
        args_str = pprint.pformat(self.args, indent=4)
        return("Covariance-Constraint Logistic Regression\n" + args_str)