import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class WeightedEnsemble(BaseEstimator, RegressorMixin):
    """Trains multiple regressors and averages predictions.

    Unlike StackingRegressor, this class properly propagates sample_weight
    to every base estimator, which is essential for upweighting rare long-stay
    patients that would otherwise be swamped by the majority short-stay class.
    """

    def __init__(self, estimators):
        self.estimators = estimators  # list of (name, estimator)

    def fit(self, X, y, sample_weight=None):
        for _, est in self.estimators:
            if sample_weight is not None:
                est.fit(X, y, sample_weight=sample_weight)
            else:
                est.fit(X, y)
        return self

    def predict(self, X):
        preds = np.array([est.predict(X) for _, est in self.estimators])
        return preds.mean(axis=0)
