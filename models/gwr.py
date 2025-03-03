from abc import ABCMeta, abstractmethod
from typing import Literal

import numba as nb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, MultiOutputMixin, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.utils._array_api import get_namespace
from sklearn.utils.validation import check_is_fitted, validate_data

type KernelType = Literal["gaussian", "bisquare", "exponential", "tricube", "triangular"]


class GeoLinearModel(BaseEstimator, metaclass=ABCMeta):
    @abstractmethod
    def fit(self, geo_x, X, y):
        pass

    def _decision_function(self, X):
        check_is_fitted(self)

        X = validate_data(self, X, accept_sparse=["csr", "csc", "coo"], reset=False)
        coef_ = self.coef_
        if coef_.ndim == 1:
            return X @ coef_ + self.intercept_
        return X @ coef_.T + self.intercept_

    def predict(self, X, Geo_x):
        """
        Predict using the linear model.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Samples.
        Geo_x : array-like or sparse matrix, shape (n_samples, n_features)

        Returns
        -------
        C : array, shape (n_samples,)
            Returns predicted values.
        """
        return self._decision_function(X)

    def _set_intercept(self, X_offset, y_offset, X_scale):
        """Set the intercept_"""

        xp, _ = get_namespace(X_offset, y_offset, X_scale)

        if self.fit_intercept:
            # We always want coef_.dtype=X.dtype. For instance, X.dtype can differ from
            # coef_.dtype if warm_start=True.
            coef_ = xp.astype(self.coef_, X_scale.dtype, copy=False)
            coef_ = self.coef_ = xp.divide(coef_, X_scale)

            if coef_.ndim == 1:
                intercept_ = y_offset - X_offset @ coef_
            else:
                intercept_ = y_offset - X_offset @ coef_.T

            self.intercept_ = intercept_

        else:
            self.intercept_ = 0.0


class GeoLinearRegression(MultiOutputMixin, RegressorMixin, GeoLinearModel):
    def __init__(
        self,
        bandwith: float = 0.5,
        kernel="gaussian",
        spherical=False,
        **lr_kwargs,
    ) -> None:
        self.bandwith = bandwith
        self.kernel = kernel
        self.spherical = spherical

        self.lr_kwargs = lr_kwargs

    @property
    def lr_args(self) -> dict:
        return {
            "fit_intercept": self.lr_kwargs.get("fit_intercept", True),
            "copy_X": self.lr_kwargs.get("copy_X", True),
            # "tol": self.lr_kwargs.get("tol", 1e-6),
            "n_jobs": self.lr_kwargs.get("n_jobs", -1),
            "positive": self.lr_kwargs.get("positive", False),
        }

    def _cdist(self, geo_x):
        assert geo_x.ndim == 2, "geo_x should be 2D array"
        if self.spherical:
            geo_x = to_cartesian(_make_sure_ndarray(geo_x))
        return np.linalg.norm(geo_x.reshape(-1, 1, 2) - geo_x, axis=2)

    def _locally_fit(self, i, X, y) -> np.ndarray:
        weight = self.weight_mat[i]

        return LinearRegression(**self.lr_args).fit(X, y, sample_weight=weight).coef_

    def fit(self, geo_x, X, y):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : object
            Returns self.
        """

        self.weight_mat = np.exp(-self._cdist(geo_x=geo_x) / self.bandwith**2)
        self.coef_array = np.array(
            Parallel(n_jobs=self.lr_args["n_jobs"])(delayed(self._locally_fit)(i, X, y) for i in range(X.shape[0])),
        )

        keys_geo_x = list(map(tuple, geo_x))
        self.coef = dict(zip(keys_geo_x, self.coef_array, strict=True))
        return self

    def fit_transform(self, geo_x, X, y):
        self.fit(geo_x, X, y)
        return self.transform(geo_x, X)

    def transform(self, geo_x, X):
        """
        Transform data by applying the linear model.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape (n_samples,)
            Returns predicted values.
        """
        try:
            coefs = np.array([self.coef[tuple(x)] for x in geo_x])
        except KeyError:
            msg = "Unseen geo_x is given"
            raise ValueError(msg) from None
        # X.shape => (n_samples, n_features)
        # coefs.shape => (n_samples, n_target, n_features)
        # pred.shape => (n_samples, n_target)
        if coefs.ndim == 3:
            X = X[:, np.newaxis, :]
        return np.sum(X * coefs, axis=-1)

    def compute_hat_matrix(self, x: np.ndarray) -> np.ndarray:
        return x @ np.linalg.inv(x.T @ self.weight_mat @ x) @ x.T @ self.weight_mat

    def score_aicc(self, geo_x, X, y) -> float:
        """
        Calculate corrected Akaike Information Criterion (AICc)

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples, n_targets)
            Target values.

        Returns
        -------
        aicc : float
            AICc value
        """
        n = X.shape[0]
        y_pred = self.transform(geo_x, X)
        residuals = y - y_pred

        s = self.compute_hat_matrix(X)
        trace_s = np.trace(s)

        res_std = np.sqrt(np.sum(residuals**2) / (len(y) - trace_s))
        return 2.0 * n * np.log(res_std) + n * np.log(2.0 * np.pi) + n * (n + trace_s) / (n - 2 - trace_s)


@nb.njit("f8[:,:](f8[:,:])", cache=True)
def to_cartesian(arr: np.ndarray) -> np.ndarray:
    """
    Convert latitude and longitude to cartesian coordinates
    """
    if arr.shape[1] == 2:
        msg = "arr should have 2 columns"
        raise ValueError(msg)

    lat, lon = np.radians(arr[:, 0]), np.radians(arr[:, 1])
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    # NOTE: numba does not support np.vstack with list of arrays
    return np.vstack((x, y))


def _make_sure_ndarray(data) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, list):
        return np.array(data)
    if isinstance(data, pd.Series):
        return data.to_numpy()
    if isinstance(data, pd.DataFrame):
        return data.to_numpy()

    msg = "data should be numpy array, list, pandas Series or DataFrame"
    raise ValueError(msg)
