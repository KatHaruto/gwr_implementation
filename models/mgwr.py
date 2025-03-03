from typing import Self

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.base import MultiOutputMixin, RegressorMixin
from tqdm import tqdm

from models.gwr import GeoLinearModel, GeoLinearRegression, KernelType


class MultiScaleGeoLinearRegression(MultiOutputMixin, RegressorMixin, GeoLinearModel):
    def __init__(
        self,
        initial_bandwith: float = 0.5,
        kernel: KernelType = "gaussian",
        epsilon: float = 1e-6,
        max_iter: int = 3,
        *,
        fit_intercept: bool = True,
        copy_x: bool = True,
        tol: float = 1e-6,
        n_jobs: int | None = None,
        positive: bool = False,
        spherical: bool = False,
    ) -> None:
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.initial_bandwith = initial_bandwith
        self.kernel = kernel
        self.fit_intercept = fit_intercept
        self.copy_X = copy_x
        self.tol = tol
        self.n_jobs = n_jobs
        self.positive = positive
        self.spherical = spherical

    def create_gwr_model(self, bandwidth=None) -> GeoLinearRegression:
        return GeoLinearRegression(
            bandwith=bandwidth if bandwidth else self.initial_bandwith,
            kernel=self.kernel,
            fit_intercept=self.fit_intercept,
            copy_X=self.copy_X,
            tol=self.tol,
            n_jobs=self.n_jobs,
            positive=self.positive,
            spherical=self.spherical,
        )

    def fit(self, geo_x, X, y) -> Self:
        """
        Fit the model according to the given training data.

        Algorithm is as follows:
            1. For each column of X_q, regress the column against X_{p-q} using the GWR and then compute the residual from that regression (X_q residuals).
            2. Regress Y against X_{p-q} using the GWR.
            3. Compute the residual from the above regression (Y residuals).
            4. Regress the Y residuals against X_{q} residuals using the OLS. This gives the estimate βˆq.
            5. Subtract X_qβ^q from Y . Regress this against X_{p-q} using the GWR to obtain estimators for the geographically varying (βˆ{p−q}) coefficients.
        """

        # epsilong = 10  # some large value
        _gwr = self.create_gwr_model().fit(geo_x, X, y)
        current_iter = 0

        bws = np.ones(X.shape[1]) * self.initial_bandwith
        current_beta = _gwr.coef_array
        # while current_iter < self.max_iter:
        for _ in tqdm(range(self.max_iter)):
            for i in range(X.shape[1]):
                err = y - np.sum(X * current_beta, axis=1)
                err_except_i = err + X[:, i] * current_beta[:, i]

                def _fit_bandwidth(bw: float, geo_x, x, y) -> float:
                    gwr = self.create_gwr_model(bandwidth=bw).fit(geo_x, x, y)
                    return gwr.score_aicc(geo_x, x, y)

                res = minimize_scalar(
                    lambda bw: _fit_bandwidth(bw, geo_x, X[:, i : i + 1], err_except_i),
                    bounds=(1e-3, 1000),
                    method="bounded",
                    options={"maxiter": 50},
                )

                bws[i] = res.x

                current_beta[:, i : i + 1] = (
                    self.create_gwr_model(bandwidth=bws[i]).fit(geo_x, X[:, i : i + 1], err_except_i).coef_array
                )
            print(bws)
            # current_iter += 1
        self.coef_array = current_beta

        keys_geo_x = list(map(tuple, geo_x))
        self.coef = dict(zip(keys_geo_x, self.coef_array, strict=True))

        self.bws = bws

        return self

    def transform(self, geo_x, x) -> np.ndarray:
        try:
            coef = np.array([self.coef[tuple(x)] for x in geo_x])
        except KeyError:
            msg = "Unseen geo_x is given"
            raise ValueError(msg) from None
        # X.shape => (n_samples, n_features)
        # coefs.shape => (n_samples, n_target, n_features)
        # pred.shape => (n_samples, n_target)
        if coef.ndim == 3:
            x = x[:, np.newaxis, :]
        return np.sum(x * coef, axis=-1)

    def fit_transform(self, geo_x, x, y) -> np.ndarray:
        self.fit(geo_x, x, y)
        return self.transform(geo_x, x)

    def compute_w(self) -> np.ndarray:
        a = np.eye(self.last_s.shape[0]) - self.last_s
        return a.T @ a

    def compute_hat_matrix(self, global_x: np.ndarray) -> np.ndarray:
        w = self.compute_w()
        return global_x @ np.linalg.inv(global_x.T @ w @ global_x) @ global_x.T @ w + self.last_s

    def score_aicc(self, geo_x, x, y) -> float:
        n = x.shape[0]
        y_pred = self.transform(geo_x, x)
        residuals = y - y_pred

        s = self.compute_hat_matrix(x)
        trace_s = np.trace(s)

        res_std = np.sqrt(np.sum(residuals**2) / (len(y) - trace_s))
        return 2.0 * n * np.log(res_std) + n * np.log(2.0 * np.pi) + n * (n + trace_s) / (n - 2 - trace_s)
