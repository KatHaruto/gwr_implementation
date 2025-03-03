from typing import Self

import numpy as np
from sklearn.base import MultiOutputMixin, RegressorMixin
from sklearn.linear_model import LinearRegression

from models.gwr import GeoLinearModel, GeoLinearRegression, KernelType


class SemiParametricGeoLinearRegression(MultiOutputMixin, RegressorMixin, GeoLinearModel):
    def __init__(
        self,
        bandwith: float = 0.5,
        kernel: KernelType = "gaussian",
        *,
        fit_intercept: bool = True,
        copy_x: bool = True,
        tol: float = 1e-6,
        n_jobs: int | None = None,
        positive: bool = False,
        spherical: bool = False,
    ) -> None:
        self.bandwith = bandwith
        self.kernel = kernel
        self.fit_intercept = fit_intercept
        self.copy_X = copy_x
        self.tol = tol
        self.n_jobs = n_jobs
        self.positive = positive
        self.spherical = spherical

    def create_gwr_model(self) -> GeoLinearRegression:
        return GeoLinearRegression(
            bandwith=self.bandwith,
            kernel=self.kernel,
            fit_intercept=self.fit_intercept,
            copy_X=self.copy_X,
            tol=self.tol,
            n_jobs=self.n_jobs,
            positive=self.positive,
            spherical=self.spherical,
        )

    def fit(self, geo_x, global_x, local_x, y) -> Self:
        """
        Fit the model according to the given training data.

        Algorithm is as follows:
            1. For each column of X_q, regress the column against X_{p-q} using the GWR and then compute the residual from that regression (X_q residuals).
            2. Regress Y against X_{p-q} using the GWR.
            3. Compute the residual from the above regression (Y residuals).
            4. Regress the Y residuals against X_{q} residuals using the OLS. This gives the estimate βˆq.
            5. Subtract X_qβ^q from Y . Regress this against X_{p-q} using the GWR to obtain estimators for the geographically varying (βˆ{p−q}) coefficients.
        """

        lr = LinearRegression(
            fit_intercept=self.fit_intercept,
            copy_X=self.copy_X,
            n_jobs=self.n_jobs,
            positive=self.positive,
        )

        xi_res = global_x - self.create_gwr_model().fit_transform(geo_x, local_x, global_x)
        y_res = y - self.create_gwr_model().fit_transform(geo_x, local_x, y)

        self.global_coef = lr.fit(xi_res, y_res).coef_
        gwr3 = self.create_gwr_model().fit(geo_x, local_x, y - global_x @ self.global_coef)
        self.local_coef_array = gwr3.coef_array

        self.last_s = gwr3.compute_hat_matrix(local_x)
        keys_geo_x = list(map(tuple, geo_x))
        self.local_coef = dict(zip(keys_geo_x, self.local_coef_array, strict=True))

        return self

    def transform(self, geo_x, global_x, local_x) -> np.ndarray:
        try:
            local_coef = np.array([self.local_coef[tuple(x)] for x in geo_x])
        except KeyError:
            msg = "Unseen geo_x is given"
            raise ValueError(msg) from None
        # X.shape => (n_samples, n_features)
        # coefs.shape => (n_samples, n_target, n_features)
        # pred.shape => (n_samples, n_target)
        if local_coef.ndim == 3:
            local_x = local_x[:, np.newaxis, :]
            global_x = global_x[:, np.newaxis, :]
        return np.sum(local_x * local_coef, axis=-1) + np.sum(global_x * self.global_coef, axis=-1)

    def fit_transform(self, geo_x, global_x, local_x, y) -> np.ndarray:
        self.fit(geo_x, global_x, local_x, y)
        return self.transform(geo_x, global_x, local_x)

    def compute_w(self) -> np.ndarray:
        a = np.eye(self.last_s.shape[0]) - self.last_s
        return a.T @ a

    def compute_hat_matrix(self, global_x: np.ndarray) -> np.ndarray:
        w = self.compute_w()
        return global_x @ np.linalg.inv(global_x.T @ w @ global_x) @ global_x.T @ w + self.last_s

    def score_aicc(self, geo_x, global_x, local_x, y) -> float:
        n = local_x.shape[0]
        y_pred = self.transform(geo_x, global_x, local_x)
        residuals = y - y_pred

        s = self.compute_hat_matrix(global_x)
        trace_s = np.trace(s)

        res_std = np.sqrt(np.sum(residuals**2) / (len(y) - trace_s))
        return 2.0 * n * np.log(res_std) + n * np.log(2.0 * np.pi) + n * (n + trace_s) / (n - 2 - trace_s)
