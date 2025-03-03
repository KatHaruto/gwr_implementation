import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from models.gwr import GeoLinearRegression
from models.mgwr import MultiScaleGeoLinearRegression
from models.sgwr import SemiParametricGeoLinearRegression

data_num = 400
local_feature_num = 3
global_feature_num = 2

rng = np.random.default_rng(seed=123)

X = rng.random((data_num, local_feature_num + global_feature_num))
geo_x = rng.integers(0, 10, size=(data_num, 2)) / 10

beta = np.tile(np.array([1.0, 2.0, 3.0, 1.0, 0.5]), (data_num, 1))
beta[:, 0] = beta[:, 0] * (geo_x[:, 0] > 0.7)
beta[:, 1] = 1 + (geo_x[:, 0] + geo_x[:, 1]) / 6.0
beta[:, 2] = beta[:, 2] * np.exp(
    -((geo_x[:, 0] - 0.5) ** 2 + (geo_x[:, 1] - 0.5) ** 2) / 0.5,
)
y = np.sum(X * beta, axis=1) + rng.normal(0, 0.1, data_num)

test_num = 30
rand_idx = rng.choice(range(data_num), size=test_num)
test_geo_x = geo_x[rand_idx]
test_beta = beta[rand_idx]
test_x = rng.random((test_num, local_feature_num + global_feature_num))
test_y = np.sum(test_x * test_beta, axis=1) + rng.normal(0, 0.1, test_num)


lr = LinearRegression(
    fit_intercept=True,
    copy_X=True,
    n_jobs=1,
)


mgwr = MultiScaleGeoLinearRegression(
    initial_bandwith=10,
    kernel="gaussian",
    fit_intercept=True,
    copy_x=True,
    tol=1e-6,
    n_jobs=1,
    positive=False,
    spherical=False,
)

mgwr.fit(geo_x, X, y)
print(mgwr.coef_array.mean(axis=0), mgwr.coef_array.std(axis=0))
print(mgwr.bws)
mgwr_beta_pred = mgwr.coef_array

min_aicc = 1e10
for bandwith_i in [0.01, 0.1, 1, 10, 100, 1000]:
    gwr = GeoLinearRegression(
        fit_intercept=True,
        copy_X=True,
        tol=1e-6,
        n_jobs=1,
        positive=False,
        spherical=False,
        bandwith=bandwith_i,
    )
    gwr.fit(geo_x, X, y)

    aicc = gwr.score_aicc(geo_x, X, y)
    print(f"Bandwith: {bandwith_i}, AICC: {aicc}")
    if aicc < min_aicc:
        min_aicc = aicc
        best_bandwith = bandwith_i
        best_gwr = gwr
gwr_beta_pred = best_gwr.coef_array
print(f"Best bandwith: {best_bandwith}")

min_aicc = 1e10
for bandwith_i in [0.01, 0.1, 1, 10, 100, 1000]:
    sgwr = SemiParametricGeoLinearRegression(
        fit_intercept=True,
        copy_x=True,
        tol=1e-6,
        n_jobs=1,
        positive=False,
        spherical=False,
        bandwith=bandwith_i,
    )
    sgwr.fit(geo_x, X[:, local_feature_num:], X[:, :local_feature_num], y)

    aicc = sgwr.score_aicc(geo_x, X[:, local_feature_num:], X[:, :local_feature_num], y)
    print(f"Bandwith: {bandwith_i}, AICC: {aicc}")
    if aicc < min_aicc:
        min_aicc = aicc
        best_bandwith = bandwith_i
        best_sgwr = sgwr

print(f"Best bandwith: {best_bandwith}")
local_beta_pred = best_sgwr.local_coef_array
global_beta_pred = best_sgwr.global_coef


lr.fit(X, y)
y_pred = lr.predict(test_x)
r2 = r2_score(test_y, y_pred)
print(f"LR R2: {r2}")
print(f"LR Coef: {lr.coef_}")

y_pred = best_gwr.transform(test_geo_x, test_x)
r2 = r2_score(test_y, y_pred)
print(f"GWR R2: {r2}")
print(f"GWR Coef: {best_gwr.coef_array.mean(axis=0)}, {best_gwr.coef_array.std(axis=0)}")

y_pred = best_sgwr.transform(test_geo_x, test_x[:, local_feature_num:], test_x[:, :local_feature_num])
r2 = r2_score(test_y, y_pred)
print(f"SGWR R2: {r2}")
print(f"SGWR Coef: {best_sgwr.local_coef_array.mean(axis=0)}, {best_sgwr.local_coef_array.std(axis=0)}")
print(f"SGWR Global Coef: {best_sgwr.global_coef}")

y_pred = mgwr.transform(test_geo_x, test_x)
r2 = r2_score(test_y, y_pred)
print(f"MGWR R2: {r2}")
print(f"MGWR Coef: {mgwr.coef_array.mean(axis=0)}, {mgwr.coef_array.std(axis=0)}")
print(f"MGWR BandWidth: {mgwr.bws}")
# draw beta_pred with 3 subplots

# draw beta with 3 subplots
fig, axs = plt.subplots(5, 4)
fig.suptitle("Beta")
for i in range(5):
    vmin = min(
        beta[:, i].min(),
        gwr_beta_pred[:, i].min(),
        local_beta_pred[:, i].min() if i < local_feature_num else global_beta_pred[i - local_feature_num],
        mgwr_beta_pred[:, i].min(),
    )
    vmax = max(
        beta[:, i].max(),
        gwr_beta_pred[:, i].max(),
        local_beta_pred[:, i].max() if i < local_feature_num else global_beta_pred[i - local_feature_num],
        mgwr_beta_pred[:, i].max(),
    )
    plot_true = axs[i, 0].scatter(
        geo_x[:, 0],
        geo_x[:, 1],
        c=beta[:, i],
        vmin=vmin,
        vmax=vmax,
    )
    axs[i, 0].set_title(f"beta_{i}")

    plot_gw_pred = axs[i, 1].scatter(
        geo_x[:, 0],
        geo_x[:, 1],
        c=gwr_beta_pred[:, i],
        vmin=vmin,
        vmax=vmax,
    )
    axs[i, 1].set_title(f"GWR beta_pred_{i}")

    plot_sgw_pred = axs[i, 2].scatter(
        geo_x[:, 0],
        geo_x[:, 1],
        c=local_beta_pred[:, i]
        if i < local_feature_num
        else np.ones(data_num) * global_beta_pred[i - local_feature_num],
        vmin=vmin,
        vmax=vmax,
    )
    axs[i, 2].set_title(f"SGWR beta_pred_{i}")

    plot_mgw_pred = axs[i, 3].scatter(
        geo_x[:, 0],
        geo_x[:, 1],
        c=mgwr_beta_pred[:, i],
        vmin=vmin,
        vmax=vmax,
    )
    axs[i, 3].set_title(f"MGWR beta_pred_{i}")

    fig.colorbar(plot_true, ax=axs[i, 0])
    fig.colorbar(plot_gw_pred, ax=axs[i, 1])
    fig.colorbar(plot_sgw_pred, ax=axs[i, 2])
    fig.colorbar(plot_mgw_pred, ax=axs[i, 3])

plt.show()
