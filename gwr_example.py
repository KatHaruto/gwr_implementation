import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from models.gwr import GeoLinearRegression

data_num = 300
feature_num = 3
X = np.random.rand(data_num, feature_num)
geo_x = np.random.randint(0, 10, size=(data_num, 2)) / 10

beta = np.tile(np.array([1.0, 2.0, 3.0]), (data_num, 1))
beta[:, 0] = beta[:, 0] * geo_x[:, 0] > 0.7
beta[:, 1] = beta[:, 1] * geo_x[:, 1] < 0.3
beta[:, 2] = beta[:, 2] * np.exp(
    -((geo_x[:, 0] - 0.5) ** 2 + (geo_x[:, 1] - 0.5) ** 2) / 0.5,
)

y = np.sum(X * beta, axis=1) + np.random.normal(0, 0.1, data_num)

test_num = 30
rand_idx = np.random.choice(range(data_num), size=test_num)
test_geo_x = geo_x[rand_idx]
test_beta = beta[rand_idx]
test_x = np.random.rand(test_num, feature_num)
test_y = np.sum(test_x * test_beta, axis=1) + np.random.normal(0, 0.1, test_num)


lr = LinearRegression(
    fit_intercept=True,
    copy_X=True,
    n_jobs=1,
)

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

print(f"Best bandwith: {best_bandwith}")
beta_pred = best_gwr.coef_array


lr.fit(X, y)
y_pred = lr.predict(test_x)
r2 = r2_score(test_y, y_pred)
print(f"LR R2: {r2}")

y_pred = best_gwr.transform(test_geo_x, test_x)
r2 = r2_score(test_y, y_pred)
print(f"GWR R2: {r2}")
# draw beta_pred with 3 subplots

# draw beta with 3 subplots
fig, axs = plt.subplots(3, 2)
fig.suptitle("Beta")
for i in range(3):
    plot_true = axs[i, 0].scatter(
        geo_x[:, 0],
        geo_x[:, 1],
        c=beta[:, i],
    )
    axs[i, 0].set_title(f"beta_{i}")

    plot_pred = axs[i, 1].scatter(
        geo_x[:, 0],
        geo_x[:, 1],
        c=beta_pred[:, i],
    )
    axs[i, 1].set_title(f"beta_pred_{i}")

    fig.colorbar(plot_true, ax=axs[i, 0])
    fig.colorbar(plot_pred, ax=axs[i, 1])


plt.show()
