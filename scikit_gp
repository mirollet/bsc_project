from matplotlib import pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import Matern
import pandas as pd

def plot_gpr_samples(gpr_model, n_samples):
    """Plot samples drawn from the Gaussian process model.

    If the Gaussian process model is not trained then the drawn samples are
    drawn from the prior distribution. Otherwise, the samples are drawn from
    the posterior distribution. Be aware that a sample here corresponds to a
    function.

    Parameters
    ----------
    gpr_model : `GaussianProcessRegressor`
        A :class:`~sklearn.gaussian_process.GaussianProcessRegressor` model.
    n_samples : int
        The number of samples to draw from the Gaussian process distribution.
    ax : matplotlib axis
        The matplotlib axis where to plot the samples.
    """
    x = np.linspace(0, 10, 100)
    X = x.reshape(-1, 1)

    y_samples = gpr_model.sample_y(X, n_samples)

    for idx, single_prior in enumerate(y_samples.T):
        plt.plot(
            x,
            single_prior.flatten(),
            linestyle="--",
            alpha=0.7,
            label=f"Sampled function #{idx + 1}",
        )

train_x = np.array([0.05, 1.2, 1.5, 2, 2.2, 3, 5, 6, 9, 7]).reshape(-1,1)
train_y = np.array([5, 3, 2, 1.5, 1.6, 1.4, 0.2, 0.5, 0.1, 0.3]).reshape(-1,1)
X = np.linspace(0, 10, 100).reshape(-1,1)

#kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
kernel = 1 * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

mean_prediction, cov_prediciton = gaussian_process.predict(X, return_std= False, return_cov=True)
plt.figure(figsize=(8,6))
plt.imshow(cov_prediciton, interpolation="none", origin="upper")
plt.colorbar(label="$covariance$")
plt.xlabel("$r$")
plt.ylabel("$r$")
plt.title("Covariance matrix, RBF kernel, pre GP fit", fontsize=14)
#plt.title("Covariance matrix, Matern kernel, pre GP fit", fontsize=14)
#plt.savefig("pre_cov_rbf.png", dpi=300)
plt.savefig("pre_cov_matern.png", dpi=300)


gaussian_process.fit(train_x, train_y)
print(gaussian_process.kernel_)

mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)
mean_prediction, cov_prediciton = gaussian_process.predict(X, return_std= False, return_cov=True)
mean_prediction = mean_prediction.flatten()

lower = mean_prediction - 1.96 * std_prediction
upper = mean_prediction + 1.96 * std_prediction

plt.figure(figsize=(8,6))
plt.fill_between(
    X.ravel(),
    upper,
    lower,
    alpha=0.5,
    label=r"95% confidence interval",
    color = "lightgrey"
)
plot_gpr_samples(gaussian_process, 5)
plt.plot(X, mean_prediction, label="Mean prediction", c="k", zorder=6)
plt.scatter(train_x, train_y, label="Observations", zorder=7, c="dodgerblue")
plt.legend(bbox_to_anchor=(1,1), ncol=2)
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
#plt.title("Gaussian process regression on a synthetic, noise-free dataset (RBF kernel)")
plt.title("Gaussian process regression on a synthetic, noise-free dataset (Matern kernel)")
#plt.savefig('scikit_gp_rbf_fit.png', dpi=200)
plt.savefig('scikit_gp_matern_fit.png', dpi=200)


plt.figure(figsize=(8,6))
plt.imshow(cov_prediciton, interpolation="none", origin="upper")
plt.colorbar(label="$covariance$")
#plt.title("Covariance matrix, RBF kernel, post GP fit", fontsize=14)
plt.title("Covariance matrix, Matern kernel, post GP fit", fontsize=14)
plt.xlabel("$r$")
plt.ylabel("$r$")
#plt.savefig("post_cov_rbf.png", dpi=300)
plt.savefig("post_cov_matern.png", dpi=300)
