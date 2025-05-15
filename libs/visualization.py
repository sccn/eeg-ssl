import matplotlib.pyplot as plt

# unused but required import for doing 3d projections with matplotlib < 3.2
import mpl_toolkits.mplot3d  # noqa: F401
from matplotlib import ticker
# plot the embeddings using t-SNE and color by the labels using their histogram bins
# plot the metrics for each attribute for each version from scores
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn import datasets, manifold

def plot_3d(points, points_color, title):
    x, y, z = points.T

    fig, ax = plt.subplots(
        figsize=(10, 8),
        facecolor="white",
        tight_layout=True,
        subplot_kw={"projection": "3d"},
    )
    fig.suptitle(title, size=16)
    col = ax.scatter(x, y, z, c=points_color, s=5, alpha=0.8)
    ax.view_init(azim=-60, elev=9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)
    plt.show()


def plot_2d(points, points_color, title):
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    add_2d_scatter(ax, points, points_color)
    plt.show()


def add_2d_scatter(ax, points, points_color, title=None):
    x, y = points.T
    ax.scatter(x, y, c=points_color, s=5, alpha=0.8)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())

def plot_linear(embeddings, labels, n_neighbors=12, n_components=2):
    params = {
        "n_neighbors": n_neighbors,
        "n_components": n_components,
        "eigen_solver": "dense",
        "random_state": 0,
    }

    lle_standard = manifold.LocallyLinearEmbedding(method="standard", **params)
    S_standard = lle_standard.fit_transform(embeddings)

    # lle_ltsa = manifold.LocallyLinearEmbedding(method="ltsa", **params)
    # S_ltsa = lle_ltsa.fit_transform(embeddings)

    # lle_hessian = manifold.LocallyLinearEmbedding(method="hessian", **params)
    # S_hessian = lle_hessian.fit_transform(embeddings)

    # lle_mod = manifold.LocallyLinearEmbedding(method="modified", **params)
    # S_mod = lle_mod.fit_transform(embeddings)

    fig, axs = plt.subplots(
        nrows=2, ncols=2, figsize=(7, 7), facecolor="white", constrained_layout=True
    )
    fig.suptitle("Locally Linear Embeddings", size=16)

    lle_methods = [
        ("Standard locally linear embedding", S_standard),
        # ("Local tangent space alignment", S_ltsa),
        # ("Hessian eigenmap", S_hessian),
        # ("Modified locally linear embedding", S_mod),
    ]
    for ax, method in zip(axs.flat, lle_methods):
        name, points = method
        add_2d_scatter(ax, points, labels, name)

    plt.show()

def plot_tsne(embeddings, labels, n_components=2):
    # Standardize the data
    # scaler = StandardScaler()
    # embeddings = scaler.fit_transform(embeddings)

    # # Reduce the dimensionality of the data using PCA
    # pca = PCA(n_components=n_components)
    # embeddings_pca = pca.fit_transform(embeddings)

    # Create a t-SNE object and fit it to the PCA-reduced data
    tsne = TSNE(n_components=n_components, perplexity=60)
    embeddings_tsne = tsne.fit_transform(embeddings)

    # Create a scatter plot of the t-SNE results
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=labels, cmap='viridis', s=5)
    plt.colorbar()
    plt.title('t-SNE of Embeddings')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()