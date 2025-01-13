from sklearn import datasets, manifold
import umap
import matplotlib.pyplot as plt
from matplotlib import ticker

def get_subject_clusters_assignment(cluster_labels, points_subjects_ids, subjects):
    '''
    Return a dictionary with cluster labels as ids and list of subjects ids as values
    Assign cluster to subject with majority voting
    '''
    assert len(cluster_labels) == len(points_subjects_ids)
    subjects_clusters_dict = {}
    for i, subj in enumerate(points_subjects_ids):
        if subj not in subjects_clusters_dict:
            subjects_clusters_dict[subj] = [cluster_labels[i]]
        else:
            subjects_clusters_dict[subj].append(cluster_labels[i])

    cluster_subjects_dict = {}
    for subj in subjects:
        most_frequent_cluster = max(subjects_clusters_dict[subj],key=subjects_clusters_dict[subj].count)
        if most_frequent_cluster not in cluster_subjects_dict:
            cluster_subjects_dict[most_frequent_cluster] = [subj]
        else:
            cluster_subjects_dict[most_frequent_cluster].append(subj)
    sum_clusters_subjects = sum([len(cluster_subjects_dict[key]) for key in cluster_subjects_dict])
    assert sum_clusters_subjects == len(subjects)
    return cluster_subjects_dict

def get_tsne_umap(embs):
    t_sne = manifold.TSNE(
        n_components=2,
        perplexity=100,
        init="random",
        # max_iter=250,
        random_state=0,
    )
    S_t_sne = t_sne.fit_transform(embs)
    fit = umap.UMAP()
    u = fit.fit_transform(embs)

    return S_t_sne, u

def plot_3d(points, points_color, title):
    x, y, z = points.T

    fig, ax = plt.subplots(
        figsize=(6, 6),
        facecolor="white",
        tight_layout=True,
        subplot_kw={"projection": "3d"},
    )
    fig.suptitle(title, size=16)
    col = ax.scatter(x, y, z, c=points_color, s=50, alpha=0.8)
    ax.view_init(azim=-60, elev=9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)
    plt.show()


def plot_2d(points, points_color, title):
    fig, ax = plt.subplots(figsize=(3, 3), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    scatter = add_2d_scatter(ax, points, points_color)
    plt.colorbar(scatter, ax=ax)
    plt.show()


def add_2d_scatter(ax, points, points_color, title=None):
    x, y = points.T
    scatter = ax.scatter(x, y, c=points_color, s=50, alpha=0.8)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    return scatter