import numpy as np
from sklearn import datasets

COLORS = [
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#f781bf",
    "#a65628",
    "#984ea3",
    "#999999",
    "#e41a1c",
    "#dede00",
    "black",
]

MARKERS = [
    "o",
    "v",
    "*",
    "D",
    "X",
    "^",
    ">",
    "1",
    "p",
]


def get_datasets():

    # ============
    # Generate datasets. We choose the size big enough to see the scalability
    # of the algorithms, but not too big to avoid too long running times
    # ============
    n_samples = 500
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05)
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
    no_structure = np.random.rand(n_samples, 2), None

    # Anisotropicly distributed data
    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    # blobs with varied variances
    varied = datasets.make_blobs(
        n_samples=n_samples,
        cluster_std=[1.0, 2.5, 0.5],
        random_state=random_state,
    )

    default_base = {
        "quantile": 0.3,
        "eps": 0.3,
        "damping": 0.9,
        "preference": -200,
        "n_neighbors": 3,
        "n_clusters": 3,
        "min_samples": 7,
        "xi": 0.05,
        "min_cluster_size": 0.1,
    }

    plot_datasets = [
        (
            noisy_circles,
            {
                "damping": 0.77,
                "preference": -240,
                "quantile": 0.2,
                "n_clusters": 2,
                "min_samples": 7,
                "xi": 0.08,
            },
        ),
        (
            noisy_moons,
            {
                "damping": 0.75,
                "preference": -220,
                "n_clusters": 2,
                "min_samples": 7,
                "xi": 0.1,
            },
        ),
        (
            varied,
            {
                "eps": 0.18,
                "n_neighbors": 2,
                "min_samples": 7,
                "xi": 0.01,
                "min_cluster_size": 0.2,
            },
        ),
        (
            aniso,
            {
                "eps": 0.15,
                "n_neighbors": 2,
                "min_samples": 7,
                "xi": 0.1,
                "min_cluster_size": 0.2,
            },
        ),
        (blobs, {"min_samples": 7, "xi": 0.1, "min_cluster_size": 0.2}),
        (no_structure, {}),
    ]
    return plot_datasets, default_base
