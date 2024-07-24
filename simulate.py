import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs


def to_float(*args):
    return [float(arg) for arg in args]


def generate_cluster_data(n_samples, centers, stds, sample):
    # To-do: refractor
    label_healthy = "Non-leukemic"
    label_diseased = "Leukemic"
    label_x = "LAIP1"
    label_y = "LAIP2"
    X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=stds, random_state=0)
    df = pd.DataFrame(X, columns=[label_x, label_y])
    df["Cluster"] = np.where(y == 1, label_diseased, label_healthy)
    df["Sample"] = sample
    return df


def simulate_samples(n_cells, frac_leukemic_AML1, frac_leukemic_AML2, frac_leukemic_AML3,
                     mean_healthy_x, std_healthy_x, mean_healthy_y, std_healthy_y,
                     dist_healthy_LAIP1, std_LAIP1_x, std_LAIP1_y,
                     dist_healthy_LAIP2, std_LAIP2_x, std_LAIP2_y):
    # Convert string widget input into floats
    (n_cells, frac_leukemic_AML1, frac_leukemic_AML2, frac_leukemic_AML3,
     mean_healthy_x, std_healthy_x, mean_healthy_y, std_healthy_y,
     dist_healthy_LAIP1, std_LAIP1_x, std_LAIP1_y,
     dist_healthy_LAIP2, std_LAIP2_x, std_LAIP2_y) = to_float(
        n_cells, frac_leukemic_AML1, frac_leukemic_AML2, frac_leukemic_AML3,
        mean_healthy_x, std_healthy_x, mean_healthy_y, std_healthy_y,
        dist_healthy_LAIP1, std_LAIP1_x, std_LAIP1_y,
        dist_healthy_LAIP2, std_LAIP2_x, std_LAIP2_y)

    # Setup clusters
    healthy_means = [mean_healthy_x, mean_healthy_y]
    healthy_stds = [std_healthy_x, std_healthy_y]
    LAIP1_means = [mean_healthy_x + dist_healthy_LAIP1, mean_healthy_y]
    LAIP1_stds = [std_LAIP1_x, std_LAIP1_y]
    LAIP2_means = [mean_healthy_x, mean_healthy_y + dist_healthy_LAIP2]
    LAIP2_stds = [std_LAIP2_x, std_LAIP2_y]

    # Combine clusters
    centers_NBM = [healthy_means]
    centers_AML1 = [healthy_means, LAIP1_means]
    centers_AML2 = centers_AML1
    centers_AML3 = [healthy_means, LAIP2_means]
    stds_NBM = [healthy_stds]
    stds_AML1 = [healthy_stds, LAIP1_stds]
    stds_AML2 = [healthy_stds, LAIP1_stds]
    stds_AML3 = [healthy_stds, LAIP2_stds]

    n_cells_AML1 = [round(n_cells * (1 - frac_leukemic_AML1)), round(n_cells * frac_leukemic_AML1)]
    n_cells_AML2 = [round(n_cells * (1 - frac_leukemic_AML2)), round(n_cells * frac_leukemic_AML2)]
    n_cells_AML3 = [round(n_cells * (1 - frac_leukemic_AML3)), round(n_cells * frac_leukemic_AML3)]

    # Generate data for each condition
    NBM = generate_cluster_data(int(n_cells), centers_NBM, stds_NBM, "NBM")
    AML1 = generate_cluster_data(n_cells_AML1, centers_AML1, stds_AML1, "AML1")
    AML2 = generate_cluster_data(n_cells_AML2, centers_AML2, stds_AML2, "AML2")
    AML3 = generate_cluster_data(n_cells_AML3, centers_AML3, stds_AML3, "AML3")
    return NBM, AML1, AML2, AML3