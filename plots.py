import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve, auc
from scipy.spatial import cKDTree
from matplotlib.colors import LinearSegmentedColormap
from simulate import simulate_samples
from scipy.spatial import ConvexHull

def plot_simulation(n_cells, frac_leukemic_AML1, frac_leukemic_AML2, frac_leukemic_AML3,
                    mean_healthy_x, std_healthy_x, mean_healthy_y, std_healthy_y,
                    dist_healthy_LAIP1, std_LAIP1_x, std_LAIP1_y,
                    dist_healthy_LAIP2, std_LAIP2_x, std_LAIP2_y):
    
    def plot_subplot(data, title, ax):
        # To-do: refractor
        label_healthy = "Non-leukemic"
        label_diseased = "Leukemic"
        label_x = "LAIP1"
        label_y = "LAIP2"
        sns.scatterplot(data=data, x=label_x, y=label_y, hue="Cluster", hue_order=[label_healthy, label_diseased],
        s=4, palette=["#009292ff", "#ff7f00ff"], ax=ax, legend=None, rasterized=True)
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_title(title)

    NBM, AML1, AML2, AML3 = simulate_samples(n_cells, frac_leukemic_AML1, frac_leukemic_AML2, frac_leukemic_AML3,
                                             mean_healthy_x, std_healthy_x, mean_healthy_y, std_healthy_y,
                                             dist_healthy_LAIP1, std_LAIP1_x, std_LAIP1_y,
                                             dist_healthy_LAIP2, std_LAIP2_x, std_LAIP2_y)

    xlims = (-3, 4)
    ylims = (-3, 4)

    fig = plt.figure(figsize=(7, 2))
    gs = gridspec.GridSpec(1, 4, figure=fig)
    
    # Subplots
    ax1 = fig.add_subplot(gs[0, 0])
    plot_subplot(NBM, "NBM", ax1)
    
    ax2 = fig.add_subplot(gs[0, 1])
    plot_subplot(AML1, "AML1", ax2)
    
    ax3 = fig.add_subplot(gs[0, 2])
    plot_subplot(AML2, "AML2", ax3)
    
    ax4 = fig.add_subplot(gs[0, 3])
    plot_subplot(AML3, "AML3", ax4)

    plt.tight_layout()
    plt.close(fig)
    return fig


def plot_supervised(n_cells, frac_leukemic_AML1, frac_leukemic_AML2, frac_leukemic_AML3,
                    mean_healthy_x, std_healthy_x, mean_healthy_y, std_healthy_y,
                    dist_healthy_LAIP1, std_LAIP1_x, std_LAIP1_y,
                    dist_healthy_LAIP2, std_LAIP2_x, std_LAIP2_y):
    # To-do: refractor
    label_healthy = "Non-leukemic"
    label_diseased = "Leukemic"
    label_x = "LAIP1"
    label_y = "LAIP2"
    
    def plot_decision_boundary(ax, clf, xx, yy, data, title):
        # To-do: refractor
        label_healthy = "Non-leukemic"
        label_diseased = "Leukemic"
        label_x = "LAIP1"
        label_y = "LAIP2"

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#009292ff", "#ff7f00ff"])

        ax.contourf(xx, yy, Z, alpha=0.2, cmap=cmap)
        data["Prediction"] = clf.predict(data[["LAIP1", "LAIP2"]]) 
        sns.scatterplot(data=data, x=label_x, y=label_y, s=4, hue="Cluster", hue_order=[label_healthy, label_diseased],
        palette=["#009292ff", "#ff7f00ff"], ax=ax, rasterized=True, legend=None)
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_title(title)

    NBM, AML1, AML2, AML3 = simulate_samples(n_cells, frac_leukemic_AML1, frac_leukemic_AML2, frac_leukemic_AML3,
                                             mean_healthy_x, std_healthy_x, mean_healthy_y, std_healthy_y,
                                             dist_healthy_LAIP1, std_LAIP1_x, std_LAIP1_y,
                                             dist_healthy_LAIP2, std_LAIP2_x, std_LAIP2_y)

    xlims = (-3, 4)
    ylims = (-3, 4)

    fig = plt.figure(figsize=(5.25, 2))
    gs = gridspec.GridSpec(1, 3, figure=fig)
    
    # Train nearest-neighbor classifier on AML1
    df_X = AML1[[label_x, label_y]]
    clf = neighbors.KNeighborsClassifier(5, weights="uniform")
    clf.fit(df_X, np.where(AML1["Cluster"] == label_diseased, 1, 0))

    # Create a mesh to plot in
    xx, yy = np.meshgrid(np.arange(xlims[0], xlims[1], 0.1), np.arange(ylims[0], ylims[1], 0.1))
    
    # Plot decision boundary and data for AML1
    ax1 = fig.add_subplot(gs[0, 0])
    plot_decision_boundary(ax1, clf, xx, yy, AML1, "AML1 (training set)")

    # Plot decision boundary and data for AML2
    ax2 = fig.add_subplot(gs[0, 1])
    plot_decision_boundary(ax2, clf, xx, yy, AML2, "AML2")

    # Plot decision boundary and data for AML3
    ax3 = fig.add_subplot(gs[0, 2])
    plot_decision_boundary(ax3, clf, xx, yy, AML3, "AML3")

    plt.tight_layout()
    plt.close(fig)
    return fig


def cluster_with_normal(control_data, test_data, k=6):
    control = control_data.copy()
    test = test_data.copy()
    combined = pd.concat([control, test], ignore_index=True)

    # Fit the KMeans model
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(combined[["LAIP1", "LAIP2"]])
    
    # Add cluster labels to the combined dataset
    combined["KMeans"] = kmeans.labels_
    test_labels = kmeans.predict(test[["LAIP1", "LAIP2"]])
    test["KMeans"] = test_labels

    df = pd.DataFrame({"combined_count": combined["KMeans"].value_counts().sort_index(),
                   "test_count": test["KMeans"].value_counts().sort_index()})
    df["test_proportion"] = df["test_count"] / df["combined_count"]
    df["test_proportion"] = df["test_proportion"].fillna(0)
    mapping = df["test_proportion"].to_dict()
    test_enrichment = [mapping[label] for label in test_labels]
    return test_labels, test_enrichment


def plot_cluster_with_normal(n_cells, frac_leukemic_AML1, frac_leukemic_AML2, frac_leukemic_AML3,
                             mean_healthy_x, std_healthy_x, mean_healthy_y, std_healthy_y,
                             dist_healthy_LAIP1, std_LAIP1_x, std_LAIP1_y,
                             dist_healthy_LAIP2, std_LAIP2_x, std_LAIP2_y):
    # To-do: refractor
    label_healthy = "Non-leukemic"
    label_diseased = "Leukemic"
    label_x = "LAIP1"
    label_y = "LAIP2"
    
    def plot_scatter(data, title, ax):
        # To-do: refractor
        label_healthy = "Non-leukemic"
        label_diseased = "Leukemic"
        label_x = "LAIP1"
        label_y = "LAIP2"

        for cluster in data["KMeans"].unique():
            enrichment = list(data[data["KMeans"]==cluster]["Enrichment"])[0]
            if enrichment > 0.95:
                fill = "#ff7f00ff"
            else:
                fill = "#009292ff"

            points = np.array(data[data["KMeans"]==cluster][["LAIP1", "LAIP2"]])
            hull = ConvexHull(points)
            vert = np.append(hull.vertices, hull.vertices[0])
            ax.plot(points[vert, 0], points[vert, 1], '--', c="black", linewidth=1)
            ax.fill(points[vert, 0], points[vert, 1], c=fill, alpha=0.2)

        data["Prediction"] = np.where(data["Enrichment"] > 0.95, label_diseased, label_healthy)
        sns.scatterplot(data=data, x=label_x, y=label_y, s=4, hue="Cluster", hue_order=[label_healthy, label_diseased],
        palette=["#009292ff", "#ff7f00ff"], ax=ax, rasterized=True, legend=None)      
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_title(title)

    NBM, AML1, AML2, AML3 = simulate_samples(n_cells, frac_leukemic_AML1, frac_leukemic_AML2, frac_leukemic_AML3,
                                             mean_healthy_x, std_healthy_x, mean_healthy_y, std_healthy_y,
                                             dist_healthy_LAIP1, std_LAIP1_x, std_LAIP1_y,
                                             dist_healthy_LAIP2, std_LAIP2_x, std_LAIP2_y)

    xlims = (-3, 4)
    ylims = (-3, 4)

    # Get cluster-with-normal results
    AML1["KMeans"], AML1["Enrichment"] = cluster_with_normal(NBM, AML1)
    AML2["KMeans"], AML2["Enrichment"] = cluster_with_normal(NBM, AML2)
    AML3["KMeans"], AML3["Enrichment"] = cluster_with_normal(NBM, AML3)

    fig = plt.figure(figsize=(5.25, 2))
    gs = gridspec.GridSpec(1, 3, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    plot_scatter(AML1, "AML1", ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    plot_scatter(AML2, "AML2", ax2)

    ax3 = fig.add_subplot(gs[0, 2])
    plot_scatter(AML3, "AML3", ax3)

    plt.tight_layout()
    plt.close(fig)
    return fig


def novelty_detection(control_data, test_data, k=5):
    # Build a KDTree for dataset A
    kdtree = cKDTree(control_data[["LAIP1", "LAIP2"]])
    distances, indices = kdtree.query(test_data[["LAIP1", "LAIP2"]], k=5)
    median_distances = np.median(distances, axis=1)
    return median_distances


def plot_novelty_detection(n_cells, frac_leukemic_AML1, frac_leukemic_AML2, frac_leukemic_AML3,
                             mean_healthy_x, std_healthy_x, mean_healthy_y, std_healthy_y,
                             dist_healthy_LAIP1, std_LAIP1_x, std_LAIP1_y,
                             dist_healthy_LAIP2, std_LAIP2_x, std_LAIP2_y):
    # To-do: refractor
    label_healthy = "Non-leukemic"
    label_diseased = "Leukemic"
    label_x = "LAIP1"
    label_y = "LAIP2"

    def plot_decision_boundary(control_data, xlims, ylims, ax, k=5):
        # To-do: refractor
        label_healthy = "Non-leukemic"
        label_diseased = "Leukemic"
        label_x = "LAIP1"
        label_y = "LAIP2"

        # Create a mesh to plot in
        xx, yy = np.meshgrid(np.arange(xlims[0], xlims[1], 0.1), np.arange(ylims[0], ylims[1], 0.1))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        kdtree = cKDTree(control_data[["LAIP1", "LAIP2"]])
        distances, indices = kdtree.query(grid_points, k=k)
        dist = np.median(distances, axis=1)
        Z = np.where(dist > 0.5, 1, 0)
        Z = Z.reshape(xx.shape)
        cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#009292ff", "#ff7f00ff"])
        ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.2)
    
    def plot_scatter(data, title, ax):
        # To-do: refractor
        label_healthy = "Non-leukemic"
        label_diseased = "Leukemic"
        label_x = "LAIP1"
        label_y = "LAIP2"

        sns.scatterplot(data=data, x=label_x, y=label_y, s=4, hue="Cluster", hue_order=[label_healthy, label_diseased],
        palette=["#009292ff", "#ff7f00ff"], ax=ax, rasterized=True, legend=None)          
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_title(title)

    NBM, AML1, AML2, AML3 = simulate_samples(n_cells, frac_leukemic_AML1, frac_leukemic_AML2, frac_leukemic_AML3,
                                             mean_healthy_x, std_healthy_x, mean_healthy_y, std_healthy_y,
                                             dist_healthy_LAIP1, std_LAIP1_x, std_LAIP1_y,
                                             dist_healthy_LAIP2, std_LAIP2_x, std_LAIP2_y)

    xlims = (-3, 4)
    ylims = (-3, 4)

    # Get cluster-with-normal results
    AML1["dist"] = novelty_detection(NBM, AML1)
    AML2["dist"] = novelty_detection(NBM, AML2)
    AML3["dist"] = novelty_detection(NBM, AML3)

    AML1["Prediction"] = np.where(AML1["dist"] > 0.5, label_diseased, label_healthy)
    AML2["Prediction"] = np.where(AML2["dist"] > 0.5, label_diseased, label_healthy)
    AML3["Prediction"] = np.where(AML3["dist"] > 0.5, label_diseased, label_healthy)

    fig = plt.figure(figsize=(5.25, 2))
    gs = gridspec.GridSpec(1, 3, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    plot_decision_boundary(NBM, xlims, ylims, ax1, k=5)
    plot_scatter(AML1, "AML1", ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    plot_decision_boundary(NBM, xlims, ylims, ax2, k=5)
    plot_scatter(AML2, "AML2", ax2)

    ax3 = fig.add_subplot(gs[0, 2])
    plot_decision_boundary(NBM, xlims, ylims, ax3, k=5)
    plot_scatter(AML3, "AML3", ax3)

    plt.tight_layout()
    plt.close(fig)
    return fig
