import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve, auc
from scipy.spatial import cKDTree

from simulate import simulate_samples

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
        sns.scatterplot(data=data, x=label_x, y=label_y, hue="Cluster", hue_order=[label_healthy, label_diseased], ax=ax)
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_title(title)

    NBM, AML1, AML2, AML3 = simulate_samples(n_cells, frac_leukemic_AML1, frac_leukemic_AML2, frac_leukemic_AML3,
                                             mean_healthy_x, std_healthy_x, mean_healthy_y, std_healthy_y,
                                             dist_healthy_LAIP1, std_LAIP1_x, std_LAIP1_y,
                                             dist_healthy_LAIP2, std_LAIP2_x, std_LAIP2_y)

    xlims = (-6, 6)
    ylims = (-6, 6)
    
    fig = plt.figure(figsize=(20, 5))
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
    plt.show()
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
        ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
        sns.scatterplot(data=data, x=label_x, y=label_y, hue="Cluster", hue_order=[label_healthy, label_diseased], ax=ax)
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_title(title)

    def plot_roc_curve(ax, clf, data, label):
        # To-do: refractor
        label_healthy = "Non-leukemic"
        label_diseased = "Leukemic"
        label_x = "LAIP1"
        label_y = "LAIP2"

        y_score = clf.predict_proba(data[[label_x, label_y]])[:, 1]
        fpr, tpr, _ = roc_curve(np.where(data["Cluster"] == label_diseased, 1, 0), y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')

    NBM, AML1, AML2, AML3 = simulate_samples(n_cells, frac_leukemic_AML1, frac_leukemic_AML2, frac_leukemic_AML3,
                                             mean_healthy_x, std_healthy_x, mean_healthy_y, std_healthy_y,
                                             dist_healthy_LAIP1, std_LAIP1_x, std_LAIP1_y,
                                             dist_healthy_LAIP2, std_LAIP2_x, std_LAIP2_y)

    xlims = (-6, 6)
    ylims = (-6, 6)
    fig = plt.figure(figsize=(20, 5))
    gs = gridspec.GridSpec(1, 4, figure=fig)
    
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

    # Plot ROC curves
    ax4 = fig.add_subplot(gs[0, 3])
    plot_roc_curve(ax4, clf, AML1, "AML1")
    plot_roc_curve(ax4, clf, AML2, "AML2")
    plot_roc_curve(ax4, clf, AML3, "AML3")
    ax4.legend()
    ax4.set_title("ROC Curves")

    plt.tight_layout()
    plt.show()
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
        data["Prediction"] = np.where(data["Enrichment"] > 0.95, label_diseased, label_healthy)
        sns.scatterplot(data=data, x="LAIP1", y="LAIP2", hue="KMeans", style="Prediction", palette="tab10", legend=None, ax=ax)        
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_title(title)

    def plot_barplot(data, ax):
        table = data[["KMeans", "Enrichment"]].value_counts().reset_index().sort_values("KMeans")
        sns.barplot(data=table, x="KMeans", y="Enrichment", palette="tab10", ax=ax)
        ax.set_ylim(0, 1.05)

    def plot_roc_curve(ax, data, label):
        # To-do: refractor
        label_healthy = "Non-leukemic"
        label_diseased = "Leukemic"
        label_x = "LAIP1"
        label_y = "LAIP2"
        fpr, tpr, _ = roc_curve(np.where(data["Cluster"] == label_diseased, 1, 0), data["Enrichment"])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')

    NBM, AML1, AML2, AML3 = simulate_samples(n_cells, frac_leukemic_AML1, frac_leukemic_AML2, frac_leukemic_AML3,
                                             mean_healthy_x, std_healthy_x, mean_healthy_y, std_healthy_y,
                                             dist_healthy_LAIP1, std_LAIP1_x, std_LAIP1_y,
                                             dist_healthy_LAIP2, std_LAIP2_x, std_LAIP2_y)


    xlims = (-6, 6)
    ylims = (-6, 6)
    
    # Get cluster-with-normal results
    AML1["KMeans"], AML1["Enrichment"] = cluster_with_normal(NBM, AML1)
    AML2["KMeans"], AML2["Enrichment"] = cluster_with_normal(NBM, AML2)
    AML3["KMeans"], AML3["Enrichment"] = cluster_with_normal(NBM, AML3)

    fig = plt.figure(figsize=(20, 5))
    gs = gridspec.GridSpec(2, 4, figure=fig, height_ratios=[0.7, 0.3])

    ax1a = fig.add_subplot(gs[0, 0])
    plot_scatter(AML1, "AML1", ax1a)
    ax1b = fig.add_subplot(gs[1, 0])
    plot_barplot(AML1, ax1b)

    ax2a = fig.add_subplot(gs[0, 1])
    plot_scatter(AML2, "AML2", ax2a)
    ax2b = fig.add_subplot(gs[1, 1])
    plot_barplot(AML2, ax2b)

    ax3a = fig.add_subplot(gs[0, 2])
    plot_scatter(AML3, "AML3", ax3a)
    ax3b = fig.add_subplot(gs[1, 2])
    plot_barplot(AML3, ax3b)

    # Plot ROC curves
    ax4 = fig.add_subplot(gs[:2, 3])
    plot_roc_curve(ax4, AML1, "AML1")
    plot_roc_curve(ax4, AML2, "AML2")
    plot_roc_curve(ax4, AML3, "AML3")
    ax4.legend()
    ax4.set_title("ROC Curves")

    plt.tight_layout()
    plt.show()
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
    
    def plot_scatter(data, title, ax):
        # To-do: refractor
        label_healthy = "Non-leukemic"
        label_diseased = "Leukemic"
        label_x = "LAIP1"
        label_y = "LAIP2"
        data["Prediction"] = np.where(data["dist"] > 0.1, label_diseased, label_healthy)
        sns.scatterplot(data=data, x="LAIP1", y="LAIP2", hue="Prediction", palette="tab10", legend=None, ax=ax)        
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_title(title)

    def plot_roc_curve(ax, data, label):
        # To-do: refractor
        label_healthy = "Non-leukemic"
        label_diseased = "Leukemic"
        label_x = "LAIP1"
        label_y = "LAIP2"
        fpr, tpr, _ = roc_curve(np.where(data["Cluster"] == label_diseased, 1, 0), data["Enrichment"])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')

    NBM, AML1, AML2, AML3 = simulate_samples(n_cells, frac_leukemic_AML1, frac_leukemic_AML2, frac_leukemic_AML3,
                                             mean_healthy_x, std_healthy_x, mean_healthy_y, std_healthy_y,
                                             dist_healthy_LAIP1, std_LAIP1_x, std_LAIP1_y,
                                             dist_healthy_LAIP2, std_LAIP2_x, std_LAIP2_y)


    xlims = (-6, 6)
    ylims = (-6, 6)
    
    # Get cluster-with-normal results
    AML1["dist"] = novelty_detection(NBM, AML1)
    AML2["dist"] = novelty_detection(NBM, AML2)
    AML3["dist"] = novelty_detection(NBM, AML3)

    fig = plt.figure(figsize=(20, 5))
    gs = gridspec.GridSpec(1, 4, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    plot_scatter(AML1, "AML1", ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    plot_scatter(AML2, "AML2", ax2)

    ax3 = fig.add_subplot(gs[0, 2])
    plot_scatter(AML3, "AML3", ax3)

    # Plot ROC curves
    ax4 = fig.add_subplot(gs[:2, 3])
    # plot_roc_curve(ax4, AML1, "AML1")
    # plot_roc_curve(ax4, AML2, "AML2")
    # plot_roc_curve(ax4, AML3, "AML3")
    ax4.legend()
    # ax4.set_title("ROC Curves")

    plt.tight_layout()
    plt.show()
    return fig
