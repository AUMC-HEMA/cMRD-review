import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn import neighbors
from sklearn.metrics import roc_curve, auc

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


















