import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def plot_fmat(
    F_mat: np.ndarray, mutation_labels: list | None = None, figsize: tuple = (8, 6)
) -> None:
    if mutation_labels is None:
        mutation_labels = [f"M{i}" for i in range(F_mat.shape[1])]

    F_mat = np.transpose(F_mat)

    mask = np.triu(np.ones_like(F_mat, dtype=bool), k=1)

    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.set_theme(style="white")

    plt.figure(figsize=figsize)

    sns.heatmap(
        F_mat,
        mask=mask,
        cmap=cmap,
        center=0,
        xticklabels=mutation_labels,
        yticklabels=mutation_labels,
        annot=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        vmax=0.3,
    )

    plt.title("Fitness matrix F", fontsize=16)


def plot_fmat_posterior(
    F_mat_posterior: np.ndarray,
    true_F_mat: np.ndarray | None = None,
    mutation_labels: list | None = None,
    figsize: tuple = (8, 7),
) -> None:
    n_mutations = F_mat_posterior.shape[1]

    F_mat_posterior = F_mat_posterior.transpose(0, 2, 1)
    if true_F_mat is not None:
        true_F_mat = true_F_mat.transpose()

    if mutation_labels is None:
        mutation_labels = [f"M{i}" for i in range(n_mutations)]

    tril_indices = np.tril_indices(n_mutations, k=0)

    fig, axes = plt.subplots(n_mutations, n_mutations, figsize=figsize)

    for i, j in zip(*tril_indices):
        ax = axes[i, j]
        sns.histplot(F_mat_posterior[:, i, j], ax=ax, kde=True)

        if true_F_mat is not None:
            ax.axvline(true_F_mat[i, j], color="darkgreen", linestyle="--")

        # Remove y-axis labels and titles for all subplots
        ax.set_ylabel("")
        ax.set_xlabel("")

    # # Hide the upper triangular subplots
    for i in range(n_mutations):
        for j in range(i + 1, n_mutations):
            axes[i, j].axis("off")

    # Add mutation labels to the left (row labels) and bottom (column labels)
    for i in range(n_mutations):
        axes[i, 0].set_ylabel(
            mutation_labels[i], rotation=0, labelpad=20, va="center", fontsize=16
        )
        axes[-1, i].set_xlabel(mutation_labels[i], fontsize=16)

    plt.tight_layout()
    plt.suptitle("Posterior of fitness matrix F", fontsize=20)
    plt.subplots_adjust(top=0.9)
