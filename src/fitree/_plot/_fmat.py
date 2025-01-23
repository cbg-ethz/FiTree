import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def plot_fmat(
    F_mat: np.ndarray,
    mutation_labels: list | None = None,
    to_sort: bool = True,
    figsize: tuple = (8, 6),
) -> None:
    if mutation_labels is None:
        mutation_labels = [f"M{i}" for i in range(F_mat.shape[1])]

    F_mat = F_mat + F_mat.T - np.diag(np.diag(F_mat))
    if to_sort:
        diagonal_values = np.diag(F_mat)
        sorted_indices = np.argsort(-diagonal_values)
        F_mat = F_mat[np.ix_(sorted_indices, sorted_indices)]
        mutation_labels = [mutation_labels[i] for i in sorted_indices]

    F_mat = np.transpose(F_mat)

    mask = np.triu(np.ones_like(F_mat, dtype=bool), k=1)

    cmap = sns.diverging_palette(145, 300, s=60, as_cmap=True)

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

    # # Determine global x-axis limits based on F_mat_posterior values
    # global_min = np.min(F_mat_posterior)
    # global_max = np.max(F_mat_posterior)
    # x_lim = max(abs(global_min), abs(global_max))  # Symmetric range for centering

    fig, axes = plt.subplots(n_mutations, n_mutations, figsize=figsize)

    for i, j in zip(*tril_indices):
        ax = axes[i, j]
        sns.histplot(
            F_mat_posterior[:, i, j], ax=ax, kde=True, bins="sturges", alpha=0.2
        )

        if true_F_mat is not None:
            ax.axvline(true_F_mat[i, j], color="darkred", linestyle="--", linewidth=3)

        ax.axvline(0, color="grey", linestyle="--", linewidth=1)

        # Remove y-axis labels and titles for all subplots
        ax.set_ylabel("")
        ax.set_xlabel("")

        # Set x-axis limits
        local_min = np.min(F_mat_posterior[:, i, j])
        local_max = np.max(F_mat_posterior[:, i, j])
        if true_F_mat is not None:
            local_min = min(local_min, true_F_mat[i, j])  # pyright: ignore
            local_max = max(local_max, true_F_mat[i, j])  # pyright: ignore

        local_x_lim = max(abs(local_min), abs(local_max)) * 1.1
        ax.set_xlim(-local_x_lim, local_x_lim)

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


def plot_epistasis(
    F_mat: np.ndarray,
    mutation_labels: list | None = None,
    to_sort: bool = True,
    figsize: tuple = (8, 6),
) -> None:
    if mutation_labels is None:
        mutation_labels = [f"M{i}" for i in range(F_mat.shape[1])]

    # Sort the rows and columns of epistasis based on diagonal elements
    F_mat = F_mat + F_mat.T - np.diag(np.diag(F_mat))
    if to_sort:
        diagonal_values = np.diag(F_mat)
        sorted_indices = np.argsort(-diagonal_values)
        F_mat = F_mat[np.ix_(sorted_indices, sorted_indices)]
        mutation_labels = [mutation_labels[i] for i in sorted_indices]

    base_effects = np.diag(F_mat).reshape(-1, 1)

    n_mutations = F_mat.shape[1]
    epistasis = F_mat.copy()
    for i in range(n_mutations):
        for j in range(i, n_mutations):
            epistasis[i, j] += np.max([F_mat[i, i], F_mat[j, j]])

    epistasis = np.round(np.transpose(epistasis), 2)

    mask = np.triu(np.ones_like(epistasis, dtype=bool), k=0)

    upp_tri_indices = np.triu_indices(n_mutations, k=0)
    all_effects = F_mat[upp_tri_indices]

    # compute vmax, vmin, and center for the colorbar
    vmax = float(all_effects.max())
    vmin = float(all_effects.min())
    vmin = min(vmin, 0)
    center = float(np.mean([vmax, vmin]))

    if vmin < 0:
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
    else:
        cmap = sns.color_palette("flare", as_cmap=True)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        gridspec_kw={"width_ratios": [2, n_mutations], "wspace": 0.5},
    )
    sns.set_theme(style="white")

    sns.heatmap(
        base_effects,
        cmap=cmap,
        annot=True,
        cbar=False,
        xticklabels=[""],
        yticklabels=mutation_labels,
        linewidths=0.5,
        vmax=vmax,
        vmin=vmin,
        center=center,
        ax=axes[0],
    )
    axes[0].set_title("Base Effects", fontsize=14)

    sns.heatmap(
        epistasis,
        mask=mask,
        cmap=cmap,
        annot=True,
        xticklabels=mutation_labels[:-1] + [""],
        yticklabels=[""] + mutation_labels[1:],
        linewidths=0.5,
        vmax=vmax,
        vmin=vmin,
        center=center,
        ax=axes[1],
    )
    axes[1].set_title("Epistasis", fontsize=14)


def plot_fmat_std(
    F_mat_sample: np.ndarray,
    mutation_labels: list | None = None,
    to_sort: bool = True,
    figsize: tuple = (8, 6),
):
    if mutation_labels is None:
        mutation_labels = [f"M{i}" for i in range(F_mat_sample.shape[1])]

    F_mat_std = F_mat_sample.std(axis=0)
    F_mat_std = F_mat_std + F_mat_std.T - np.diag(np.diag(F_mat_std))
    if to_sort:
        diagonal_values = np.diag(F_mat_std)
        sorted_indices = np.argsort(-diagonal_values)[::-1]
        F_mat_std = F_mat_std[np.ix_(sorted_indices, sorted_indices)]
        mutation_labels = [mutation_labels[i] for i in sorted_indices]

    F_mat_std = np.transpose(F_mat_std)

    plot_fmat(F_mat_std, mutation_labels, False, figsize)
