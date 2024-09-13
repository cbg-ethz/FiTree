import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def plot_fmat(
    F_mat: np.ndarray, mutation_labels: dict = None, figsize: tuple = (8, 6)
) -> None:
    mutation_label_list = [mutation_labels[i] for i in range(F_mat.shape[1])]

    F_mat = np.transpose(np.log(F_mat))

    mask = np.triu(np.ones_like(F_mat, dtype=bool), k=1)

    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.set_theme(style="white")

    plt.figure(figsize=figsize)

    sns.heatmap(
        F_mat,
        mask=mask,
        cmap=cmap,
        center=0,
        xticklabels=mutation_label_list,
        yticklabels=mutation_label_list,
        annot=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        vmax=0.3,
    )

    plt.title("Fitness matrix in log scale", fontsize=16)
