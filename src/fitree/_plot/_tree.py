import graphviz
from anytree import PreOrderIter
import numpy as np
from IPython.display import display

from fitree import TumorTreeCohort

COLOR_PALETTE = [
    "lightcoral",
    "skyblue3",
    "sandybrown",
    "paleturquoise3",
    "thistle",
    "darkolivegreen3",
    "lightpink",
    "mediumpurple",
    "darkorange",
    "lightblue",
    "darkseagreen",
    "lightsteelblue",
    "darkkhaki",
    "lightgoldenrodyellow",
    "darkslategray",
    "lightcyan",
    "darkslateblue",
]


def plot_tree(
    cohort: TumorTreeCohort, tree_id: int, filename: str | None = None
) -> None:
    """Plot a tree in the cohort"""
    tree = cohort.trees[tree_id]
    tree_label = cohort.tree_labels[tree_id]
    tumor_size = tree.tumor_size

    # Define the directed graph
    graph = graphviz.Digraph(
        name=tree_label,
        format="pdf",
    )

    # Set global graph attributes
    graph.attr(
        nodesep="1.5",
        labelloc="t",
        fontname="Helvetica",
        fontsize="28",
        label=f"Tree: {tree_label}",
        size="10,10",
        ratio="fill",
    )

    # Set default node attributes
    graph.attr(
        "node",
        color="dimgray",
        fontname="Helvetica",
        fontsize="24",
        fontcolor="dimgray",
        penwidth="5",
    )

    for node in PreOrderIter(tree.root):
        if node.is_root:
            graph.node(
                name=str(node.node_id),
                label="Root",
            )
        else:
            mutations = ", ".join(cohort.mutation_labels[i] for i in node.mutation_ids)
            graph.node(
                name=str(node.node_id),
                label=mutations,
            )
            graph.edge(
                tail_name=str(node.parent.node_id),
                head_name=str(node.node_id),
                color="dimgray",
                weight="4",
                penwidth="5",
            )

            node_width = np.sqrt(node.cell_number) / np.sqrt(tumor_size) * 4
            cell_number = node.cell_number
            cell_percentage = node.cell_number / tumor_size
            color = COLOR_PALETTE[node.node_id % len(COLOR_PALETTE)]
            graph.node(
                name=f"s_{node.node_id}",
                label=f"{cell_number:.4E}\n({cell_percentage:.2%})",
                width=str(node_width),
                style="filled",
                color=color,
                shape="circle",
                fixedsize="true",
                fontsize="16",
                fontcolor="black",
            )
            graph.edge(
                tail_name=str(node.node_id),
                head_name=f"s_{node.node_id}",
                arrowhead="none",
                style="dashed",
                weight="4",
                penwidth="5",
                color=color,
            )

    if tree.sampling_time:
        graph.node(
            name="sampling_time",
            label=f"Sampling time: {tree.sampling_time}",
            shape="box",
            fontsize="24",
            fontname="Helvetica",
        )

        with graph.subgraph() as s:  # pyright: ignore
            s.attr(rank="sink")
            s.node("sampling_time")

    if filename:
        graph.render(filename, format="pdf", cleanup=True)
    else:
        display(graphviz.Source(graph.source))
