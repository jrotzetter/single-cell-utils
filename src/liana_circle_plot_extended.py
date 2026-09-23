# Portions of this file are derived from LIANA+ (BSD-3-Clause)
# Copyright (c) 2025, Daniel Dimitrov
# https://github.com/scverse/liana

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from liana._constants import Keys as K
from liana.plotting._common import (
    _filter_by,
    _get_top_n,
    _invert_scores,
    _prep_liana_res,
)
from liana.plotting._circle_plot import (
    _pivot_liana_res,
    _scale_list,
    _get_adata_colors,
    get_mask_df,
)


def circle_plot_extended(
    adata: sc.AnnData,
    uns_key: str | None = K.uns_key,
    liana_res: pd.DataFrame | None = None,
    groupby: str = None,
    colorby: str = None,
    source_key: str = "source",
    target_key: str = "target",
    score_key: str = None,
    inverse_score: bool = False,
    top_n: int = None,
    orderby: str | None = None,
    orderby_ascending: bool | None = None,
    orderby_absolute: bool = False,
    filter_fun: Callable = None,
    source_labels: list[str] | str | None = None,
    target_labels: list[str] | str | None = None,
    ligand_complex: list[str] | str | None = None,
    receptor_complex: list[str] | str | None = None,
    pivot_mode: Literal["counts", "mean"] = "counts",
    mask_mode: Literal["and", "or"] = "or",
    figure_size: tuple[float, float] = (5, 5),
    edge_alpha: float = 0.5,
    edge_arrow_size: int = 10,
    edge_width_scale: tuple[float, float] = (1, 5),
    node_alpha: float = 1,
    node_size_scale: tuple[float, float] = (100, 400),
    node_label_offset: tuple[float, float] = (0.1, -0.2),
    node_label_size: int = 8,
    node_label_alpha: float = 0.7,
) -> Axes:
    """
    Visualize the cell-cell communication network using a circular plot.

    Parameters
    ----------
    %(adata)s
    %(uns_key)s
    %(liana_res)s
    %(groupby)s
    colorby
        Key in `adata.obs` to color nodes by (e.g., 'region', 'tissue').
        Requires a 1-to-1 mapping with `groupby`; if a `groupby` category maps
        to multiple `colorby` values, a `ValueError` is raised. Automatically
        adds a legend.
    %(source_key)s
    %(target_key)s
    %(score_key)s
    inverse_score
        Whether to invert the score, by default False. If True, the score will be -log10(score).
    %(top_n)s
    %(orderby)s
    %(orderby_ascending)s
    %(orderby_absolute)s
    %(filter_fun)s
    %(source_labels)s
    %(target_labels)s
    %(ligand_complex)s
    %(receptor_complex)s
    pivot_mode
        The mode of the pivot table, by default 'counts'.
        - 'counts': The number of connections between source and target.
        - 'mean': The mean of the values of `score_key` between source and target cell types (groupby).
        Note that `filter_fun` differs by pivot_mode: when counts it would remove all interactions
        that don't pass the filter, while for 'mean' it would retain interactions don't pass the filter
        if the same interaction passes it for any cell type pair.
    mask_mode
        The mode of the mask, by default 'or'.
        - 'or': Include the source or target cell type.
        - 'and': Include the source and target cell type.
    %(figure_size)s
    edge_alpha
        The transparency of the edges, by default .5.
    edge_arrow_size
        The size of the arrow, by default 10.
    edge_width_scale
        The scale of the edge width, by default (1, 5).
    node_alpha
        The transparency of the nodes, by default 1.
    node_size_scale
        The scale of the node size, by default (100, 400).
    node_label_offset
        The offset of the node label, by default (0.1, -0.2).
    node_label_size
        The size of the node label, by default 8.
    node_label_alpha
        The transparency of the node label, by default .7.

    Returns
    -------
    The figure axes containing the circle plot.

    Raises
    ------
    ValueError
        If `groupby` is not provided

    """
    if groupby is None:
        raise ValueError("`groupby` must be provided!")

    # Check if any category of 'groupby' maps to multiple categories of 'colorby' (which would cause ambiguity)
    if colorby is not None:
        if colorby not in adata.obs.columns:
            raise ValueError(f"`colorby='{colorby}'` not found in adata.obs")

        # Check for 1-to-1 mapping consistency
        mapping_check = adata.obs.groupby(groupby, observed=False)[colorby].nunique()
        if (mapping_check > 1).any():
            ambiguous_types = mapping_check[mapping_check > 1].index.tolist()
            raise ValueError(
                f"Ambiguous mapping: The following {groupby} entries map to multiple "
                f"{colorby} categories: {ambiguous_types}. "
                f"Ensure each {groupby} category corresponds to exactly one {colorby} category."
            )

    liana_res = _prep_liana_res(
        adata=adata,
        source_labels=None,
        target_labels=None,
        ligand_complex=ligand_complex,
        receptor_complex=receptor_complex,
        uns_key=uns_key,
    )

    if pivot_mode == "counts":
        if filter_fun is not None:
            mask = liana_res.apply(filter_fun, axis=1).astype(bool)
            liana_res = liana_res[mask]
    elif pivot_mode == "mean":
        liana_res = _filter_by(liana_res, filter_fun)
    else:
        raise ValueError("`pivot_mode` must be 'counts' or 'mean'!")
    liana_res = _get_top_n(
        liana_res, top_n, orderby, orderby_ascending, orderby_absolute
    )

    if inverse_score:
        liana_res[score_key] = _invert_scores(liana_res[score_key])

    pivot_table = _pivot_liana_res(
        liana_res,
        source_key=source_key,
        target_key=target_key,
        score_key=score_key,
        mode=pivot_mode,
    )

    # Get unique values for both columns to build a mapping
    if colorby is not None:
        # Ensure both columns exist in adata.obs
        if groupby not in adata.obs.columns or colorby not in adata.obs.columns:
            raise ValueError(
                f"Both '{groupby}' and '{colorby}' must be columns in adata.obs"
            )

        # Create a mapping: groupby_label -> colorby_label
        # Assumes a 1-to-1 mapping (each groupby category belongs to one colorby category).
        # If a groupby label maps to multiple colorby labels, this takes the first unique value.
        mapping_series = adata.obs.groupby(groupby, observed=False)[colorby].first()

        # Get colors for the unique colorby categories
        unique_colorby_cats = mapping_series.unique()
        colorby_colors = _get_adata_colors(adata, label=colorby)

        # Build the final lookup: groupby_label -> color
        # This maps, e.g. "Neuroblasts" -> "Brain" -> <Color Hex>
        node_color_map = {
            gb_label: colorby_colors.get(
                cb_label, "#888888"
            )  # Fallback if color missing
            for gb_label, cb_label in mapping_series.items()
        }
    else:
        # Original behavior: color by groupby
        node_color_map = None
        colorby_colors = _get_adata_colors(adata, label=groupby)

    # Mask pivot table
    _pivot_table = get_mask_df(
        pivot_table,
        source_cell_type=source_labels,
        target_cell_type=target_labels,
        mode=mask_mode,
    )

    G = nx.convert_matrix.from_pandas_adjacency(_pivot_table, create_using=nx.DiGraph())
    pos = nx.circular_layout(G)

    # Assign Colors
    if colorby is not None:
        # Map each node in G to its color using the pre-built map
        groupby_colors = {node: node_color_map.get(node, "#cccccc") for node in G.nodes}
    else:
        # Original behavior
        groupby_colors = colorby_colors

    # Extract edge/node properties (Unchanged to base circle_plot)
    edge_color = [groupby_colors[cell[0]] for cell in G.edges]
    edge_width = np.asarray([G.edges[e]["weight"] for e in G.edges()])
    edge_width = _scale_list(
        edge_width, max_val=edge_width_scale[1], min_val=edge_width_scale[0]
    )

    node_color = [groupby_colors[cell] for cell in G.nodes]
    node_size = pivot_table.sum(axis=1).values
    node_size = _scale_list(
        node_size, max_val=node_size_scale[1], min_val=node_size_scale[0]
    )

    fig, ax = plt.subplots(figsize=figure_size)

    # Visualize network (Unchanged to base circle_plot)
    nx.draw_networkx_edges(
        G,
        pos,
        alpha=edge_alpha,
        arrowsize=edge_arrow_size,
        arrowstyle="-|>",
        width=edge_width,
        edge_color=edge_color,
        connectionstyle="arc3,rad=-0.3",
        ax=ax,
    )

    nx.draw_networkx_nodes(
        G, pos, node_color=node_color, node_size=node_size, alpha=node_alpha, ax=ax
    )
    label_options = {"ec": "k", "fc": "white", "alpha": node_label_alpha}
    _ = nx.draw_networkx_labels(
        G,
        {k: v + np.array(node_label_offset) for k, v in pos.items()},
        font_size=node_label_size,
        bbox=label_options,
        ax=ax,
    )

    # Draw connector lines between nodes and labels
    if colorby is not None or node_label_offset != (0, 0):
        for node in G.nodes():
            node_pos = pos[node]
            # Label position (same offset used in draw_networkx_labels)
            label_pos = node_pos + np.array(node_label_offset)

            # To draw connector lines with node-matching colors
            # node_color_val = groupby_colors[node]

            # Draw a faint line connecting them
            ax.plot(
                [node_pos[0], label_pos[0]],
                [node_pos[1], label_pos[1]],
                color="gray",
                # color=node_color_val, # Uncomment to draw connector lines with node-matching colors
                alpha=0.8,
                linewidth=0.8,
                linestyle="--",
                zorder=0,  # Behind nodes and labels
            )

    ax.set_frame_on(False)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    coeff = 1.2
    ax.set_xlim((xlim[0] * coeff, xlim[1] * coeff))
    ax.set_ylim((ylim[0] * coeff, ylim[1]))
    ax.set_aspect("equal")

    # Add legend for colorby categories only
    if colorby is not None:
        # Create legend handles for each unique colorby category
        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                label=category,
                markersize=10,
                linestyle="None",
            )
            for category, color in colorby_colors.items()
        ]

        ax.legend(
            handles=legend_handles,
            title=colorby,
            loc="upper right",
            bbox_to_anchor=(1.05, 0.98),
            frameon=True,
            fontsize=8,
            title_fontsize=9,
        )

    return ax
