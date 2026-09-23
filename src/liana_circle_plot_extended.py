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
    add_node_label: bool = True,
    cluster_nodes: bool = True,
    alternate_labels: Literal["selective", True, False] = True,
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
    add_node_label
        Whether to display the node_labels on the plot.
    cluster_nodes
        Whether to sort nodes by `colorby` categories to group them visually.
        By default True. If False, nodes are plotted in the default graph order.
        Only has an effect if `colorby` is provided.
    alternate_labels
        Whether to alternate label positions inward and outward radially.
        By default True. If False, all labels are placed radially outward.
        Only has an effect if `add_node_label` is True.

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

    ### --- Colorby Grouping --- ###
    # Determine Node Order
    sorted_nodes = list(G.nodes())

    # Only sort nodes by category to group them if cluster_nodes is True AND colorby is provided
    if cluster_nodes and colorby is not None:
        node_to_colorby = adata.obs.groupby(groupby, observed=False)[colorby].first()
        sorted_nodes = sorted(
            G.nodes(), key=lambda node: (node_to_colorby.get(node, ""), node)
        )

    # Rotate list so the first target node is at index 0
    # (This ensures it lands at 3 o'clock initially, which will then be rotated to 12 o'clock)
    if target_labels is not None:
        targets = [target_labels] if isinstance(target_labels, str) else target_labels
        for i, node in enumerate(sorted_nodes):
            if node in targets:
                sorted_nodes = sorted_nodes[i:] + sorted_nodes[:i]
                break

    # Rebuild graph with the determined order
    G_ordered = nx.DiGraph()
    G_ordered.add_nodes_from(sorted_nodes)
    G_ordered.add_edges_from(G.edges(data=True))
    G = G_ordered

    # Generate circular layout (starts at 3 o'clock / 0 radians)
    pos = nx.circular_layout(G)

    # Rotate all positions by 90 degrees (pi/2) to move index 0 to the TOP
    rotation_angle = np.pi / 2  # 90 degrees counter-clockwise

    # for node in pos:
    #     x, y = pos[node]
    #     # Apply rotation matrix
    #     new_x = x * np.cos(rotation_angle) - y * np.sin(rotation_angle)
    #     new_y = x * np.sin(rotation_angle) + y * np.cos(rotation_angle)
    #     pos[node] = np.array([new_x, new_y])

    # Vectorized rotation matrix using NumPy broadcasting
    rotation_matrix = np.array(
        [
            [np.cos(rotation_angle), -np.sin(rotation_angle)],
            [np.sin(rotation_angle), np.cos(rotation_angle)],
        ]
    )
    pos = {node: rotation_matrix @ coord for node, coord in pos.items()}

    ### --- Colorby Grouping End --- ###

    # Assign Colors
    if colorby is not None:
        # Map each node in G to its color using the pre-built map
        groupby_colors = {node: node_color_map.get(node, "#cccccc") for node in G.nodes}
    else:
        # Original behavior
        groupby_colors = colorby_colors

    # Extract edge/node properties
    edge_color = [groupby_colors[cell[0]] for cell in G.edges]
    edge_width = np.asarray([G.edges[e]["weight"] for e in G.edges()])
    edge_width = _scale_list(
        edge_width, max_val=edge_width_scale[1], min_val=edge_width_scale[0]
    )

    node_color = [groupby_colors[cell] for cell in G.nodes]
    # Calculate sizes aligned to the sorted node order
    node_size = [
        pivot_table.loc[node, :].sum() if node in pivot_table.index else 0
        for node in G.nodes()
    ]
    node_size = _scale_list(
        node_size, max_val=node_size_scale[1], min_val=node_size_scale[0]
    )
    fig, ax = plt.subplots(figsize=figure_size)

    # Visualize network (Unchanged)
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
    if add_node_label:
        # Generate the radial label positions
        offset_mag = np.sqrt(node_label_offset[0] ** 2 + node_label_offset[1] ** 2)
        if offset_mag == 0:
            offset_mag = 0.15  # Default fallback if offset is (0,0)

        # Select offset function based on parameter
        if alternate_labels == "selective":
            label_pos = get_selective_radial_offset(pos, base_offset=offset_mag)
        elif alternate_labels:
            # Calculate alternating inward/outward radial offset
            label_pos = get_alternating_radial_offset(pos, base_offset=offset_mag)
        else:
            # Use simple outward offset for all nodes
            label_pos = get_radial_offset(pos, offset_magnitude=offset_mag)

        # Draw Labels at the calculated radial positions
        label_options = {"ec": "k", "fc": "white", "alpha": node_label_alpha}
        _ = nx.draw_networkx_labels(
            G, label_pos, font_size=node_label_size, bbox=label_options, ax=ax
        )

        # Draw Connectors
        for node in G.nodes():
            node_pos = pos[node]
            label_xy = label_pos[node]

            ax.plot(
                [node_pos[0], label_xy[0]],
                [node_pos[1], label_xy[1]],
                color="gray",
                alpha=0.8,
                linewidth=0.8,
                linestyle="--",
                zorder=0,
            )

    ax.set_frame_on(False)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    coeff = 1.2
    ax.set_xlim((xlim[0] * coeff, xlim[1] * coeff))
    ax.set_ylim((ylim[0] * coeff, ylim[1] * coeff))
    ax.set_aspect("equal")

    # Add legend for colorby categories only
    if colorby is not None:
        # # Create legend handles for each unique colorby category
        # legend_handles = [
        #     Line2D([0], [0], marker='o', color='w',
        #            markerfacecolor=color, label=category,
        #            markersize=10, linestyle='None')
        #     for category, color in colorby_colors.items()
        # ]

        # Only show categories that appear in the plot
        present_colors = {
            node_color_map[node] for node in G.nodes() if node in node_color_map
        }
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
            if color in present_colors
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


def get_radial_offset(positions, offset_magnitude=0.15):
    label_positions = {}
    for node, (x, y) in positions.items():
        # Calculate angle of the node from the center (0,0)
        angle = np.arctan2(y, x)
        # Calculate new position pushed outward along that angle
        label_positions[node] = np.array(
            [x + offset_magnitude * np.cos(angle), y + offset_magnitude * np.sin(angle)]
        )
    return label_positions


def get_alternating_radial_offset(positions, base_offset=0.15):
    label_positions = {}
    # Sort nodes by angle to ensure consistent alternating pattern around the circle
    sorted_nodes = sorted(
        positions.keys(), key=lambda n: np.arctan2(positions[n][1], positions[n][0])
    )

    for i, node in enumerate(sorted_nodes):
        x, y = positions[node]
        angle = np.arctan2(y, x)

        # Alternate direction: even indices OUTWARD (+), odd indices INWARD (-)
        # Multiply by 1.2 for outward nodes to give them slightly more breathing room
        direction = 1 if i % 2 == 0 else -1
        magnitude = base_offset * (1.2 if direction > 0 else 1.0)

        label_positions[node] = np.array(
            [
                x + direction * magnitude * np.cos(angle),
                y + direction * magnitude * np.sin(angle),
            ]
        )
    return label_positions


def get_selective_radial_offset(positions, base_offset=0.15, stagger_angle_range=45.0):
    """
    Apply alternating inward/outward offsets only to nodes at the top and bottom.
    Nodes on the left and right sides are pushed strictly outward.

    Parameters
    ----------
    positions : dict
        Node positions {node: [x, y]}.
    base_offset : float
        Base magnitude of the offset.
    stagger_angle_range : float
        Degrees from the vertical axis (90 and 270) within which to apply staggering.
        Default 45.0 means staggering occurs between 45-135° (top) and 225-315° (bottom).
    """
    label_positions = {}

    # Sort nodes by angle to ensure consistent alternating pattern within the staggered zones
    sorted_nodes = sorted(
        positions.keys(), key=lambda n: np.arctan2(positions[n][1], positions[n][0])
    )

    # Counter for alternating logic (only increments when hit a staggered node)
    stagger_index = 0

    for node in sorted_nodes:
        x, y = positions[node]
        # Calculate angle in degrees [0, 360)
        angle_rad = np.arctan2(y, x)
        angle_deg = np.degrees(angle_rad) % 360

        # Determine if node is in Top or Bottom zone
        # Top: 90 +/- range, Bottom: 270 +/- range
        in_top_zone = (
            (90 - stagger_angle_range) <= angle_deg <= (90 + stagger_angle_range)
        )
        in_bottom_zone = (
            (270 - stagger_angle_range) <= angle_deg <= (270 + stagger_angle_range)
        )

        is_staggered_zone = in_top_zone or in_bottom_zone

        if is_staggered_zone:
            # Apply alternating logic
            direction = 1 if stagger_index % 2 == 0 else -1
            magnitude = base_offset * (1.2 if direction > 0 else 1.0)
            stagger_index += 1  # Only increment counter for staggered nodes
        else:
            # Force outward for side labels (Left/Right)
            direction = 1
            magnitude = base_offset

        label_positions[node] = np.array(
            [
                x + direction * magnitude * np.cos(angle_rad),
                y + direction * magnitude * np.sin(angle_rad),
            ]
        )

    return label_positions
