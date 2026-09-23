# single-cell-utils

Utility functions for single-cell transcriptomics analysis, developed during a bioinformatics internship at the Service of Endocrinology, Diabetology and Metabolism, CHUV (Lausanne University Hospital), Switzerland.

## Functions

| Function | Description |
|---|---|
| `find_leiden_resolution` | Iteratively adjusts Leiden resolution until the cluster count falls within a target range. |
| `find_all_markers` | Fast, Seurat/Scanpy-compatible marker detection using [illico](https://github.com/remydubois/illico), with optional pre-filtering and CSV export. |
| `circle_plot_extended` | Extended [LIANA+](https://github.com/scverse/liana) `circle_plot` with additional styling (node-label connector lines, additional key to color nodes by), node clustering, and label-placement options. |

Please see the source files for full signatures and docstrings.

## Dependencies

Developed against:

- Python >= 3.10
- [Scanpy](https://scanpy.org) 1.12.x
- [illico](https://github.com/remydubois/illico) 0.6.0
- [LIANA+](https://github.com/scverse/liana) 1.7.3

*(Versions reflect the tested environment; newer minor versions may work but are unverified.)*

## Extended Circle Plot Examples

> Note: All examples use the same data and `groupby`/`colorby` keys unless otherwise noted.

### Vanilla vs. Extended
<table>
<tr>
<td align="center"><img src="examples/circle_plot_vanilla.png" width="400" style="max-width:100%" title="Vanilla LIANA circle_plot (liana==1.7.3)"><br><code>vanilla `circle_plot` (liana==1.7.3)</code></td>
<td align="center"><img src="examples/circle_plot_extended_colorby_clustered.png" width="400" style="max-width:100%" title="Extended circle_plot with connector lines, secondary node coloring key, and node clustering"><br><code>extended (colorby="region", cluster_nodes=True)</code></td>
</tr>
</table>

### Label Placement
<table>
<tr>
<td align="center"><img src="examples/circle_plot_extended_colorby_none.png" width="400" style="max-width:100%" title="Extended circle_plot with only connector lines"><br><code>colorby=None</code></td>
<td align="center"><img src="examples/circle_plot_extended_manual.png" width="400" style="max-width:100%" title="Extended circle_plot with connector lines, colorby, no node clustering, and LIANA's default label placement"><br><code>cluster_nodes=False, label_placement="manual"</code></td>
</tr>
<tr>
<td align="center"><img src="examples/circle_plot_extended_hybrid.png" width="400" style="max-width:100%" title="Extended circle_plot with connector lines, colorby, no node clustering, and hybrid label placement (inward/outward for top/bottom, outward for sides)"><br><code>cluster_nodes=False, label_placement="hybrid"</code></td>
<td align="center"><img src="examples/circle_plot_extended_alternating.png" width="400" style="max-width:100%" title="Extended circle_plot with connector lines, colorby, no node clustering, and alternating inward/outward label placement"><br><code>cluster_nodes=False, label_placement="alternating"</code></td>
</tr>
<tr>
<td align="center"><img src="examples/circle_plot_extended_outward.png" width="400" style="max-width:100%" title="Extended circle_plot with connector lines, colorby, no node clustering, and radially outward label placement"><br><code>cluster_nodes=False, label_placement="outward"</code></td>
<td align="center"><img src="examples/circle_plot_extended_no_label.png" width="400" style="max-width:100%" title="Extended circle_plot with colorby, no node clustering, and disabled node labels and connector drawing"><br><code>cluster_nodes=False, label_placement=None</code></td>
</tr>
</table>

> Note: Node labels are blurred to protect unpublished data.