# -----------------------------------------------------------------------------
# Original implementation by Maxime Brunner (Messina Lab, CHUV, 2026)
# Modified by Jérémy Rotzetter to include:
#   - Added Seurat-style pre-filtering (`min_pct`, `logfc_threshold`) for speed and memory efficiency.
#   - Improved input validation, edge case and error handling.
#   - Fixed AnnData view management: now forces a clean copy to prevent side-effects and illico errors.
#   - Enforced categorical dtype for cluster labels and implemented natural sorting (e.g., 1, 2, 10) for alphanumeric strings, ensuring intuitive cluster ordering.
#   - Improved statistical stability with symmetric log-ratio `diff` metric (enables unified thresholds for up/down regulation).
# -----------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import scanpy as sc
from illico import asymptotic_wilcoxon
import re

def natural_sort_key(text):
    """
    Generates a key for natural sorting by splitting text into numeric 
    and non-numeric parts. Numbers are converted to integers for correct ordering.
    
    Example:
        'Cluster 2'  -> ['Cluster ', 2]
        'Cluster 10' -> ['Cluster ', 10]
        # Result: 'Cluster 2' sorts before 'Cluster 10'
    """
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', str(text))]

def find_all_markers(
    adata,
    group_keys='leiden',
    layer='normalized',
    reference=None,
    n_threads=4,
    pval_cutoff=0.01,
    min_pct=0.01,       # Seurat-like pre-filter: min fraction of cells expressing gene
    logfc_threshold=0.0, # Seurat-like pre-filter: min abs log-fold change
    outdir=None,
    filename='all_markers.csv',
    save_csv=True,
):
    """
    Fast, Seurat/Scanpy-compatible marker detection using illico.

    Original Author
    ---------------
    Maxime Brunner (Messina Lab, CHUV, 2026)
    Initial implementation of illico/Scanpy integration.
    
    Modified by Jérémy Rotzetter
    - Added Seurat-style pre-filtering (`min_pct`, `logfc_threshold`) for speed and memory efficiency.
    - Improved input validation, edge case and error handling.
    - Fixed AnnData view management: now forces a clean copy to prevent side-effects and illico errors.
    - Enforced categorical dtype for cluster labels and implemented natural sorting (e.g., 1, 2, 10) for alphanumeric strings, ensuring intuitive cluster ordering.
    - Improved statistical stability with symmetric log-ratio `diff` metric (enables unified thresholds for up/down regulation).
    
    Parameters
    ----------
    adata : AnnData
        Input annotated data matrix.
    group_keys : str
        Column in adata.obs containing cluster labels.
    layer : str
        Layer to use for expression values (e.g., 'normalized').
    reference : str or None
        Reference group for comparison. If None, uses 'rest' (One-vs-Rest).
    n_threads : int
        Number of threads for illico.
    pval_cutoff : float
        Adjusted p-value threshold for final output.
    min_pct : float
        Global pre-filter: Only test genes detected in >= this fraction 
        of cells across the ENTIRE dataset. Speeds up computation.
        Note: Unlike Seurat (which checks per-cluster), this removes genes 
        globally if they are rare overall, even if enriched in a small cluster.
        Set to 0 to test all genes (default: 0.01).
    logfc_threshold : float
        Only return genes with an absolute log-fold change >= this value.
    outdir : str or None
        Directory to save CSV. Required if save_csv=True.
    filename : str
        Output CSV filename.
    save_csv : bool
        Whether to save the results to CSV.
        
    Returns
    -------
    pd.DataFrame
        Combined marker table for all clusters, sorted by cluster and score.
    """
    # 1. Input Validation
    if layer not in adata.layers:
        raise ValueError(f"Layer '{layer}' not found in adata.layers.")
    if group_keys not in adata.obs:
        raise ValueError(f"Column '{group_keys}' not found in adata.obs.")
    
    # Ensure categorical
    if not isinstance(adata.obs[group_keys].dtype, pd.CategoricalDtype):
        adata.obs[group_keys] = adata.obs[group_keys].astype('category')
    
    groups = adata.obs[group_keys]
    X = adata.layers[layer]
    
    # 2. Pre-filtering (Seurat Style: min.pct)
    # Calculate global detection to filter genes BEFORE testing
    valid_genes = np.ones(adata.n_vars, dtype=bool)
    if min_pct > 0:
        gene_detection = (X > 0).mean(axis=0)
        if hasattr(gene_detection, 'A1'):
            gene_detection = gene_detection.A1
        valid_genes = gene_detection >= min_pct
        
    # Subset adata (even if keeping all genes)
    # Ensure adata_test is not a view before passing to illico
    # This prevents potential errors if the original adata was a view
    adata_test = adata[:, valid_genes].copy()

    # Prepare layer for illico (must be sorted CSR)
    adata_test.layers[layer].sort_indices()
    
    # 3. Run illico Wilcoxon
    results = asymptotic_wilcoxon(
        adata_test,
        is_log1p=True,
        group_keys=group_keys,
        reference=reference,
        n_threads=n_threads,
        layer=layer,
        return_as_scanpy=True,
    )
    
    # Explicitly assign results to the test object's .uns
    adata_test.uns['rank_genes_groups'] = results
    
    # 4. Compute pct.1 / pct.2 (Scanpy 'pts' logic)
    X_calc = adata_test.layers[layer]
    groups_calc = adata_test.obs[group_keys]
    
    pct_by_cluster = {}
    for g in groups_calc.cat.categories:
        mask = (groups_calc == g).values
        pct_in = np.asarray((X_calc[mask] > 0).mean(axis=0)).ravel() * 100
        pct_out = np.asarray((X_calc[~mask] > 0).mean(axis=0)).ravel() * 100
        
        pct_by_cluster[g] = pd.DataFrame(
            {'pct.1': pct_in, 'pct.2': pct_out}, index=adata_test.var_names
        )

    # 5. Build DataFrame
    all_dfs = []
    # Safe sort: handles numeric, string, and mixed alphanumeric labels naturally
    sorted_clusters = sorted(groups_calc.cat.categories, key=natural_sort_key)
    
    for cl in sorted_clusters:
        df = sc.get.rank_genes_groups_df(adata_test, group=cl)
        df = df.merge(pct_by_cluster[cl], left_on='names', right_index=True, how='left')
        
        # Apply logfc threshold (min_pct pre-filter already handled in step 2)
        if logfc_threshold > 0:
            df = df[np.abs(df['logfoldchanges']) >= logfc_threshold]

        # Calculate metrics
        df['diff'] = np.log2((df['pct.1'] + 1e-3) / (df['pct.2'] + 1e-3))
        df['direction'] = np.where(df['logfoldchanges'] > 0, 'up', 'down')
        
        # Final p-value cutoff
        df = df[df['pvals_adj'] < pval_cutoff]
        df = df.sort_values('scores', ascending=False)
        df['cluster'] = cl
        
        # Select and order columns
        cols = ['cluster', 'names', 'direction', 'logfoldchanges', 'pvals_adj',
                'scores', 'pct.1', 'pct.2', 'diff']
        cols = [c for c in cols if c in df.columns]
        df = df[cols]

        # Logging
        n_up = (df['direction'] == 'up').sum()
        n_down = (df['direction'] == 'down').sum()
        print(f"Cluster {cl}: {len(df)} genes - {n_up} up / {n_down} down")

        all_dfs.append(df)

    if not all_dfs:
        print("No markers found matching the criteria.")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    # 6. Save to CSV
    if save_csv:
        if outdir is None:
            raise ValueError("save_csv=True requires an outdir to be specified.")
        os.makedirs(outdir, exist_ok=True)
        outpath = os.path.join(outdir, filename)
        combined.to_csv(outpath, index=False)
        print(f"Saved combined markers ({len(combined)} rows) -> {outpath}")

    return combined