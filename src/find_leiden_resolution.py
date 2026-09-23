import scanpy as sc

def find_leiden_resolution(adata, start_res=0.5, step=0.1, max_res=2.0, target_min=60, target_max=70, cleanup=False):
    """
    Runs leiden clustering iteratively until the number of clusters is within [target_min, target_max]
    or resolution exceeds max_res.
    
    Parameters
    ----------
    adata : AnnData
        The annotated data matrix.
    start_res : float
        Starting resolution.
    step : float
        Resolution increment step.
    max_res : float
        Maximum resolution to try.
    target_min : int
        Minimum target number of clusters.
    target_max : int
        Maximum target number of clusters.
    cleanup : bool
        If True, removes intermediate clustering columns from adata.obs.
        
    Returns
    -------
    tuple
        (final_resolution, n_clusters, key_added) or (None, None, None) if target not reached.
    """
    if 'connectivities' not in adata.obsp:
        raise ValueError("Neighbors graph not found. Please run sc.pp.neighbors(adata) first.")

    resolution = start_res
    
    while resolution <= max_res + 1e-6: # Small epsilon for float comparison
        res_rounded = round(resolution, 2) # Round resolution to avoid floating point errors in key naming
        key_added = f"leiden_res_{res_rounded}"
        
        sc.tl.leiden(adata, resolution=res_rounded, key_added=key_added, flavor="igraph")
        n_clusters = adata.obs[key_added].nunique()
        
        print(f"Resolution: {res_rounded} -> Found {n_clusters} clusters")
        
        if target_min <= n_clusters <= target_max:
            print(f"Target reached! {n_clusters} clusters at resolution {res_rounded}.")
            return res_rounded, n_clusters, key_added
        
        # Clean up intermediate results to save memory
        if cleanup:
            del adata.obs[key_added]
        
        resolution += step

    print(f"Target range [{target_min}, {target_max}] not reached within resolution limit {max_res}.")
    last_res = round(resolution - step, 2)
    last_key = f"leiden_res_{last_res}"
    # Ensure the last tested result is kept if cleanup was on
    if cleanup and last_key not in adata.obs:
         sc.tl.leiden(adata, resolution=last_res, key_added=last_key, flavor="igraph")
         n_clusters = adata.obs[last_key].nunique()
    
    return last_res, n_clusters, last_key