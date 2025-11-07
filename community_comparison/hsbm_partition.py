from typing import Dict, List, Any, Set, Tuple, Optional
import numpy as np
import networkx as nx

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def _adjacency_from_nx(G: nx.Graph, idx_of: Dict[Any, int], binarize: bool = True) -> np.ndarray:
    n = len(idx_of)
    A = np.zeros((n, n), dtype=float)
    for u, v, d in G.edges(data=True):
        i, j = idx_of[u], idx_of[v]
        w = float(d.get("weight", 1.0))
        A[i, j] = A[j, i] = (1.0 if (binarize and w > 0) else w)
    np.fill_diagonal(A, 0.0)
    return A


def _ase_dense(A: np.ndarray, dim: Optional[int] = None) -> np.ndarray:
    # Symmetrize
    A = 0.5 * (A + A.T)
    # Eigen-decompose
    vals, vecs = np.linalg.eigh(A)
    # Sort by absolute value (magnitude), smallest→largest, then take the largest-k by |λ|
    order = np.argsort(np.abs(vals))
    vals, vecs = vals[order], vecs[:, order]
    # If dim not given, choose k by a simple elbow on |λ|
    if dim is None:
        sig = np.abs(vals)
        # work only on the top half to avoid tiny noise
        tail = sig[-min(len(sig), 64):]
        gaps = np.diff(tail)
        k = 1 if gaps.size == 0 else (np.argmax(gaps) + 1)
    else:
        k = max(1, min(dim, len(vals)))
    # Take top-k by |λ|
    vals_k = vals[-k:]
    vecs_k = vecs[:, -k:]
    # ASE uses sqrt(|λ|)
    return vecs_k * np.sqrt(np.abs(vals_k))


def _ase_sparse(A: np.ndarray, dim: Optional[int] = None) -> np.ndarray:
    S = csr_matrix(0.5 * (A + A.T))
    n = A.shape[0]
    # ask for a decent number; for elbow we need >k if dim is None
    k_try = min(32, n - 1) if dim is None else min(dim, n - 1)
    if k_try <= 0:
        return np.zeros((n, 1))
    # largest magnitude eigenpairs
    vals, vecs = eigsh(S, k=k_try, which="LM")
    # sort by |λ|
    order = np.argsort(np.abs(vals))
    vals, vecs = vals[order], vecs[:, order]

    if dim is None:
        sig = np.abs(vals)
        tail = sig[-min(len(sig), 64):]
        gaps = np.diff(tail)
        k = 1 if gaps.size == 0 else (np.argmax(gaps) + 1)
        k = max(1, min(k, len(vals)))
    else:
        k = max(1, min(dim, len(vals)))

    vals_k = vals[-k:]
    vecs_k = vecs[:, -k:]
    return vecs_k * np.sqrt(np.abs(vals_k))



def _ase(A: np.ndarray, dim: Optional[int] = None) -> np.ndarray:
    if _HAS_SCIPY and A.size > 2_000_000:
        try:
            return _ase_sparse(A, dim=dim)
        except Exception:
            pass
    return _ase_dense(A, dim=dim)


def _kmeans_lloyd(X: np.ndarray, K: int, seed: int = 0, iters: int = 50, restarts: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n, d = X.shape
    best_inertia = np.inf
    best_z = None
    for _ in range(restarts):
        idx = rng.choice(n, size=K, replace=False)
        C = X[idx].copy()
        z = np.argmin(((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2), axis=1)
        for _ in range(iters):
            for k in range(K):
                mask = (z == k)
                if np.any(mask):
                    C[k] = X[mask].mean(axis=0)
                else:
                    C[k] = X[rng.integers(0, n)]
            z_new = np.argmin(((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2), axis=1)
            if np.all(z_new == z):
                break
            z = z_new
        inertia = ((X - C[z]) ** 2).sum()
        if inertia < best_inertia:
            best_inertia = inertia
            best_z = z.copy()
    return best_z  # type: ignore


def _mle_sbm(A: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = A.shape[0]
    K = int(z.max()) + 1
    E = np.zeros((K, K), dtype=float)
    C = np.zeros((K, K), dtype=float)
    for i in range(n):
        zi = z[i]
        for j in range(i + 1, n):
            zj = z[j]
            E[zi, zj] += A[i, j]
            E[zj, zi] += A[i, j]
            C[zi, zj] += 1.0
            C[zj, zi] += 1.0
    with np.errstate(divide='ignore', invalid='ignore'):
        P = np.where(C > 0, E / C, 1e-9)
        P = np.clip(P, 1e-9, 1 - 1e-9)
    pi = np.bincount(z, minlength=K).astype(float) / n
    return P, pi


def _sbm_loglik(A: np.ndarray, z: np.ndarray) -> float:
    P, _ = _mle_sbm(A, z)
    ll = 0.0
    n = A.shape[0]
    for i in range(n):
        zi = z[i]
        for j in range(i + 1, n):
            zj = z[j]
            p = P[zi, zj]
            a = A[i, j]
            ll += a * np.log(p) + (1.0 - a) * np.log(1.0 - p)
    return ll


def _sbm_bic(A: np.ndarray, z: np.ndarray) -> float:
    n = A.shape[0]
    m = n * (n - 1) / 2.0
    K = int(z.max()) + 1
    params = K * (K + 1) / 2.0 + (K - 1)
    ll = _sbm_loglik(A, z)
    return 2.0 * ll - params * np.log(max(m, 1.0))


def _best_split(A_sub: np.ndarray, seed: int = 0, candidates=(2, 3, 4)) -> Tuple[Optional[np.ndarray], float, float]:
    n = A_sub.shape[0]
    z_single = np.zeros(n, dtype=int)
    bic_single = _sbm_bic(A_sub, z_single)
    X = _ase(A_sub)
    best_z, best_bic = None, bic_single
    for K in candidates:
        if K >= n:
            continue
        z = _seeded_subspace_cluster(X, K, seed=seed)
        bic = _sbm_bic(A_sub, z)
        if bic > best_bic:
            best_bic, best_z = bic, z
    return best_z, best_bic, bic_single


def hsbm_multi_level(
    G: nx.Graph,
    max_depth: Optional[int] = 4,
    min_size: int = 20,
    seed: int = 0,
    binarize: bool = True,
    candidates=(2, 3, 4),
) -> Dict[int, List[Set[Any]]]:
    """
    Hierarchical community detection (HSBM-style, no motif step).

    Parameters
    ----------
    G : nx.Graph
        Undirected (simple) graph.
    max_depth : int or None
        Maximum recursion depth (root = 0). None = unlimited.
    min_size : int
        Do not attempt to split subgraphs smaller than this.
    seed : int
        RNG seed for seeded subspace clustering / tie-breaking.
    binarize : bool
        If True, treat any positive edge weight as 1.
    candidates : tuple[int]
        Candidate numbers of clusters to try at each split.

    Returns
    -------
    levels : dict[int, list[set]]
        Mapping level -> list of communities (as sets of original node IDs).
        Level 0 is the first split of the whole graph (or the trivial partition).
    """
    # Map nodes <-> indices
    idx_of: Dict[Any, int] = {}
    node_of: Dict[int, Any] = {}
    for i, n in enumerate(G.nodes()):
        idx_of[n] = i
        node_of[i] = n

    # Build adjacency once
    A = _adjacency_from_nx(G, idx_of, binarize=binarize)
    n = A.shape[0]

    # Queue of (indices_of_subgraph, level)
    root = np.arange(n, dtype=int)
    queue: List[Tuple[np.ndarray, int]] = [(root, 0)]

    # Per-level raw labels over ALL nodes; -1 means "unchanged at this level"
    raw_labels: Dict[int, np.ndarray] = {}

    # BFS over levels (top-down)
    while queue:
        idx, level = queue.pop(0)

        # Stopping conditions
        if (max_depth is not None) and (level >= max_depth):
            continue
        if idx.size < min_size:
            continue

        # Try to split this subgraph
        A_sub = A[np.ix_(idx, idx)]
        z, best_bic, bic_single = _best_split(A_sub, seed=seed + level, candidates=candidates)
        if z is None or best_bic <= bic_single:
            # No beneficial split
            continue

        # Initialize the level array once
        if level not in raw_labels:
            raw_labels[level] = np.full(n, -1, dtype=int)

        # --- KEY FIX: ensure unique labels within this level across disjoint subgraphs ---
        if np.any(raw_labels[level] >= 0):
            base = raw_labels[level][raw_labels[level] >= 0].max() + 1
        else:
            base = 0
        raw_labels[level][idx] = z + base
        # -------------------------------------------------------------------------------

        # Enqueue children at next level for further splitting
        K = int(z.max()) + 1
        child_level = level + 1
        for k in range(K):
            child_idx = idx[z == k]
            if child_idx.size > 0:
                queue.append((child_idx, child_level))

    # If nothing was split at all, return a single community at level 0
    if not raw_labels:
        return {0: [set(G.nodes())]}

    # Assemble hierarchical partitions level by level
    levels: Dict[int, List[Set[Any]]] = {}

    # Level 0: start from either its labels or a trivial single cluster
    if 0 in raw_labels and np.any(raw_labels[0] >= 0):
        prev = raw_labels[0].copy()
        prev[prev < 0] = 0  # fill unsplit nodes with a default label
    else:
        prev = np.zeros(n, dtype=int)
    levels[0] = _labels_to_sets(prev, node_of)

    # Higher levels: refine only where that level produced labels
    max_level = max(raw_labels.keys())
    for L in range(1, max_level + 1):
        if L in raw_labels and np.any(raw_labels[L] >= 0):
            rl = raw_labels[L]
            tmp = prev.copy()
            mask = (rl >= 0)
            tmp[mask] = rl[mask]

            # Compact labels to 0..K-1 to keep outputs clean
            uniq = np.unique(tmp)
            remap = {u: i for i, u in enumerate(uniq)}
            prev = np.array([remap[v] for v in tmp], dtype=int)

            levels[L] = _labels_to_sets(prev, node_of)
        else:
            # If no new splits at this level, you can either:
            # (a) carry forward the previous level (uncomment next line), or
            # (b) skip emitting this level. We'll skip to keep only "real" levels.
            # levels[L] = _labels_to_sets(prev, node_of)
            pass

    return levels



def _labels_to_sets(labels: np.ndarray, node_of: Dict[int, Any]) -> List[Set[Any]]:
    K = int(labels.max()) + 1
    out = [set() for _ in range(K)]
    for i, lab in enumerate(labels):
        if lab >= 0:
            out[lab].add(node_of[i])
    return [c for c in out if c]


def hsbm_communities(
    G: nx.Graph,
    seed: int = 0,
    min_size: int = 20,
    binarize: bool = True,
    candidates=(2, 3, 4),
) -> List[Set[Any]]:
    idx_of: Dict[Any, int] = {}
    node_of: Dict[int, Any] = {}
    for i, n in enumerate(G.nodes()):
        idx_of[n] = i
        node_of[i] = n
    if len(idx_of) < min_size:
        return [set(G.nodes())]
    A = _adjacency_from_nx(G, idx_of, binarize=binarize)
    z, best_bic, bic_single = _best_split(A, seed=seed, candidates=candidates)
    if (z is None) or (best_bic <= bic_single):
        return [set(G.nodes())]
    K = int(z.max()) + 1
    comms = [set() for _ in range(K)]
    for i in range(A.shape[0]):
        comms[int(z[i])].add(node_of[i])
    return [c for c in comms if c]


def _seeded_subspace_cluster(X: np.ndarray, K: int, seed: int = 0) -> np.ndarray:
    """
    Seeded nearest-subspace clustering: pick K seeds, improve them for near-orthogonality,
    then assign each point by max inner product to a seed.
    Deterministic given `seed`.
    """
    n, d = X.shape
    rng = np.random.default_rng(seed)

    # init seeds
    seeds_idx = rng.choice(n, size=K, replace=False)
    S = X[seeds_idx].copy()

    # improve seeds
    for i in range(n):
        x = X[i]
        ip_S = S @ S.T
        np.fill_diagonal(ip_S, -np.inf)
        worst = np.max(ip_S)             # largest pairwise inner product in S
        best_to_S = float(np.max(S @ x)) # best similarity of x to any seed
        if best_to_S < worst:
            a, b = np.unravel_index(np.argmax(ip_S), ip_S.shape)
            S[b] = x

    # assign
    z = np.argmax(X @ S.T, axis=1)
    return z.astype(int)




def _adjacency_from_nx_sparse(G: nx.Graph, idx_of: Dict[Any, int], binarize: bool = True) -> csr_matrix:
    rows, cols, data = [], [], []
    for u, v, d in G.edges(data=True):
        i, j = idx_of[u], idx_of[v]
        w = float(d.get("weight", 1.0))
        w = 1.0 if (binarize and w > 0) else w
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([w, w])
    n = len(idx_of)
    A = csr_matrix((data, (rows, cols)), shape=(n, n))
    A.setdiag(0.0)
    A.eliminate_zeros()
    return A