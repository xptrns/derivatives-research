import numpy as np

def greedy_d_optimal_selection(fisher_matrices, n_select=None, regularization=1e-10):
    """
    Greedily select option observations using the D-optimality criterion.

    Parameters
    ----------
    fisher_matrices: dict
        Maps each option identifier, e.g. (T, K), to its 5x5 Fisher information matrix.

    n_select: int or None
        Number of observations to select. If None, select all candidates.

    regularization: float
        Small diagonal regularization used when evaluating log-determinants.

    Returns
    -------
    selected: list
        Options in order of selection.

    logdet_history: list
        log(det(I + regularization * I_5)) after each selection.

    """
    candidates = list(fisher_matrices.keys())

    if n_select is None:
        n_select = len(candidates)

    selected = []
    remaining = candidates.copy()

    n_params = next(iter(fisher_matrices.values())).shape[0]
    fisher_current = np.zeros((n_params, n_params))

    logdet_history = []

    for _ in range(n_select):
        best_option = None
        best_logdet = -np.inf

        for option in remaining:
            fisher_candidate = fisher_current + fisher_matrices[option]
            sign, logdet = np.linalg.slogdet(fisher_candidate + regularization * np.eye(n_params))

            if sign > 0 and logdet > best_logdet:
                best_logdet = logdet
                best_option = option

        selected.append(best_option)
        remaining.remove(best_option)

        fisher_current += fisher_matrices[best_option]
        logdet_history.append(best_logdet)

    return selected, logdet_history