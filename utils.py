"""
Utility functions for this method-merge-tests repository.
"""
import numpy as np
import pandas as pd

import pints


def run_replicates(iterations, n_replicates, test):
    df = pd.DataFrame(columns=['iterations', 'replicate', 'kld', 'ess'],
                      index=np.arange(len(iterations) * n_replicates))
    k = 0
    for it in iterations:
        for rep in range(n_replicates):
            result = test(it)
            df.iloc[k] = {'iterations': it, 'replicate': rep,
                          'kld': result['kld'], 'ess': result['mean-ess']}
            k += 1
    df['iterations'] = pd.to_numeric(df['iterations'])
    df['kld'] = pd.to_numeric(df['kld'])
    df['ess'] = pd.to_numeric(df['ess'])
    return df


def run_replicates2(iterations, n_replicates, test, parallel=False):
    """
    Runs ``test(i)`` for all entries ``i`` in ``iterations``, repeating each
    test ``n_replicates`` times.

    The argument ``test`` is expected to return a dictionary of (scalar valued)
    results.

    The returned value is a pandas DataFrame with
    ``len(iterations) * n_replicates`` rows. Each column contains an index, the
    number of iterations performed as ``iterations``, the index of the repeat
    as ``replicate``, followed by the entries of the corresponding test
    result.

    Parallel evaluation can be enabled by setting ``parallel`` to ``True`` or
    to the number of worker processes to use. However, this can cause issues in
    Jupyter notebooks.
    """
    df = pd.DataFrame(index=np.arange(len(iterations) * n_replicates))
    df['iterations'] = np.repeat(iterations, n_replicates)
    df['replicate'] = np.tile(np.arange(n_replicates), len(iterations))

    results = pints.evaluate(test, list(df['iterations']), parallel=parallel)
    assert len(results) > 0, 'Empty result set generated'
    for key in results[0].keys():
        df[key] = np.array([r[key] for r in results], copy=False)

    return df


def ecdf_norm_plotter(draws, normal_sd, x=np.linspace(-5, 5, 100)):
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    from statsmodels.distributions.empirical_distribution import ECDF

    ecdf_fun = ECDF(draws)
    ecdf = [ecdf_fun(y) for y in x]
    cdf = [norm.cdf(y, 0, normal_sd) for y in x]

    x1 = np.linspace(0, 1, 100)
    y = [y for y in x1]
    plt.scatter(ecdf, cdf)
    plt.plot(x1, y, 'k-')
    plt.xlabel('Estimated cdf')
    plt.ylabel('True cdf')
    plt.show()


def run_replicates_distance(iterations, n_replicates, test):
    df = pd.DataFrame(columns=['iterations', 'replicate', 'distance', 'ess'],
                      index=np.arange(len(iterations) * n_replicates))
    k = 0
    for it in iterations:
        for rep in range(n_replicates):
            result = test(it)
            df.iloc[k] = {'iterations': it, 'replicate': rep,
                          'distance': result['distance'],
                          'ess': result['mean-ess']}
            k += 1
    df['iterations'] = pd.to_numeric(df['iterations'])
    df['distance'] = pd.to_numeric(df['distance'])
    df['ess'] = pd.to_numeric(df['ess'])
    return df


def run_replicates_annulus(iterations, n_replicates, test):
    df = pd.DataFrame(columns=['iterations', 'replicate', 'distance', 'ess'],
                      index=np.arange(len(iterations) * n_replicates))
    k = 0
    for it in iterations:
        for rep in range(n_replicates):
            result = test(it)
            df.iloc[k] = {'iterations': it, 'replicate': rep, 'distance':
                          result['distance'], 'ess': result['mean-ess']}
            k += 1
    df['iterations'] = pd.to_numeric(df['iterations'])
    df['distance'] = pd.to_numeric(df['distance'])
    df['ess'] = pd.to_numeric(df['ess'])
    return df
