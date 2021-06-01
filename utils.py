import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def ecdf_norm_plotter(draws, normal_sd, xrange=np.linspace(-5, 5, 100)):
    from scipy.stats import norm
    from statsmodels.distributions.empirical_distribution import ECDF
    ecdf_fun = ECDF(draws)
    ecdf = [ecdf_fun(y) for y in xrange]
    cdf = [norm.cdf(y, 0, normal_sd) for y in xrange]

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
                          'distance': result['distance'], 'ess': result['mean-ess']}
            k += 1
    df['iterations'] = pd.to_numeric(df['iterations'])
    df['distance'] = pd.to_numeric(df['distance'])
    df['ess'] = pd.to_numeric(df['ess'])
    return df
