"""
Utility functions for this method-merge-tests repository.
"""
import numpy as np
import pandas as pd

import pints


def run_replicates(iterations, n_replicates, test, parallel=False):
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

    # Evaluate the cases in reverse order:
    # - Assuming that the iterations are sorted from low to high, the longest
    #   running tasks are at the end.
    # - If we start with short tasks and end with long ones, the last process
    #   to start will be the last one to finish.
    # - Instead, do the long running tasks first, and then whoever finishes
    #   first can start on the shorter tasks.
    iterations = list(reversed(df['iterations']))
    results = pints.evaluate(test, iterations, parallel=parallel)
    results.reverse()

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


def technicolor_dreamline(ax, x, y, z=None, lw=1):
    """
    Draws a multi-coloured line on a set of matplotlib axes ``ax``.

    The points to plot should be passed in as ``x`` and ``y``, and optionally
    ``z`` for a 3d plot.

    Line width can be set with ``lw``,

    Code adapted from: https://github.com/CardiacModelling/FourWaysOfFitting
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    colormap = 'jet'
    cmap_fix = 1

    x = np.asarray(x)
    y = np.asarray(y)
    if z is not None:
        z = np.asarray(z)

    # Invisible plot for automatic x & y limits
    if z is None:
        ax.plot(x, y, alpha=0)
    else:
        ax.plot(x, y, z, alpha=0)

    # Create collection of line segments
    stride = max(1, int(len(x) / 1000))
    n = 1 + (len(x) - 1) // stride
    segments = []
    for i in range(n):
        lo = i * stride
        hi = lo + stride + 1
        xs = x[lo:hi]
        ys = y[lo:hi]
        if z is None:
            segments.append(np.vstack((xs, ys)).T)
        else:
            zs = z[lo:hi]
            segments.append(np.vstack((xs, ys, zs)).T)
    n = len(segments)

    if z is None:
        Collection = matplotlib.collections.LineCollection
    else:
        Collection = Line3DCollection

    cmap = plt.cm.get_cmap(colormap)
    norm = matplotlib.colors.Normalize(0, cmap_fix)
    idxs = np.linspace(0, 1, n)
    ax.add_collection(
        Collection(segments, cmap=cmap, norm=norm, array=idxs, lw=lw))


def function_between_points(
        ax, f, x_true, x_found, padding=0.25, evaluations=20):
    """
    Like :meth:`pints.plot.function_between_points`, but takes a matplotlib
    axes as first argument.
    """
    import matplotlib.pyplot as plt

    # Check function and get n_parameters
    if not (isinstance(f, pints.LogPDF) or isinstance(f, pints.ErrorMeasure)):
        raise ValueError(
            'Given function must be pints.LogPDF or pints.ErrorMeasure.')
    n_param = f.n_parameters()

    # Check points
    point_1 = pints.vector(x_true)
    point_2 = pints.vector(x_found)
    del(x_true, x_found)
    if not (len(point_1) == len(point_2) == n_param):
        raise ValueError('Both points must have the same number of parameters'
                         + ' as the given function.')

    # Check padding
    padding = float(padding)
    if padding < 0:
        raise ValueError('Padding cannot be negative.')

    # Check evaluation
    evaluations = int(evaluations)
    if evaluations < 3:
        raise ValueError('The number of evaluations must be 3 or greater.')

    # Figure setting
    #ax.set_xlabel('T')
    ax.set_ylabel('Error')

    # Generate some x-values near the given parameters
    s = np.linspace(-padding, 1 + padding, evaluations)

    # Direction
    r = point_2 - point_1

    # Calculate function with other parameters fixed
    x = [point_1 + sj * r for sj in s]
    y = pints.evaluate(f, x, parallel=False)

    # Plot
    ax.plot(s, y, color='green')
    ax.axvline(0, color='#1f77b4', label='True parameters')
    ax.axvline(1, color='#7f7f7f', label='Estimated parameters')
    ax.legend()


def function(axes, f, x, scales=None, evaluations=20):
    """
    Like :class:`pints.plot.function`, but takes a set of axes as input, and
    uses a list of scales instead of lower and upper bounds.
    """
    import matplotlib.pyplot as plt

    # Check function and get n_parameters
    if not (isinstance(f, pints.LogPDF) or isinstance(f, pints.ErrorMeasure)):
        raise ValueError(
            'Given function must be pints.LogPDF or pints.ErrorMeasure.')
    n_param = f.n_parameters()

    # Check axes
    if len(axes) != n_param:
        raise ValueError('Axes list must have length f.n_parameters().')

    # Check point
    x = pints.vector(x)
    if len(x) != n_param:
        raise ValueError('Point x must have length f.n_parameters().')

    # Check scales
    if scales is None:
        # Guess boundaries based on point x
        scales = x * 0.05
    else:
        scales = pints.vector(scales)
        if len(scales) != n_param:
            raise ValueError('Scales must have length f.n_parameters().')
    lower = x - scales
    upper = x + scales

    # Check number of evaluations
    evaluations = int(evaluations)
    if evaluations < 1:
        raise ValueError('Number of evaluations must be greater than zero.')

    # Create points to plot
    xs = np.tile(x, (n_param * evaluations, 1))
    for j in range(n_param):
        i1 = j * evaluations
        i2 = i1 + evaluations
        xs[i1:i2, j] = np.linspace(lower[j], upper[j], evaluations)

    # Evaluate points
    fs = pints.evaluate(f, xs, parallel=False)

    # Create figure
    axes[0].set_xlabel('Function')
    for j, p in enumerate(x):
        i1 = j * evaluations
        i2 = i1 + evaluations
        axes[j].plot(xs[i1:i2, j], fs[i1:i2], c='green', label='Function')
        axes[j].axvline(p, c='blue', label='Value')
        axes[j].set_xlabel('Parameter ' + str(1 + j))
        axes[j].legend()

