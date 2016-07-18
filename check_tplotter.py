"""Various checks to make sure that the testerPlotter() works decently.
These checks do not run fully automatically, they have to be validated by
a human.

How to run them:
    python check_tplotter.py
"""
from __future__ import print_function, division
from tplotter import TestPlotter
import numpy as np
from time import sleep

def main():
    """Run various checks on the LossAccPlotter.
    They all follow the same pattern: Generate some random data (lines)
    to display. Then display them (using various settings).
    """
    print("")
    print("------------------")
    print("1 datapoint")
    print("------------------")
    # generate example values for: predicted labels, actual labels
    (y_predicted, y_actual) = create_values(1)

    # generate a plot showing the example values
    show_chart(y_predicted, y_actual,
               title="A single datapoint")

    print("")
    print("------------------")
    print("150 datapoints")
    print("Saved to file 'plot.png'")
    print("------------------")
    (y_predicted, y_actual) = create_values(150)
    show_chart(y_predicted, y_actual,
               tp=TestPlotter(save_to_filepath="plot.png"),
               title="150 datapoints, saved to file 'plot.png'")
    '''
    print("")
    print("------------------")
    print("150 datapoints")
    print("No actual chart")
    print("------------------")
    (y_predicted, _) = create_values(150)
    show_chart(y_predicted, np.array([]),
               tp=TestPlotter(show_actual_plot=False),
               title="150 datapoints, no actual chart")

    print("")
    print("------------------")
    print("150 datapoints")
    print("No predicted chart")
    print("------------------")
    (_, y_actual) = create_values(150)
    show_chart(np.array([]), y_actual,
               tp=TestPlotter(show_predicted_plot=False),
               title="150 datapoints, no predicted chart")

    print("")
    print("------------------")
    print("150 datapoints")
    print("No actual chart")
    print("------------------")
    (y_predicted, _) = create_values(150)
    show_chart(y_predicted, np.array([]),
               tp=TestPlotter(show_actual_plot=False),
               title="150 datapoints, no actual chart")
    '''
    print("")
    print("------------------")
    print("150 datapoints")
    print("No regressions")
    print("------------------")
    (y_predicted, y_actual) = create_values(150)
    show_chart(y_predicted, y_actual,
               tp=TestPlotter(show_regressions=False),
               title="150 datapoints, regressions deactivated")

    print("")
    print("------------------")
    print("150 datapoints")
    print("No averages")
    print("------------------")
    (y_predicted, y_actual) = create_values(150)
    show_chart(y_predicted, y_actual,
               tp=TestPlotter(show_averages=False),
               title="150 datapoints, averages deactivated")

    print("")
    print("------------------")
    print("150 datapoints")
    print("x-index 5 of y_predicted should create a warning as its set to NaN")
    print("------------------")
    (y_predicted, y_actual) = create_values(150)

    # this should create a warning when LossAccPlotter.add_values() gets called.
    y_predicted[5] = float("nan")

    show_chart(y_predicted, y_actual,
               title="150 datapoints, one having value NaN (y predicted at x=5)")

    print("")
    print("------------------")
    print("1000 datapoints training")
    print("100 datapoints validation")
    print("------------------")
    nb_points_predicted = 1000
    nb_points_actual = 100
    (y_predicted, y_actual) = create_values(nb_points_predicted)

    # set 9 out of 10 values of the validation arrays to -1.0 (Which will be
    # interpreted as None in show_chart(). Numpy doesnt support None directly,
    # only NaN, which is already used before to check whether the Plotter
    # correctly creates a warning if any data point is NaN.)
    all_indices = np.arange(0, nb_points_predicted-1, 1)
    keep_indices = np.arange(0, nb_points_predicted-1, int(nb_points_predicted / nb_points_actual))
    set_to_none_indices = np.delete(all_indices, keep_indices)
    y_predicted[set_to_none_indices] = -1.0
    y_actual[set_to_none_indices] = -1.0

    show_chart(y_predicted, y_actual,
               title="1000 training datapoints, but only 100 validation datapoints")

    print("")
    print("------------------")
    print("5 datapoints")
    print("slowly added, one by one")
    print("------------------")
    (y_predicted, y_actual) = create_values(5)
    tp = TestPlotter(title="5 datapoints, slowly added one by one")

    for idx in range(y_predicted.shape[0]):
        tp.add_values(idx,
                       y_predicted=y_predicted[idx], y_actual=y_actual[idx],
                       redraw=True)
        sleep(1.0)

    print("Close the chart to continue.")
    tp.block()

def create_values(nb_points):
    """Generate example (y-)values for all lines with some added random noise.

    Args:
        nb_points: Number of example values
    Returns:
        2 numpy arrays of values as a tuple: (array, array)
        First Array: y predicted
        Second Array: y actual
    """
    # we add a bit more noise (0.1) to the actual data compared to the
    # predicted data (0.05)
    y_predicted = add_noise(np.linspace(0.8, 0.1, num=nb_points), 0.05)
    y_actual = add_noise(np.linspace(0.7, 0.2, num=nb_points), 0.1)
    return (y_predicted, y_actual)

def add_noise(values, scale):
    """Add normal distributed noise to an array.
    Args:
        values: Numpy array of values, e.g. [0.7, 0.6, ...]
        scale: Standard deviation of the normal distribution.
    Returns:
        Input array with added noise.
    """
    return values + np.random.normal(scale=scale, size=values.shape[0])

def show_chart(y_predicted, y_actual, tp=None, title=None):
    """Shows a plot using the LossAccPlotter and all provided values.

    Args:
        y_predicted: predicted label values of the test dataset.
        y_actual: actual label values of the test dataset.
        tp: A TestPlotter-Instance or None. If None then a new TestPlotter
            will be instantiated. (Default is None.)
        title: The title to use for the plot, i.e. TestPlotter.title.
    """
    tp = TestPlotter() if tp is None else tp

    # set the plot title, which will be shown at the very top of the plot
    if title is not None:
        tp.title = title

    # add predicted label line/values
    for idx in range(y_predicted.shape[0]):
        p_val = y_predicted[idx] if y_predicted[idx] != -1.0 else None
        tp.add_values(idx, y_predicted=p_val, redraw=False)

    # add actual labels line/values
    for idx in range(y_actual.shape[0]):
        a_val = y_actual[idx] if y_actual[idx] != -1.0 else None
        tp.add_values(idx, y_actual=a_val, redraw=False)

    # redraw once after adding all values, because that's significantly
    # faster than redrawing many times
    tp.redraw()

    # block at the end so that the plot does not close immediatly.
    print("Close the chart to continue.")
    tp.block()

if __name__ == "__main__":
    main()
