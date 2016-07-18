"""This script creates some example plots for the README.md."""
from __future__ import print_function, division
from tplotter import TestPlotter
from check_tplotter import show_chart, add_noise
import numpy as np

def main():
    """Create the example plots in the following way:
    1. Generate example data (all plots use more or less the same data)
    2. Generate plot 1: "standard" example with predicted and actual labels
    3. Generate plot 2: Same as 1, but no actual labels (only predicted labels)
    4. Generate plot 3: Same as 1, but use only 1 predicted label for every 10th
                        actual label
    """
    nb_points = 500

    y_predicted = add_noise(np.linspace(0.9, 0.1, num=nb_points), 0.025)
    y_actual = add_noise(np.linspace(0.7, 0.3, num=nb_points), 0.045)

    # Normal example plot
    tp = TestPlotter(save_to_filepath="example_plot.png")
    show_chart(y_predicted, y_actual, tp=tp,
               title="Example Plot with Predicted and Actual Labels")

    # Plot showing only predicted labels
    tp = TestPlotter(save_to_filepath="example_plot_only_prediction.png")
    show_chart(y_predicted, np.array([]), tp=tp,
               title="Example Plot, only Predicted Labels / no Actual Labels")

    # Plot with a different update interval for predicted and actual labels
    # (i.e. only one prediction value for every 10 actual labels)
    #
    # Set 9 out of 10 prediction values to -1, which will be transformed into
    # None in show_chart(). (same technique as in check_tplotter.py)
    nb_points_actual = nb_points
    nb_points_predicted = int(nb_points * 0.1)
    all_indices = np.arange(0, nb_points_actual-1, 1)
    keep_indices = np.arange(0, nb_points_actual-1, int(nb_points_actual / nb_points_predicted))
    set_to_none_indices = np.delete(all_indices, keep_indices)
    y_predicted[set_to_none_indices] = -1.0
    tp = TestPlotter(save_to_filepath="example_plot_update_intervals.png")
    show_chart(y_predicted, y_actual, tp=tp,
               title="Example Plot with different Update Intervals for Actual " \
                     "and Predicted Labels")

if __name__ == "__main__":
    main()
