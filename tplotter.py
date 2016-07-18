"""A class to generate plots for the predicted output labels vs actual labels
of models trained with machine learning methods. As of now works only on plotting
against a 1D x-value (so produces a 2D, x-y plot).

Based on LossAccPlotter, derived from github.com/aleju/LossAccPlotter

Example:
    plotter = TestPlotter()
    for x_t, y_t in zip(X_test, Y_test):
        y_predicted = your_model.predict(x_t)
        plotter.add_values(x_t,
                           y_predicted=y_predicted, y_actual=y_t)
    plotter.block()
"""
from __future__ import absolute_import
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import warnings
import math
from collections import OrderedDict

def ignore_nan_and_inf(value, label, x_index):
    """Helper function that creates warnings on NaN/INF and converts them to None.
    Args:
        value: The value to check for NaN/INF.
        label: For which line the value was used (usually "y predicted", "y actuall", ...)
            This is used in the warning message.
        x_index: At which x-index the value was used. This is used in the warning message.
    Returns:
        value, but None if value is NaN or INF.
    """
    if value is None:
        return None
    elif math.isnan(value):
        warnings.warn("Got NaN for value '%s' at x-index %d" % (label, x_index))
        return None
    elif math.isinf(value):
        warnings.warn("Got INF for value '%s' at x-index %d" % (label, x_index))
        return None
    else:
        return value

class TestPlotter(object):
    """Class to plot predicted and actual labels charts (for test data)."""
    def __init__(self,
                 title=None,
                 save_to_filepath=None,
                 show_regressions=True,
                 show_averages=True,
                 show_plot_window=True,
                 x_label=None,
                 y_label=None):
        """Constructs the plotter.

        Args:
            title: An optional title which will be shown at the top of the
                plot. E.g. the name of the experiment or some info about it.
                If set to None, no title will be shown. (Default is None.)
            save_to_filepath: The path to a file in which the plot will be saved,
                e.g. "/tmp/last_plot.png". If set to None, the chart will not be
                saved to a file. (Default is None.)
            show_regressions: Whether or not to show a regression (on predicted
                values only), indicating where each line might end up in the future.
            show_averages: Whether to plot moving averages in the charts (on
                predicted values only). This value may only be True or False. To
                change the interval (default is 20 x-values), change the instance
                variable "averages_period" to the new integer value. (Default is True.)
            show_plot_window: Whether to show the plot in a window (True)
                or hide it (False). Hiding it makes only sense if you
                set save_to_filepath. (Default is True.)
            x_label: Label on the x-axis of the chart. (Default is None.)
            y_label: Label on the y-axis of the chart. (Default is None.)
        """
        assert save_to_filepath is not None or show_plot_window

        self.title = title
        self.title_fontsize = 14
        self.show_regressions = show_regressions
        self.show_averages = show_averages
        self.show_plot_window = show_plot_window
        self.save_to_filepath = save_to_filepath
        self.x_label = x_label
        self.y_label = y_label

        # alpha values
        # 0.8 = quite visible line
        # 0.5 = moderately visible line
        # thick is used for averages and regression (also for the main values,
        # if there are no averages),
        # thin is used for the main values
        self.alpha_thick = 0.8
        self.alpha_thin = 0.5

        # the interval for the moving averages, e.g. 20 = average over 20 x-values
        self.averages_period = 20

        # these values deal with the regression
        self.poly_forward_perc = 0.1
        self.poly_backward_perc = 0.2
        self.poly_n_forward_min = 5
        self.poly_n_backward_min = 10
        self.poly_n_forward_max = 100
        self.poly_n_backward_max = 100
        self.poly_degree = 1

        # whether to show grids in both charts
        self.grid = True

        # the styling of the lines
        # sma = simple moving average
        self.linestyles = {
            "y_predicted": "g-",
            "y_predicted_sma": "g-",
            "y_predicted_regression": "g:",
            "y_actual": "b-",
            "y_actual_sma": "b-",
            "y_actual_regression": "b:"
        }
        # different linestyles for the first x-value (if only one value is available),
        # because no line can then be drawn (needs 2+ points) and only symbols will
        # be shown.
        # No regression here, because regression always has at least at least
        # two xy-points (last real value and one (or more) predicted values).
        # No averages here, because the average over one value would be identical
        # to the value anyways.
        self.linestyles_one_value = {
            "y_predicted": "gs-",
            "y_actual": "b^-"
        }

        # these values will be set in _initialize_plot() upon first call of redraw()
        # fig: the figure of the whole plot
        # ax: chart axis
        self.fig = None
        self.ax = None

        # dictionaries with x, y values for each line
        self.values_y_predicted = OrderedDict()
        self.values_y_actual = OrderedDict()

    def add_values(self, x_index, y_predicted=None, y_actual=None,
                   redraw=True):
        """Function to add new values for each line for a specific x-value.

        Meaning of the values / lines:
         - y_predicted: model-predicted labels on the test set.
         - y_actual:    actual labels on the test set.

        Values that are None will be ignored.
        Values that are INF or NaN will be ignored, but create a warning.

        It is currently assumed that added values follow logically after
        each other (progressive order), so the first x_index might be 1 (first entry),
        then 2 (second entry), then 3 (third entry), ...
        Not allowed would be e.g.: 10, 11, 5, 7, ...
        If that is not the case, you will get a broken line graph.

        Args:
            x_index: The x-coordinate.
            y_predicted: The model-predicted label of the test set at the given x_index.
                If None, no value for the predicted label line will be added at
                the given x_index. (Default is None.)
            y_actual: Same as y_predicted for the actual label line.
                (Default is None.)
            redraw: Whether to redraw the plot immediatly after receiving the
                new values. If you add many values in a row, set this to
                False and call redraw() at the end (significantly faster).
                (Default is True.)
        """
        assert isinstance(x_index, (int, long))

        y_predicted = ignore_nan_and_inf(y_predicted, "predicted", x_index)
        y_actual = ignore_nan_and_inf(y_actual, "actual", x_index)

        if y_predicted is not None:
            self.values_y_predicted[x_index] = y_predicted
        if y_actual is not None:
            self.values_y_actual[x_index] = y_actual

        if redraw:
            self.redraw()

    def block(self):
        """Function to show the plot in a blocking way.

        This should be called at the end of your program. Otherwise the
        chart will be closed automatically (at the end).
        By default, the plot is shown in a non-blocking way, so that the
        program continues execution, which causes it to close automatically
        when the program finishes.

        This function will silently do nothing if show_plot_window was set
        to False in the constructor.
        """
        if self.show_plot_window:
            plt.figure(self.fig.number)
            plt.show()

    def save_plot(self, filepath):
        """Saves the current plot to a file.

        Args:
            filepath: The path to the file, e.g. "/tmp/last_plot.png".
        """
        self.fig.savefig(filepath, bbox_inches="tight")

    def _initialize_plot(self):
        """Creates empty figure and axes of the plot and shows it in a new window.
        """
        fig, ax = plt.subplots(ncols=1, figsize=(12, 8))
        self.fig = fig
        self.ax = ax

        # set_position is neccessary here in order to make space at the bottom
        # for the legend
        if self.ax is not None:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1,
                             box.width, box.height * 0.9])

        # draw the title
        # it seems to be necessary to set the title here instead of in redraw(),
        # otherwise the title is apparently added again and again with every
        # x-value, making it ugly and bold
        if self.title is not None:
            self.fig.suptitle(self.title, fontsize=self.title_fontsize)

        if self.show_plot_window:
            plt.show(block=False)

    def redraw(self):
        """Redraws the plot with the current values.

        This is a full redraw and includes recalculating averages and regressions.
        It should not be called many times per second as that would be slow.
        Calling it every couple seconds should create no noticeable slowdown though.

        Args:
            x-value: The index of the current x-value, starting at 0.
            y_predicted: All of the predicted labels of each x-value (list of floats).
            y_actual: All of the actual labels of each x-value (list of floats).
        """
        # initialize the plot if it's the first redraw
        if self.fig is None:
            self._initialize_plot()

        # activate the plot, in case another plot was opened since the last call
        plt.figure(self.fig.number)

        # shorter local variables
        ax = self.ax

        # set chart titles, x-/y-labels and grid
        if ax:
            ax.clear()
            #ax.set_title(self.title)
            ax.set_ylabel(self.y_label)
            ax.set_xlabel(self.x_label)
            ax.grid(self.grid)

        # Plot main lines, their averages and the regressions (predictions)
        self._redraw_main_lines()
        self._redraw_averages()
        self._redraw_regressions()

        # Add legends (below both chart)
        ncol = 1
        labels = ["$CHART predicted", "$CHART actual"]
        if self.show_averages:
            labels.extend(["$CHART predicted (avg %d)" % (self.averages_period,),
                           "$CHART actual (avg %d)" % (self.averages_period,)])
            ncol += 1
        if self.show_regressions:
            labels.extend(["$CHART predicted (regression)",
                           "$CHART actual (regression)"])
            ncol += 1

        if ax:
            ax.legend([label.replace("$CHART", "labels") for label in labels],
                       loc="upper center",
                       bbox_to_anchor=(0.5, -0.08),
                       ncol=ncol)

        plt.draw()

        # save the redrawn plot to a file upon every redraw.
        if self.save_to_filepath is not None:
            self.save_plot(self.save_to_filepath)

    def _redraw_main_lines(self):
        """Draw the main lines of values (i.e. loss train, loss val, acc train, acc val).

        Returns:
            List of handles (one per line).
        """
        handles = []
        ax = self.ax

        # Set the styles of the lines used in the charts
        # Different line style for x-values after the first one, because
        # the very first x-value has only one data point and therefore no line
        # and would be invisible without the changed style.
        v_y_predicted = self.linestyles["y_predicted"]
        v_y_actual = self.linestyles["y_actual"]
        if len(self.values_y_predicted) == 1:
            v_y_predicted = self.linestyles_one_value["y_predicted"]
        if len(self.values_y_actual) == 1:
            v_y_actual = self.linestyles_one_value["y_actual"]

        # Plot the lines
        alpha_main = self.alpha_thin if self.show_averages else self.alpha_thick
        if ax:
            h_p, = ax.plot(self.values_y_predicted.keys(), self.values_y_predicted.values(),
                           v_y_predicted, label="predicted value", alpha=alpha_main)
            h_a, = ax.plot(self.values_y_actual.keys(), self.values_y_actual.values(),
                           v_y_actual, label="actual value", alpha=alpha_main)
            handles.extend([h_p, h_a])

        return handles

    def _redraw_averages(self):
        """Draw the moving averages of each line.

        If moving averages has been deactived in the constructor, this function
        will do nothing.

        Returns:
            List of handles (one per line).
        """
        # abort if moving averages have been deactivated
        if not self.show_averages:
            return []

        handles = []
        ax = self.ax

        # calculate the xy-values
        if ax:
            (p_sma_x, p_sma_y) = self._calc_sma(self.values_y_predicted.keys(),
                                                self.values_y_predicted.values())
            (a_sma_x, a_sma_y) = self._calc_sma(self.values_y_actual.keys(),
                                                self.values_y_actual.values())

        # plot the xy-values
        alpha_sma = self.alpha_thick
        if ax:
            # for loss chart
            h_p, = ax.plot(p_sma_x, p_sma_y, self.linestyles["y_predicted_sma"],
                            label="predicted val (avg %d)" % (self.averages_period,),
                            alpha=alpha_sma)
            h_a, = ax.plot(a_sma_x, a_sma_y, self.linestyles["y_actual_sma"],
                            label="actual val (avg %d)" % (self.averages_period,),
                            alpha=alpha_sma)
            handles.extend([h_p, h_a])
        return handles

    def _redraw_regressions(self):
        """Draw the moving regressions of each line, i.e. the predictions of
        future values.

        If regressions have been deactived in the constructor, this function
        will do nothing.

        Returns:
            List of handles (one per line).
        """
        if not self.show_regressions:
            return []

        handles = []
        ax = self.ax

        # calculate future values for loss train (lt), loss val (lv),
        # acc train (at) and acc val (av)
        if ax:
            # for loss chart
            p_regression = self._calc_regression(self.values_y_predicted.keys(),
                                                 self.values_y_predicted.values())
            a_regression = self._calc_regression(self.values_y_actual.keys(),
                                                 self.values_y_actual.values())

        # plot the predicted values
        alpha_regression = self.alpha_thick
        if ax:
            # for loss chart
            h_p, = ax.plot(p_regression[0], p_regression[1],
                            self.linestyles["y_predicted_regression"],
                            label="predicted val regression",
                            alpha=alpha_regression)
            h_a, = ax.plot(a_regression[0], a_regression[1],
                            self.linestyles["y_actual_regression"],
                            label="actual val regression",
                            alpha=alpha_regression)
            handles.extend([h_p, h_a])

        return handles

    def _calc_sma(self, x_values, y_values):
        """Calculate the moving average for one line (given as two lists, one
        for its x-values and one for its y-values).

        Args:
            x_values: x-coordinate of each value.
            y_values: y-coordinate of each value.

        Returns:
            Tuple (x_values, y_values), where x_values are the x-values of
            the line and y_values are the y-values of the line.
        """
        result_y, last_ys = [], []
        running_sum = 0
        period = self.averages_period
        # use a running sum here instead of avg(), should be slightly faster
        for y_val in y_values:
            last_ys.append(y_val)
            running_sum += y_val
            if len(last_ys) > period:
                poped_y = last_ys.pop(0)
                running_sum -= poped_y
            result_y.append(float(running_sum) / float(len(last_ys)))
        return (x_values, result_y)

    def _calc_regression(self, x_values, y_values):
        """Calculate the regression for one line (given as two lists, one
        for its x-values and one for its y-values).

        Args:
            x_values: x-coordinate of each value.
            y_values: y-coordinate of each value.

        Returns:
            Tuple (x_values, y_values), where x_values are the predicted x-values
            of the line and y_values are the predicted y-values of the line.
        """
        if not x_values or len(x_values) < 2:
            return ([], [])

        # This currently assumes that the last added x-value for the line
        # was indeed that highest x-value.
        # This could be avoided by tracking the max value for each line.
        last_x = x_values[-1]
        nb_values = len(x_values)

        # Compute regression lines based on n_backwards x-values
        # in the past, e.g. based on the last 10 values.
        # n_backwards is calculated relative to the current x-value
        # (e.g. at x-value 100 compute based on the last 10 values,
        # at 200 based on the last 20 values...). It has a minimum (e.g. never
        # use less than 5 x-values (unless there are only less than 5 x-values))
        # and a maximum (e.g. never use more than 1000 x-values).
        # The minimum prevents bad predictions.
        # The maximum
        #   a) is better for performance
        #   b) lets the regression react faster in case you change something
        #      in the hyperparameters after a long time of training.
        n_backward = int(nb_values * self.poly_backward_perc)
        n_backward = max(n_backward, self.poly_n_backward_min)
        n_backward = min(n_backward, self.poly_n_backward_max)

        # Compute the regression lines for the n_forward future x-values.
        # n_forward also has a reletive factor, as well as minimum and maximum
        # values (see n_backward).
        n_forward = int(nb_values * self.poly_forward_perc)
        n_forward = max(n_forward, self.poly_n_forward_min)
        n_forward = min(n_forward, self.poly_n_forward_max)

        # return nothing of the values turn out too low
        if n_backward <= 1 or n_forward <= 0:
            return ([], [])

        # create/train the regression model
        fit = np.polyfit(x_values[-n_backward:], y_values[-n_backward:],
                         self.poly_degree)
        poly = np.poly1d(fit)

        # calculate future x- and y-values
        # we use last_x to last_x+n_forward here instead of
        #        last_x+1 to last_x+1+n_forward
        # so that the regression line is better connected to the current line
        # (no visible gap)
        future_x = [i for i in range(last_x, last_x + n_forward)]
        future_y = [poly(x_idx) for x_idx in future_x]

        return (future_x, future_y)
