# About

The TestPlotter is a small class to generate plots during the testing of machine learning algorithms (specifically neural networks) showing the following values:
* _Actual_ data labels, when applied to a _test_ dataset.
* _Predicted_ data labels from the current model, when applied to a _test_ dataset.

(TestPlotter, and this ReadMe, is adapted from the similar loss and accuracy plotter (for plotting loss and accuracy values during the training of a machine learning model), at [LossAccPlotter](https://github.com/aleju/LossAccPlotter) by alehu)

Some Features:
* Automatic regression on your values to predict future values over the next N x-values.
* Automatic generation of averages to get a better estimate of your true performance (i.e. to get rid of variance).
* Option to save the plot automatically to a file (at every update).
* The plot is non-blocking, so your program can train in the background while the plot gets continuously updated.
* Currently builds 2D plots - i.e., predicted and actual 1D values plotted against 1D x-values.

# Requirements

* matplotlib
* numpy
* python 2.7 (only tested in that version - may or may not work in other versions)

# Example images

![Example plot with predicted and actual values](images/example_plot.png?raw=true "Example plot with predicted and actual values")

![Example plot, different update intervals](images/example_plot_update_intervals.png?raw=true "Example plot, different update intervals")

![Example plot, only predicted values](images/example_plot_only_prediction.png?raw=true "Example plot, only prediced values")

# Example code

In order to use the `TestPlotter`, simply copy `tplotter.py` into your project's directory, import `TestPlotter` from the file and then add some values to the plotted lines, as shown in the following examples.

Example loop over 100 test x-values:

```python
from tplotter import TestPlotter

plotter = TestPlotter()

for x_t, y_t in zip(X_test, Y_test):
    # somehow generate predicted values with your model
    y_predicted = your_model.predict(x_t)

    # plot the last values
    plotter.add_values(x_t,
                       y_predicted=y_predicted, y_actual=y_t)

# As the plot is non-blocking, we should call plotter.block() at the end, to
# change it to the blocking-mode. Otherwise the program would instantly end
# and thereby close the plot.
plotter.block()
```

All available settings for the `TestPlotter`:

```python
from tplotter import TestPlotter

# What these settings do:
# - title: A title shown at the top of the plot.
# - save_to_filepath: File to save the plot to at every update.
# - show_regressions: Whether to predict future values with regression.
# - show_averages: Whether to show moving averages for all lines.
# - show_plot_window: Whether to show the plot as a window (on e.g. clusters you might want to deactivate that and only save to a file).
# - x_label: Label of the x-axis.
# - y_label: Label of the y-axis.
plotter = TestPlotter(title="This is an example plot",
                         save_to_filepath="/tmp/my_plot.png",
                         show_regressions=True,
                         show_averages=True,
                         show_plot_window=True,
                         x_label="Date",
                         y_label="Price")

# ...
```

You don't have to provide values for all lines at every x-value:

```python
from tplotter import TestPlotter

plotter = TestPlotter()
i=1
for x_t, y_t in zip(X_test, Y_test):

    # plot predicted label only every 25th test x-value
    # the prediction line will be smoother than the line of the actual dataset
    if i % 25 == 0:
        y_predicted = your_model.predict(x_t)
    else:
        y_predicted = None, None

    plotter.add_values(x_t,
                       y_predicted=y_predicted, y_actual=y_t)
plotter.block()
```

When adding many values in a row (e.g. when loading a history from a file), you should add `redraw=False` to the `add_values()` call, otherwise the plotter will spend a lot of time rebuilding the chart many times:

```python
from tplotter import TestPlotter
import numpy as np

plotter = TestPlotter()

# generate some example values for the loss training line
example_values = np.linspace(0.8, 0.1, num=100)

# add them all
for x, value in enumerate(example_values):
    # deactivate redrawing after each update
    plotter.add_values(x, y_predicted=value, redraw=False)

# redraw once at the end
plotter.redraw()

plotter.block()
```


# Tests

There are no automated tests, as it's rather hard to measure the quality of a plot automatically.
You can however run a couple of checks on the library, which show various example plots.
These plots should all look plausible and "beautiful".
Run the checks via:

```python
python check_tplotter.py
```

# License

MIT
