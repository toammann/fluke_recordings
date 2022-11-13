import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector


def onselect(min, max):
    print("onlect")
    the_table.add_cell(1, 1, 0.1, 0.1, text="onselect")


fig, ax = plt.subplots()

# Add a table at the bottom of the axes
the_table = plt.table(cellText=("12", "34"),
                      loc='center')

#Create spanSlector widget
span = SpanSelector(
    ax,
    onselect = onselect,
    direction = 'horizontal',
    minspan = 0,
    useblit = True,
    interactive = True,
    button = 1, #left mouse button
    props = {'facecolor':'grey', 'alpha':0.3}
)

plt.show()