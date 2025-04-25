import pyqtgraph as pg
import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


pg.setConfigOption('leftButtonPan', False)
pg.setConfigOption('background', 'white')
pg.setConfigOption('foreground', 'k')


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        super(FigureCanvasQTAgg, self).__init__()
        fig = Figure()
        self.axes = fig.add_subplot(111)
        super().__init__(fig)



def create_pyqtgraph_widget(layout=None,
                            title="pyqtgraph widget",
                            number_of_references=1):
    window = pg.GraphicsLayoutWidget()
    layout.addWidget(window)
    plot_item = window.addPlot()
    plot_item.setTitle(title, size='18', color='k')
    plot_item.addLegend()

    _ref = np.arange(1, number_of_references+1)

    references = {}
    for i in _ref:
        references[i] = plot_item.plot()

    return plot_item, references