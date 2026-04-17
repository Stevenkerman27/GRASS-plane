import matplotlib.pyplot as plt

class DynamicPlot():
    def __init__(self, title, xdata, ydata, shape=(1, 1), subplot_configs=None, figsize=None):
        """
        ydata: dict of all lines: {label: data_list}
        shape: (rows, cols)
        subplot_configs: list of dicts, each dict defines a subplot:
                         {'title': str, 'lines': [label1, label2, ...]}
        figsize: tuple (width, height), default is (4*cols, 3*rows)
        """
        if len(xdata) == 0:
            return
        plt.ion()
        if figsize is None:
            figsize = (4 * shape[1], 3 * shape[0])
        self.fig, self.axs = plt.subplots(shape[0], shape[1], figsize=figsize)
        if shape == (1, 1):
            self.axs = [self.axs]
        else:
            self.axs = self.axs.flatten()
            
        self.fig.suptitle(title, color='C0')
        self.yline = {}
        self.xdata = xdata
        
        # If no config provided, put all lines in the first subplot
        if subplot_configs is None:
            subplot_configs = [{'title': title, 'lines': list(ydata.keys())}]
            
        for i, config in enumerate(subplot_configs):
            if i >= len(self.axs): break
            ax = self.axs[i]
            ax.set_title(config['title'])
            ax.set_xlim(xdata[0], xdata[-1])
            ax.grid(True)
            
            for j, label in enumerate(config['lines']):
                if label not in ydata: continue
                data = ydata[label]
                # Cycle through colors, but skip C0 as it's used for suptitle sometimes
                color_idx = (j + 1) % 9
                self.yline[label], = ax.plot(xdata, data, f'C{color_idx}', label=" ".join(label.split('_')))
            ax.legend()

    def setxlim(self, xliml, xlimh):
        for ax in self.axs:
            ax.set_xlim(xliml, xlimh)

    def setylim(self, *args):
        if len(args) == 2:
            yliml, ylimh = args
            subplot_idx = 0
        elif len(args) == 3:
            subplot_idx, yliml, ylimh = args
        else:
            raise TypeError("setylim() takes 2 or 3 positional arguments but {} were given".format(len(args)))
            
        if subplot_idx < len(self.axs):
            self.axs[subplot_idx].set_ylim(yliml, ylimh)

    def update_plots(self, ydata, auto_yscale=True):
        # Update each axis's y-limits based on current data to ensure visibility
        for i, ax in enumerate(self.axs):
            # This is a bit expensive but keeps the plot readable
            min_y, max_y = float('inf'), float('-inf')
            has_data = False
            
            # Subplot configs logic to find which lines belong to which ax
            # For simplicity, we just check which lines are already drawn on this ax
            for line in ax.get_lines():
                label = line.get_label().replace(" ", "_")
                if label in ydata:
                    data = ydata[label]
                    # Filter out None values
                    valid_data = [v for v in data if v is not None]
                    if valid_data:
                        line.set_ydata(data)
                        min_y = min(min_y, min(valid_data))
                        max_y = max(max_y, max(valid_data))
                        has_data = True
            
            if has_data and auto_yscale:
                margin = (max_y - min_y) * 0.1 if max_y != min_y else 0.1
                ax.set_ylim(min_y - margin, max_y + margin)
                
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def save(self, filename):
        self.fig.savefig(filename)
        print(f"Loss curve saved to {filename}")
