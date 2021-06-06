import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

# plt.style.use('ggplot')

class AnimatedScatter(object):
    """Animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, numpoints=50, x=[], y=[],xfig=16.5,yfig=5):
        self.numpoints = numpoints
        self.stream = self.data_stream(x,y) # the data to plot

        # Setup figure
        self.fig, self.ax = plt.subplots(figsize=(xfig,yfig))
        img = plt.imread("../stanza.png")
        self.ax.imshow(img,zorder=0, extent=[0, xfig, 0, yfig])
        # setup FuncAnimation.
        self.anim = animation.FuncAnimation(self.fig, self.update, interval=50, 
                                           init_func=self.setup_plot, blit=True,
                                           frames=len(x)-1)
        # se interessa il filmino mp4
        self.anim.save('scatter.mp4', writer='ffmpeg')
        

    def data_stream(self, x=[], y=[]):
        """Generate a walk. Data is scaled."""
        xy = np.zeros((self.numpoints,2))
        i = 0
        while True:
            xy[i,0]=x[i];xy[i,1]=y[i]
            i = (i+1) % self.numpoints
            yield np.c_[xy[:,0], xy[:,1]]

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        x, y = next(self.stream).T
        c = None
        s = 100
        self.scat = self.ax.scatter(x, y, c=c, s=s, 
                                    vmin=0, vmax=1,
                                    cmap="jet", edgecolor="k")
        self.ax.axis([0, 16.5, 0, 5])
        self.ax.set_ylabel('wouldbe x')
        self.ax.set_xlabel('wouldbe y')
        # return the updated artist to FuncAnimation
        # It expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def update(self, i):
        """Update the scatter plot."""
        data = next(self.stream)

        # Set x and y data...
        self.scat.set_offsets(data[:, :2])
        # Set sizes...
        self.scat.set_sizes(np.ones(self.numpoints)*75)
        # Set colors..
        self.scat.set_color('#22ff22')
        self.scat.set_edgecolor('#000000')

        # return the updated artist to FuncAnimation
        # It expects a sequence of artists, thus the trailing comma.
        return self.scat,
