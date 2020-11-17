import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

# plt.style.use('ggplot')

class AnimatedScatter(object):
    """Animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, numpoints=50, x=[], y=[],xmin=0,xmax=16, ymin=0, ymax=5):
        self.numpoints = numpoints
        xfig = 16.5
        yfig = 5
        self.normalize_xy(x,y,xfig,yfig)
        self.stream = self.data_stream(self.xn,self.yn) # the data to plot

        # Setup figure
        self.fig, self.ax = plt.subplots(figsize=(xfig,yfig))
        img = plt.imread("stanza.PNG")
        self.ax.imshow(img,zorder=0, extent=[0, 16.5, 0, 5])
        # setup FuncAnimation.
        self.anim = animation.FuncAnimation(self.fig, self.update, interval=100, 
                                           init_func=self.setup_plot, blit=True,
                                           frames=len(x)-1)
        # se interessa il filmino mp4
        self.anim.save('scatter.mp4', writer='ffmpeg')
        
    def normalize_xy(self,x,y,xfig,yfig):
        xmin = min(x.min(),x.min())
        xmax = max(x.max(),x.max())
        ymin = min(y.min(),y.min())
        ymax = max(y.max(),y.max())
        self.xn = x #(x-xmin)/(xmax-xmin)*xfig
        self.yn = y #(y-ymin)/(ymax-ymin)*yfig

    def data_stream(self, x=[], y=[]):
        """Generate a walk. Data is scaled."""
        xy = np.zeros((self.numpoints,2))
        s, c = np.random.random((self.numpoints, 2)).T  # size, color
        i = 0
        while True:
            xy[i,0]=x[i];xy[i,1]=y[i]
            i = (i+1) % self.numpoints
            yield np.c_[xy[:,0], xy[:,1], s, c]

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        x, y, s, c = next(self.stream).T
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
        self.scat.set_sizes(300 * abs(data[:, 2])**1.5 + 100)
        # Set colors..
        self.scat.set_array(data[:, 3])

        # return the updated artist to FuncAnimation
        # It expects a sequence of artists, thus the trailing comma.
        return self.scat,
