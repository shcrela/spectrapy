#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 18:13:46 2021

@author: dejan
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage import io, transform
from matplotlib.widgets import Slider


class ShowCollection(object):
    '''Helps visualize a collection of images.

    Parameters:
    ---------------
    image_pattern : str
        Can take asterixes as wildcards. For ex.: "./my_images/*.jpg" to select
        all the .jpg images from the folder "my_images"
    load_func : function
        The function to apply when loading the images
    first_frame : int
        The frame from which you want to stard your slideshow
    load_func_kwargs : dict
        The named arguments of the load function

    Outputs:
    ----------
        Interactive graph displaying the images one by one, whilst you can
        scroll trough the collection using the slider or the keyboard arrows

    Example:
    -----------
    >>> import numpy as np
    >>> from skimage import io, transform

    >>> def binarization_load(f, shape=(132,132)):
    >>>     im = io.imread(f, as_gray=True)
    >>>     return transform.resize(im, shape, anti_aliasing=True)

    >>> ss = ShowCollection(images, load_func=binarization_load, shape=(128,128))
    '''

    def __init__(self, image_pattern, load_func=io.imread, first_frame=0,
                 **load_func_kwargs):

        self.coll_all = io.ImageCollection(image_pattern, load_func=load_func,
                                           **load_func_kwargs)
        self.first_frame = first_frame
        self.nb_pixels = self.coll_all[0].size
        self.titles = np.arange(len(self.coll_all))
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(left=0.1, bottom=0.2)
        self.last_frame = len(self.coll_all)-1
        self.l = plt.imshow(self.coll_all[self.first_frame])
        self.ax.set_title(f"{self.titles[self.first_frame]}")

        self.axcolor = 'lightgoldenrodyellow'
        self.axframe = plt.axes([0.15, 0.1, 0.7, 0.03], facecolor=self.axcolor)

        self.sframe = Slider(self.axframe, 'Frame', self.first_frame,
                             self.last_frame, valinit=self.first_frame,
                             valfmt='%d', valstep=1)
        self.sframe.on_changed(self.update) # calls the update function when changing the slider position

        # Calling the press function on keypress event
        # (only arrow keys left and right work)
        self.fig.canvas.mpl_connect('key_press_event', self.press)

        plt.show()


    def update(self, val):
        '''This function is for using the slider to scroll through frames'''
        frame = int(self.sframe.val)
        img = self.coll_all[frame]
        self.l.set_data(img)
        self.ax.set_title(f"{self.titles[frame]}")
        self.fig.canvas.draw_idle()


    def press(self, event):
        '''This function is to use arrow keys left and right to scroll
        through frames one by one'''
        frame = int(self.sframe.val)
        if event.key == 'left' and frame > 0:
            new_frame = frame - 1
        elif event.key == 'right' and frame < len(coll_all)-1:
            new_frame = frame + 1
        else:
            new_frame = frame
        self.sframe.set_val(new_frame)
        img = self.coll_all[new_frame]
        self.l.set_data(img)
        self.ax.set_title(f"{self.titles[new_frame]}")
        self.fig.canvas.draw_idle()


#%%

class AllMaps(object):
    '''
    Allows one to rapidly visualize maps of Raman spectra.
    You can also choose to visualize the map and plot the
    corresponding component side by side if you set the
    "components" parameter.

    Parameters:
    ----------------
    map_spectra : 3D ndarray
        the spectra shaped as (n_lines, n_columns, n_wavenumbers)
    sigma : 1D ndarray
        an array of wavenumbers (len(sigma)=n_wavenumbers)
    components: 2D ndarray
        The most evident use-case would be to help visualize the decomposition
        results from PCA or NMF. In this case, the function will plot the
        component with the corresponding map visualization of the given
        components' presence in each of the points in the map.
        So, in this case, your map_spectra would be for example
        the matrix of components' contributions in each spectrum,
        while the "components" array will be your actual components.
        In this case you can ommit your sigma values or set them to
        something like np.arange(n_components)
    components_sigma: 1D ndarray
        in the case explained above, this would be the actual wavenumbers
    **kwargs: dict
        can only take 'title' as a key for the moment

    Returns:
    --------------
    The interactive visualization.
    (you can scroll through sigma values with a slider,
     or using left/right keyboard arrows)
    '''

    def __init__(self, map_spectra, sigma=None, components=None,
                 components_sigma=None, **kwargs):
        self.map_spectra = map_spectra
        if sigma is None:
            self.sigma = np.arange(map_spectra.shape[-1])
        else:
            assert map_spectra.shape[-1] == len(sigma), "Check your Ramans shifts array"
            self.sigma = sigma
        self.first_frame = 0
        self.last_frame = len(self.sigma)-1
        if components is not None:
            #assert len(components) == map_spectra.shape[-1], "Check your components"
            self.components = components
            if components_sigma is None:
                self.components_sigma = np.arange(components.shape[-1])
            else:
                self.components_sigma = components_sigma
        else:
            self.components = None
        if components is not None:
            self.fig, (self.ax2, self.ax, self.cbax) = plt.subplots(ncols=3, gridspec_kw={'width_ratios':[40,40,1]})
            self.cbax.set_box_aspect(40*self.map_spectra.shape[0]/self.map_spectra.shape[1])
        else:
            self.fig, (self.ax, self.cbax) = plt.subplots(ncols=2, gridspec_kw={'width_ratios':[40,1]})
            self.cbax.set_box_aspect(40*self.map_spectra.shape[0]/self.map_spectra.shape[1])
            #self.cbax = self.fig.add_axes([0.92, 0.3, 0.03, 0.48])
        # Create some space for the slider:
        self.fig.subplots_adjust(bottom=0.19, right=0.89)
        self.title = kwargs.get('title', None)

        self.im = self.ax.imshow(self.map_spectra[:,:,0])
        self.im.set_clim(np.percentile(self.map_spectra[:,:,0], [1,99]))
        if self.components is not None:
            self.line, = self.ax2.plot(self.components_sigma, self.components[0])
            self.ax2.set_box_aspect(self.map_spectra.shape[0]/self.map_spectra.shape[1])
            self.ax2.set_title(f"Component {0}")
        self.titled(0)
        self.axcolor = 'lightgoldenrodyellow'
        self.axframe = self.fig.add_axes([0.15, 0.1, 0.7, 0.03], facecolor=self.axcolor)


        self.sframe = Slider(self.axframe, 'Frame',
                             self.first_frame, self.last_frame,
                             valinit=self.first_frame, valfmt='%d', valstep=1)



        self.my_cbar = mpl.colorbar.colorbar_factory(self.cbax, self.im)

        self.sframe.on_changed(self.update) # calls the "update" function when changing the slider position
        # Calling the "press" function on keypress event
        # (only arrow keys left and right work)
        self.fig.canvas.mpl_connect('key_press_event', self.press)
        plt.show()

    def titled(self, frame):
        if self.components is None:
            if self.title is None:
                self.ax.set_title(f"Raman shift = {self.sigma[frame]:.1f}cm⁻¹")
            else:
                self.ax.set_title(f"{self.title} n°{frame}")
        else:
            self.ax2.set_title(f"Component {frame}")
            if self.title is None:
                self.ax.set_title(f"Component n°{frame} contribution")
            else:
                self.ax.set_title(f"{self.title} n°{frame}")

    def update(self, val):
        '''This function is for using the slider to scroll through frames'''
        frame = int(self.sframe.val)
        img = self.map_spectra[:,:,frame]
        self.im.set_data(img)
        self.im.set_clim(np.percentile(img, [1,99]))
        if self.components is not None:
            self.line.set_ydata(self.components[frame])
            self.ax2.relim()
            self.ax2.autoscale_view()
        self.titled(frame)
        self.fig.canvas.draw_idle()

    def press(self, event):
        '''This function is to use arrow keys left and right to scroll
        through frames one by one'''
        frame = int(self.sframe.val)
        if event.key == 'left' and frame > 0:
            new_frame = frame - 1
        elif event.key == 'right' and frame < len(self.sigma)-1:
            new_frame = frame + 1
        else:
            new_frame = frame
        self.sframe.set_val(new_frame)
        img = self.map_spectra[:,:,new_frame]
        self.im.set_data(img)
        self.im.set_clim(np.percentile(img, [1,99]))
        self.titled(new_frame)
        if self.components is not None:
            self.line.set_ydata(self.components[new_frame])
            self.ax2.relim()
            self.ax2.autoscale_view()
        self.fig.canvas.draw_idle()


# %%

class NavigationButtons(object):
    '''This class allows you to visualize multispectral data and
    navigate trough your spectra simply by clicking on the
    navigation buttons on the graph.

    Parameters:
    -----------------
        sigma: 1D ndarray
            1D numpy array of your x-values (raman shifts, par ex.)
        spectra: 2D or 3D ndarray
            3D or 2D ndarray of shape (n_spectra, len(sigma), n_curves).
            The last dimension may be ommited it there is only one curve
            to be plotted for each spectra).
        autoscale: bool
            determining if you want to adjust the scale to each spectrum
        title: str
            The initial title describing where the spectra comes from
        label: list
            A list explaining each of the curves. len(label) = n_curves
    Output:
    ----------------
        matplotlib graph with navigation buttons to cycle through spectra

    Example:
    ---------------
        Let's say you have a ndarray containing 10 spectra,
        and let's suppose each of those spectra contains 500 points.

        >>> my_spectra.shape
        (10, 500)
        >>> sigma.shape
        (500, )

        Then let's say you show the results of baseline substraction.

        >>> my_baseline[i] = baseline_function(my_spectra[i])
        >>> # your baseline should have the same shape as your initial spectra.
        >>> multiple_curves_to_plot = np.stack(
                (my_spectra, my_baseline, my_spectra - my_baseline), axis=-1)
        >>> NavigationButtons(sigma, multiple_curves_to_plot)
    '''
    ind = 0

    def __init__(self, sigma, spectra, autoscale_y=False, title='Spectrum', label=False,
                 **kwargs):
        self.y_autoscale = autoscale_y

        if len(spectra.shape) == 2:
            self.s = spectra[:,:, np.newaxis]
        elif len(spectra.shape) == 3:
            self.s = spectra
        else:
            raise ValueError("Check the shape of your spectra.\n"
                             "It should be (n_spectra, n_points, n_curves)\n"
                             "(this last dimension might be ommited if it's equal to one)")
        self.n_spectra = self.s.shape[0]
        if isinstance(title, list) or isinstance(title, np.ndarray):
            if len(title) == spectra.shape[0]:
                self.title = title
            else:
                raise ValueError(f"you have {len(title)} titles,\n"
                                f"but you have {len(spectra)} spectra")
        else:
            self.title = [title]*self.n_spectra

        self.sigma = sigma
        if label:
            if len(label)==self.s.shape[2]:
                self.label = label
            else:
                warn("You should check the length of your label list.\nFalling on to default labels...")
                self.label = ["Curve n°"+str(numb) for numb in range(self.s.shape[2])]
        else:
            self.label = ["Curve n°"+str(numb) for numb in range(self.s.shape[2])]

        self.figr, self.axr = plt.subplots(**kwargs)
        self.axr.set_title(f'{title[0]}')
        self.figr.subplots_adjust(bottom=0.2)
        self.l = self.axr.plot(self.sigma, self.s[0], lw=2, alpha=0.7) # l potentially contains multiple lines
        self.axr.legend(self.l, self.label)
        self.axprev1000 = plt.axes([0.097, 0.05, 0.1, 0.04])
        self.axprev100 = plt.axes([0.198, 0.05, 0.1, 0.04])
        self.axprev10 = plt.axes([0.299, 0.05, 0.1, 0.04])
        self.axprev1 = plt.axes([0.4, 0.05, 0.1, 0.04])
        self.axnext1 = plt.axes([0.501, 0.05, 0.1, 0.04])
        self.axnext10 = plt.axes([0.602, 0.05, 0.1, 0.04])
        self.axnext100 = plt.axes([0.703, 0.05, 0.1, 0.04])
        self.axnext1000 = plt.axes([0.804, 0.05, 0.1, 0.04])

        self.bprev1000 = Button(self.axprev1000, 'Prev.1000')
        self.bprev1000.on_clicked(self.prev1000)
        self.bprev100 = Button(self.axprev100, 'Prev.100')
        self.bprev100.on_clicked(self.prev100)
        self.bprev10 = Button(self.axprev10, 'Prev.10')
        self.bprev10.on_clicked(self.prev10)
        self.bprev = Button(self.axprev1, 'Prev.1')
        self.bprev.on_clicked(self.prev1)
        self.bnext = Button(self.axnext1, 'Next1')
        self.bnext.on_clicked(self.next1)
        self.bnext10 = Button(self.axnext10, 'Next10')
        self.bnext10.on_clicked(self.next10)
        self.bnext100 = Button(self.axnext100, 'Next100')
        self.bnext100.on_clicked(self.next100)
        self.bnext1000 = Button(self.axnext1000, 'Next1000')
        self.bnext1000.on_clicked(self.next1000)

    def update_data(self):
        _i = self.ind % self.n_spectra
        for ll in range(len(self.l)):
            yl = self.s[_i][:, ll]
            self.l[ll].set_ydata(yl)
        self.axr.relim()
        self.axr.autoscale_view(None, False, self.y_autoscale)
        self.axr.set_title(f'{self.title[_i]}; N°{_i}')
        self.figr.canvas.draw()
        self.figr.canvas.flush_events()

    def next1(self, event):
        self.ind += 1
        self.update_data()

    def next10(self, event):
        self.ind += 10
        self.update_data()

    def next100(self, event):
        self.ind += 100
        self.update_data()

    def next1000(self, event):
        self.ind += 1000
        self.update_data()

    def prev1(self, event):
        self.ind -= 1
        self.update_data()

    def prev10(self, event):
        self.ind -= 10
        self.update_data()

    def prev100(self, event):
        self.ind -= 100
        self.update_data()

    def prev1000(self, event):
        self.ind -= 1000
        self.update_data()
