import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
import numpy as np

def globe_plot(latitude, longitude, temperature, view=(100., 0.), flat=False,
               vmin=None, vmax=None, cmap='plasma', smooth_factor=1.0,
               save_fig=''):
    """Plots temperature on a globe.

    Args:
        latitude (array-like): Array of shape (m,) containing latitude values.
        longitude (array-like): Array of shape (n,) containing longitude values.
        temperature (array-like): Array of shape (m, n) containing temperatures.
        view (tuple of floats): Longitude and latitude values for the view of
            the globe.
        flat (bool): If True, creates flat map plot, otherwise plots globe.
        smooth_factor (float): If not 1, smooths the data by sampling it
            smooth_factor times more finely and interpolating.
        save_fig (string): If non-empty, saves image at location given by
            save_fig.
    """
    # Set map projection
    if flat:
        ax = plt.axes(projection=ccrs.PlateCarree())
    else:
        ax = plt.axes(projection=ccrs.Orthographic(*view))

    # Add coastlines
    ax.coastlines()

    # Plot climate data
    if smooth_factor == 1.0:
        mesh = ax.pcolormesh(longitude, latitude, temperature,
                             transform=ccrs.PlateCarree(), cmap=cmap)
    else:
        interp_func = interp2d(longitude, latitude, temperature, kind='cubic')
        num_lons = int(smooth_factor * len(longitude))
        lon_new = np.linspace(0, 360 - (360 / num_lons), num_lons)
        lat_new = np.linspace(latitude[0], latitude[-1], int(smooth_factor * len(latitude)))
        # Not really sure why we need the minus for latitude here, but it just seems to work...
        temp_new = interp_func(lon_new, -lat_new)
        mesh = ax.pcolormesh(lon_new, -lat_new, temp_new, transform=ccrs.PlateCarree(),
                             cmap=cmap)

    if len(save_fig):
        plt.savefig(save_fig, format='png', dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()
    else:
        plt.show()


def globe_plot_from_data_tensor(data, view=(100., 0.), flat=False, vmin=None,
                                vmax=None, cmap='plasma', smooth_factor=1.0,
                                save_fig=''):
    """Helper function to plot directly from data tensor.

    Args:
        data (torch.Tensor): Shape (3, num_lats, num_lons).
    """
    latitude = data[0, :, 0]
    longitude = data[1, 0, :]
    temperature = data[2]
    globe_plot(latitude, longitude, temperature, view, flat, vmin, vmax, cmap,
               smooth_factor, save_fig)



def globe_plot_from_data_batch(batch, threshold=0.5, ncol=None, view=None, globe=False,smooth_factor=1.0, save_fig='',**kwargs):

        x = batch.numpy()
        batch_size = x.shape[0]

        if ncol == None:
            ncol = int(np.ceil(np.sqrt(batch_size)))

        nrow = 1 if batch_size == 2 else ncol
        if globe:
            view = (100., 0.)
            projection = ccrs.Orthographic(*view)
        else:
            projection = ccrs.PlateCarree()

        N = batch_size

        fig, axes = plt.subplots(nrow, ncol,
                                 figsize=(4 * nrow, 2 * ncol),
                                 subplot_kw={'projection': projection}
                                 )

        # fig, axes = plt.subplots(ncols=ncol, nrows=ceil(N/ncol), layout='constrained',
        #                         figsize=(3.5 * 4, 3.5 * ceil(N/ncol)),
        #                          subplot_kw={'projection': projection}
        #                          )

        idx = 0
        for i in range(nrow):
            for j in range(ncol):
                if nrow == 1 and ncol == 1:
                    ax_ij = axes
                elif nrow == 1:
                    ax_ij = axes[j]
                else:
                    ax_ij = axes[i, j]

                x_i = x[idx]
                latitude = x_i[0, :, 0]
                longitude = x_i[1, 0, :]
                temperature = x_i[2]


                mesh = ax_ij.pcolormesh(longitude, latitude, temperature,
                                        transform=ccrs.PlateCarree(),
                                        cmap='plasma')
                # Add coastlines
                ax_ij.coastlines()                                        
                # x_i = np.transpose( x[idx], (1, 2, 0))

                # ax_ij.set_xticks([])
                # ax_ij.set_yticks([])
                # ax_ij.set_axis_off()
                idx += 1
                if idx == batch_size: break
            if idx == batch_size: break
        
        plt.tight_layout()

        
        if len(save_fig):
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.savefig(save_fig)
            plt.clf()
            plt.close('all')
        else:
            plt.show()
