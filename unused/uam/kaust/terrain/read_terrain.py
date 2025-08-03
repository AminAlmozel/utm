import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

# from mpl_toolkits.mplot3d import Axes3D

#https://e4ftl01.cr.usgs.gov/provisional/MEaSUREs/NASADEM/Africa/hgt_merge/
file_1 = 'N43E005.hgt'   #Source of file is from the link above

#https://cloud.sdsc.edu/v1/AUTH_opentopography/Raster/SRTM_GL1/SRTM_GL1_srtm/North/North_30_60/
file_2 = 'N34E036.hgt'   #Source of file is from the link above

file_3 = 'N21E039.hgt'

SAMPLES = 1201 # Change this to 1201 for SRTM3

def read_elevation_from_file(hgt_file, lon, lat):
    with open(hgt_file, 'rb') as hgt_data:
        # Each data is 16bit signed integer(i2) - big endian(>)
        elevations = np.fromfile(hgt_data, np.dtype('>i2'), SAMPLES*SAMPLES).reshape((SAMPLES, SAMPLES))
        lat_row = int(round((lat - int(lat)) * (SAMPLES - 1), 0))
        lon_row = int(round((lon - int(lon)) * (SAMPLES - 1), 0))
        print("interation")
    # return elevations[SAMPLES - 1 - lat_row, lon_row].astype(int)
    return elevations

def crush(elevation, threshold):
    mask0 = np.where(elevation < threshold)
    mask1 = np.where(elevation >= threshold)
    elevation[mask0] = 0
    elevation[mask1] = 1
    return elevation

#test example
lat = 34.0 + (3599.0/3600.0)
lon = 36.0 + (4.0/3600.0)

elv_1 = read_elevation_from_file(file_1,lon,lat)

# arcsecond square area in meters square
asa = 30 * 30
lon = 5
lat = 43
# n = 100
# elv_1 = elv_1[0:n, 0:n]
print(elv_1)  # this yields 423

# elv_1 = crush(elv_1, 400)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Make data.
X = np.linspace(-5,5,elv_1.shape[0])
Y = np.linspace(-5,5,elv_1.shape[1])
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.fliplr(elv_1)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=1, antialiased=True)

# Customize the z axis.
scale = 3
ax.set_zlim(0, scale * Z.max())
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.3, aspect=10)
# fig.subplots_adjust(bottom=0.0)
# fig.subplots_adjust(top=1.0)
fig.tight_layout()

plt.show()


# ax = fig.add_subplot(111, projection='3d')

# ax.plot_surface(X, Y, rv.pdf(pos), cmap="plasma")