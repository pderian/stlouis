"""Configuration for the Wavefront experiment.

Written by P. DERIAN 2017-01-06.
"""
###
import numpy
###

### Path
ROOT_RAWDATA_DIR = ''
ROOT_PREPROC_DIR = ''
ROOT_ESTIM_DIR = ''
# this is the directory where ffmpeg & co are installed, since libav sucks ass.
FFMPEG_DIR = '/opt/local/bin'

### Parameters for the St Louis dataset
PARAMS_SAINTLOUIS = {
    # input data
    'label': 'StLouis', # a label for the frame directory
    'image_format': 'jpg', # format of output images
    # control points
    'cp_file': 'resources/DJI_SAL_20161211_GCP170611_GOOD.prn',
    # grid
    'resolution': 1., #[m/px]
    'origin': (337700., 1769100.), # origin (x,y) easting, northing of the domain in [m]
    'dimensions': (700., 1500.), # domain size in [m]
    'rotation': 0., # domain rotation in [degree] around origin
    }

### Helpers
def domain_grid(origin, dimensions, rotation, resolution):
    """Reference function for the generation of the domain grid from the origin, dimensions
    rotation and resolution parameters.

    Written by P. DERIAN 2017-01-11.
    """
    x = numpy.arange(0., dimensions[0], step=resolution,) # 1d
    y = numpy.arange(0., dimensions[1], step=resolution,) # 1d
    X, Y = numpy.meshgrid(x, y) # 2d
    cos_theta = numpy.cos(numpy.deg2rad(rotation))
    sin_theta = numpy.sin(numpy.deg2rad(rotation))
    Xr = cos_theta*X - sin_theta*Y + origin[0] #apply rotation
    Yr = sin_theta*X + cos_theta*Y + origin[1]
    return Xr, Yr
