""" Rectification of Wavefront video frames

Written by P. DERIAN 2016-2017
www.pierrederian.net
"""
###
import glob
import os
###
import numpy
import matplotlib.pyplot as pyplot
import matplotlib.patches as patches
import scipy.ndimage as ndimage
import scipy.io as scio
import skimage.io as skio
import skimage.transform as sktransform
###
from config import *
###

### IMAGE PREPROCESSOR ###
##########################
class DataPreprocessor:
    """
    """
    def __init__(self, H=None, cp_file=None, **kwargs):
        """
        Expected keywords:
            H: 3x3 homography (projection) matrix, or alternatively
            cp_file: file from which control points are loaded to estimate H.
            origin: (x0, y0) the origin in [meter]
            dimensions: (xdim, ydim) the domain size in [meter]
            rotation: the rotation angle of the grid around (x0, y0) in [degree]
            resolution: the grid resolution in [meter].

        Written by P. DERIAN 2016-01-23
        Modified by P. DERIAN 2017-01-11: added domain rotation
        Modified by P. DERIAN 2017-08-18: added optional control points for init
        """
        ### Store parameters
        if (H is None) and (cp_file is None):
            raise RuntimeError('Either H or cp_file must be provided')
        self.params = kwargs
        ### Load the projection matrix and create the corresponding transform
        if H is not None:
            self.cp_file = None
            self.control_points = None
            self.H = numpy.array(self.params['H'])
            self.projection = sktransform.ProjectiveTransform(self.H)
            print('DataPreprocessor: projection initialized from matrix H')
        ### or load control points and estimate projection
        elif cp_file is not None:
            self.cp_file = cp_file
            self.control_points = numpy.loadtxt(self.cp_file)
            cp_wYX = self.control_points[:,[1,0]] #world coords in [m]
            cp_iYX = self.control_points[:,[-1,-2]] #image coords in [px]
            self.projection = sktransform.ProjectiveTransform()
            self.projection.estimate(cp_iYX, cp_wYX)
            self.H = self.projection.params
            print('DataPreprocessor: projection initialized from control points')
        ### Real world interpolation coordinates
        self.X, self.Y = domain_grid(self.params['origin'], self.params['dimensions'],
                                     self.params['rotation'], self.params['resolution'])
        YX = numpy.hstack((self.Y.ravel().reshape((-1,1)),
                           self.X.ravel().reshape((-1,1)),
                           )) # interpolation coordinates
        self.shape = self.X.shape # this is the grid shape
        ### Image interpolation coordinates
        self.iYX = self.projection.inverse(YX)

    def __call__(self, args):
        """
        For use with multiprocessing.

        Written by P. DERIAN 2016-05-20
        """
        return self.process_image(*args)

    def process_image(self, image, as_uint=False):
        """
        Main processing pipe.

        Arguments:
            - image: a (M,N) (graysacle) or (M,N,3) (RGB) image;
            - as_uint=False: convert output to valid [0, 255] uint8 images.
        Return: img_rect
            - img_filt: the filtered version of img_rect.

        Written by P. DERIAN 2016-03-09
        Modified by P. DERIAN 2017-01-05: added clipping and int conversion.
        """
        # first rectify and grid
        img_rect = self.grid_image(image)
        if as_uint:
            numpy.clip(img_rect, 0., 1., img_rect)
            img_rect = (255.*img_rect).astype('uint8')
            if img_filt is not None:
                numpy.clip(img_filt, 0., 1., img_filt)
                img_filt = (255.*img_filt).astype('uint8')
        return img_rect

    def grid_image(self, image):
        """
        Grid the supplied image on the domain.

        Arguments:
            image a 2D grayscale or RBG image.
        Output:
            the 2D re-gridded image.

        Written by P. DERIAN 2016-01-23
        """
        # interpolate, reshape and return
        if image.ndim==2:
            return ndimage.interpolation.map_coordinates(image, self.iYX.T).reshape(self.shape)
        # TODO: make it faster???
        elif image.ndim==3:
            return numpy.dstack(
                (ndimage.interpolation.map_coordinates(image[:,:,i], self.iYX.T).reshape(self.shape)
                 for i in range(3)),
                )

    def is_out_of_image(self, im_shape, margin=0):
        """For a given input image shape, return the mask of rectified grid points
        located outside the image domain.

        Argument:
            - im_shape: the input image shape;
            - margin=0: an optional margin to be excluded as well.
        Output: a binary mask, True where outside image.

        Written by P. DERIAN 2017-08-18.
        """
        return ((self.iYX[:,0]<margin) | (self.iYX[:,0]>=im_shape[0]-margin) |
                (self.iYX[:,1]<margin) | (self.iYX[:,1]>=im_shape[1]-margin)
               ).reshape(self.shape)

    def coordinates_of_image(self, im_shape):
        """Compute the world coordinates of the pixels of a given image shape.

        :param im_shape: the shape in Numpy sense (rows, columns) of the considered image.
        :return: x_world, y_world, thow arrays (rows, columns) of world coordinates.
            i.e. image[i, j] has world coordinates (x[i,j], y[i,j]).

        Written by P. DERIAN 2018-04-07.
        """
        ny, nx = im_shape[:2]
        # grid of pixel coordinates
        x_px, y_px = numpy.meshgrid(numpy.arange(nx, dtype=float),
                                    numpy.arange(ny, dtype=float))
        # reshape for projection
        yx_px = numpy.concatenate((y_px.reshape((-1,1)), x_px.reshape((-1,1))), axis=-1)
        # transform pixel=>world
        yx_world = self.projection(yx_px)
        # reshape and return x_world, y_world
        return yx_world[:,1].reshape((ny, nx)), yx_world[:,0].reshape((ny, nx))

    def demo(self, im_file, out_image_file=None, out_matrix_file=None, out_coord_file=None):
        """
        Show the preprocessor output.

        Arguments:
            im_file: path to an image to be processed
        """
        print('\n*** {} Demo ***'.format(self.__class__.__name__))
        ### image
        print('Loading input image: {}'.format(im_file))
        im = pyplot.imread(im_file)
        rim = self.grid_image(im)

        ### grid domain
        dm_YX = numpy.vstack(
            (self.Y[[0, 0, -1, -1], [0, -1, -1, 0]],
             self.X[[0, 0, -1, -1], [0, -1, -1, 0]]
             ))
        dm_iYX = self.projection.inverse(dm_YX.T)

        ### print and save homography
        print('Homography matrix:\n{}'.format(self.H))
        if self.control_points is not None:
            print('Estimated from control points: {}'.format(self.cp_file))
        if out_matrix_file is not None:
            numpy.savetxt(out_matrix_file, self.H)
            print('Saved homography matrix: {}'.format(out_matrix_file))

        ### compute world coordinates of all image pixels
        world_X, world_Y = self.coordinates_of_image(im.shape)
        if out_coord_file is not None:
            scio.savemat(
                out_coord_file,
                {'im_shape': im.shape,
                 'Xutm': world_X,
                 'Yutm': world_Y,
                 'H': self.H,
                 'source': '{} by P. DERIAN - www.pierrederian.net'.format(os.path.basename(__file__)),
                 'descr': '"Xutm", "Yutm" are world coordinates (UTM) of the pixels of an image of shape "im_shape", i.e. image[i, j] has world coordinates (Xutm[i,j], Yutm[i,j]) with (i, j) zero-based indices. Coordinates were computed using homography matrix "H".'
                 })
            print('Saved world coordinate matrices: {}'.format(out_coord_file))

        ### plot
        dpi = 90.
        fig, axes = pyplot.subplots(1,2, figsize=(1921./dpi, 1080./dpi))
        # input frame
        ax = axes[0]
        ax.set_title('Input UAV frame')
        ax.imshow(im, cmap='gray', vmin=0, vmax=1)
        ax.add_artist(patches.Polygon(dm_iYX[:,::-1], fill=False, ec='k'))
        ax.set_xlim(0., im.shape[1])
        ax.set_ylim(im.shape[0], 0.)
        ax.set_xlabel('m (px)')
        ax.set_ylabel('n (px)')
        # rectified image
        ax = axes[1]
        ax.set_title('Rectified {} m/px'.format(self.params['resolution']))
        ax.imshow(rim, cmap='gray', vmin=0, vmax=1,
                  origin='bottom',
                  extent=[self.X[0,0], self.X[0,-1], self.Y[0,0], self.Y[-1,0]],
                  )
        ax.set_xlim(self.X[0,0], self.X[0,-1])
        ax.set_ylim(self.Y[0,0], self.Y[-1,0])
        ax.set_xlabel('{} x (m)'.format('Easting' if self.params['rotation']==0. else ''))
        ax.set_ylabel('{} y (m)'.format('Northing' if self.params['rotation']==0. else ''))
        ax.grid()
        # control points
        if self.control_points is not None:
            axes[0].plot(self.control_points[:,-2], self.control_points[:,-1], 'or')
            axes[1].plot(self.control_points[:,0], self.control_points[:,1], 'or',
                         label='Control points - {}'.format(os.path.basename(self.cp_file)))
            axes[1].legend(loc='center', frameon=False,
                           bbox_to_anchor=[0.3, 0.12], bbox_transform=fig.transFigure)
        # save / display
        pyplot.subplots_adjust(left=.05, bottom=.07, right=1.04, top=.92, wspace=0.)
        if out_image_file is not None:
            pyplot.savefig(out_image_file, dpi=dpi)
            print('Saved output figure: {}'.format(out_image_file))
        pyplot.show()


### MAIN ####
#############
if __name__=="__main__":

    ### run the demo
    params = PARAMS_SAINTLOUIS
    preproc = DataPreprocessor(**params)

    preproc.demo(
        "resources/{}_sample_frame.jpg".format(params['label']),
        out_image_file="resources/{}_config_demo.png".format(params['label']),
        out_matrix_file="resources/{}_homography_demo.txt".format(params['label']),
        out_coord_file="resources/{}_coordinates_demo.mat".format(params['label']),
        )

