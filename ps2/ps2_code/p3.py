import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import scipy.io as sio
import matplotlib.gridspec as gridspec
from epipolar_utils import *
np.set_printoptions(precision=6, suppress=True)

'''
FACTORIZATION_METHOD The Tomasi and Kanade Factorization Method to determine
the 3D structure of the scene and the motion of the cameras.
Arguments:
    points_im1 - N points in the first image that match with points_im2
    points_im2 - N points in the second image that match with points_im1

    Both points_im1 and points_im2 are from the get_data_from_txt_file() method
Returns:
    structure - the structure matrix
    motion - the motion matrix
'''
def factorization_method(points_im1, points_im2):
    # First find average of x_ik
    x1_mean = np.mean(points_im1, axis=0)
    x2_mean = np.mean(points_im2, axis=0)

    # Calculate x_hat for each image
    x1_hat = (points_im1 - x1_mean)[:, 0:2].T
    x2_hat = (points_im2 - x2_mean)[:, 0:2].T

    # Build D matrix
    D = np.vstack((x1_hat, x2_hat))
    U, S, Vt = np.linalg.svd(D, full_matrices=True)

    # We need rank 3 sigma
    S_rank3 = np.diag(S[0:3])
    U_3 = U[:, 0:3]
    Vt_3 = Vt[0:3, :]

    # Tomasi and Kanade concluded that a robust choice
    # of the factorization is M=U*sqrt(Sigma) and S=sqrt(Sigma)*Vt
    M = np.dot(U_3, S_rank3 ** 0.5)
    S = np.dot(S_rank3 ** 0.5, Vt_3)
    # Remind that M and S can't be unique
    return S, M

if __name__ == '__main__':
    for im_set in ['data/set1', 'data/set1_subset']:
        # Read in the data
        im1 = imread(im_set+'/image1.jpg')
        im2 = imread(im_set+'/image2.jpg')
        points_im1 = get_data_from_txt_file(im_set + '/pt_2D_1.txt')
        points_im2 = get_data_from_txt_file(im_set + '/pt_2D_2.txt')
        points_3d = get_data_from_txt_file(im_set + '/pt_3D.txt')
        assert (points_im1.shape == points_im2.shape)

        # Run the Factorization Method
        structure, motion = factorization_method(points_im1, points_im2)

        # Plot the structure
        fig = plt.figure()
        ax = fig.add_subplot(121, projection = '3d')
        scatter_3D_axis_equal(structure[0,:], structure[1,:], structure[2,:], ax)
        ax.set_title('Factorization Method')
        ax = fig.add_subplot(122, projection = '3d')
        scatter_3D_axis_equal(points_3d[:,0], points_3d[:,1], points_3d[:,2], ax)
        ax.set_title('Ground Truth')

        plt.show()
