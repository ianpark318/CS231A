import numpy as np
import matplotlib.pyplot as plt
from p1 import *
from epipolar_utils import *
np.set_printoptions(precision=6, suppress=True)

'''
COMPUTE_EPIPOLE computes the epipole e in homogenous coordinates
given the fundamental matrix
Arguments:
    F - the Fundamental matrix solved for with normalized_eight_point_alg(points1, points2)

Returns:
    epipole - the homogenous coordinates [x y 1] of the epipole in the image
'''
def compute_epipole(F):
    # Return of F is e2 and return of F^T is e1
    # Fe' = 0 and e' = e1 so F have to be transposed.
    F = F.T
    _, _, Vt = np.linalg.svd(F, full_matrices=True)
    V = Vt.T
    e = V[:, -1]
    # e should be homogeneous coordinate
    # so we should normalize it
    e = e / e[-1]
    return e

'''
COMPUTE_H computes a homography to map an epipole to infinity along the horizontal axis 
Arguments:
    e - the epipole
    im2 - the image
Returns:
    H - homography matrix
'''
def compute_H(e, im):
    h, w = im.shape

    # Translation matrix
    T = np.eye(3)
    T[0, 2] = -(w / 2.)
    T[1, 2] = -(h / 2.)
    # Rotation matrix
    R = np.zeros((3, 3))
    R[2, 2] = 1.
    t_e = np.dot(T, e)
    if t_e[0] >= 0:
        alpha = 1
    else:
        alpha = -1
    R[0, 0] = alpha * t_e[0] / (t_e[0] ** 2 + t_e[1] ** 2) ** 0.5
    R[0, 1] = alpha * t_e[1] / (t_e[0] ** 2 + t_e[1] ** 2) ** 0.5
    R[1, 0] = - alpha * t_e[1] / (t_e[0] ** 2 + t_e[1] ** 2) ** 0.5
    R[1, 1] = alpha * t_e[0] / (t_e[0] ** 2 + t_e[1] ** 2) ** 0.5
    # Transformation G matrix
    G = np.eye(3)
    rt_e = np.dot(R, t_e)
    G[2, 0] = - (1. / rt_e[0])
    # H_2
    H_2 = np.dot(np.dot(np.dot(np.linalg.inv(T), G), R), T)
    return H_2

'''
COMPUTE_MATCHING_HOMOGRAPHIES determines homographies H1 and H2 such that they
rectify a pair of images
Arguments:
    e2 - the second epipole
    F - the Fundamental matrix
    im2 - the second image
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
Returns:
    H1 - the homography associated with the first image
    H2 - the homography associated with the second image
'''
def compute_matching_homographies(e2, F, im2, points1, points2):
    H_2 = compute_H(e2, im2)
    # So we have to find H_1 for image1.
    # now H_1 = H_A H_2 H_M

    # Cross product matrix [e]x
    ex = np.array([[0, -e2[2], e2[1]],
                   [e2[2], 0, -e2[0]],
                   [-e2[1], e2[0], 0]])

    # Then we need M
    v = np.array([1., 1., 1.])
    M = np.dot(ex, F) + np.outer(e2, v)

    p_hat = np.dot(np.dot(H_2, M), points1.T).T
    p_hat_prime = np.dot(H_2, points2.T).T

    # Normalize p_hat and p_prime_hat because it is homogeneous coordinates.
    p_hat = p_hat / p_hat[:, -1].reshape(-1, 1)
    p_hat_prime = p_hat_prime / p_hat_prime[:, -1].reshape(-1, 1)

    # Wa = b
    b = p_hat_prime[:, 0]
    W = np.ones(p_hat.shape)
    W[:, 0:2] = p_hat[:, 0:2]
    a = np.linalg.lstsq(W, b, rcond=None)[0]
    H_A = np.eye(3)
    H_A[0, :] = a.T

    H_1 = np.dot(np.dot(H_A, H_2), M)
    return H_1, H_2

if __name__ == '__main__':
    # Read in the data
    im_set = 'data/set1'
    im1 = imread(im_set+'/image1.jpg')
    im2 = imread(im_set+'/image2.jpg')
    points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
    points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
    assert (points1.shape == points2.shape)

    F = normalized_eight_point_alg(points1, points2)
    # F is such that (points2)^T * F * points1 = 0, so e1 is e' and e2 is e
    e1 = compute_epipole(F.T)
    e2 = compute_epipole(F)
    print("e1", e1)
    print("e2", e2)

    # Find the homographies needed to rectify the pair of images
    H1, H2 = compute_matching_homographies(e2, F, im2, points1, points2)
    print('')

    # Transforming the images by the homographies
    new_points1 = H1.dot(points1.T)
    new_points2 = H2.dot(points2.T)
    new_points1 /= new_points1[2,:]
    new_points2 /= new_points2[2,:]
    new_points1 = new_points1.T
    new_points2 = new_points2.T
    rectified_im1, offset1 = compute_rectified_image(im1, H1)
    rectified_im2, offset2 = compute_rectified_image(im2, H2)
    new_points1 -= offset1 + (0,)
    new_points2 -= offset2 + (0,)

    # Plotting the image
    F_new = normalized_eight_point_alg(new_points1, new_points2)
    plot_epipolar_lines_on_images(new_points1, new_points2, rectified_im1, rectified_im2, F_new)
    plt.show()
