import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from epipolar_utils import *
np.set_printoptions(precision=6, suppress=True)

'''
LLS_EIGHT_POINT_ALG  computes the fundamental matrix from matching points using 
linear least squares eight point algorithm
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    F - the fundamental matrix such that (points2)^T * F * points1 = 0
Please see lecture notes and slides to see how the linear least squares eight
point algorithm works
'''
def lls_eight_point_alg(points1, points2):
    N = points1.shape[0]
    W = np.ones((N, 9))
    # make W matrix (N x 9)
    # we are finding F such that (points2)^T * F * points1 = 0
    # so W matrix's each row is (uu', u'v, u', uv', vv', v', u, v, 1)
    for i in range(N):
        u = points1[i, 0]; u_prime = points2[i, 0]
        v = points1[i, 1]; v_prime = points2[i, 1]
        W[i, 0] = u * u_prime
        W[i, 1] = u_prime * v
        W[i, 2] = u_prime
        W[i, 3] = u * v_prime
        W[i, 4] = v * v_prime
        W[i, 5] = v_prime
        W[i, 6] = u
        W[i, 7] = v

    _, _, Vt = np.linalg.svd(W)
    V = Vt.T
    # Last column of V would be F_hat (F11, F12, F13, F21, F22, F23, F31, F32, F33)
    F_hat = V[:, -1].reshape((3, 3))
    # Normalize for |F|=1 because V is orthogonal matrix (not orthonormal)
    F_hat = F_hat / np.linalg.norm(F_hat)

    # We found F_hat which may have full rank
    # But true fundamental matrix has rank 2
    # So we should look for a solution that is the best rank-2 approximation of F_hat.
    U, S, Vt = np.linalg.svd(F_hat)
    S_rank_2 = np.zeros((3, 3))
    S_rank_2[0, 0] = S[0]
    S_rank_2[1, 1] = S[1]
    F = np.dot(np.dot(U, S_rank_2), Vt)
    return F

'''
NORMALIZED_EIGHT_POINT_ALG  computes the fundamental matrix from matching points
using the normalized eight point algorithm
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    F - the fundamental matrix such that (points2)^T * F * points1 = 0
Please see lecture notes and slides to see how the normalized eight
point algorithm works
'''
def normalized_eight_point_alg(points1, points2):
    # calculate each centroids
    p_centroid = np.mean(points1, axis=0)
    p_prime_centroid = np.mean(points2, axis=0)
    # translate (the origin of the new coordinate system should be centroid)
    t_p = (points1 - p_centroid)[:, 0:-1]
    t_p_prime = (points2 - p_prime_centroid)[:, 0:-1]

    # scaling factor (2 / mean_distance)
    s = 2 / np.mean(np.sqrt(np.sum(t_p ** 2, axis=1)))
    s_prime = 2 / np.mean(np.sqrt(np.sum(t_p_prime ** 2, axis=1)))
    # So now we know scaling and translation matrix
    # Thus we can build transformation matrix T and T_prime
    # Translation first and then scaling occured
    # Watch out for the sequence of translation and scaling
    T = np.array([[s, 0, -(s * p_centroid[0])],
                  [0, s, -(s * p_centroid[1])],
                  [0, 0, 1]])
    T_prime = np.array([[s_prime, 0, -(s * p_centroid[0])],
                        [0, s_prime, -(s * p_centroid[1])],
                        [0, 0, 1]])
    q = np.dot(T, points1.T).T
    q_prime = np.dot(T_prime, points2.T).T

    # Find fundamental matrix with normalized points
    F_q = lls_eight_point_alg(q, q_prime)

    # De-normalize is (F = T^T * F_q * T_prime) if (points1)^T * F * points2 = 0
    # but now F is fundamental matrix such that (points2)^T * F * points1 = 0
    # So de-normalize equation should be F = T'^T * F_q * T
    F = np.dot(np.dot(T_prime.T, F_q), T)
    return F

'''
PLOT_EPIPOLAR_LINES_ON_IMAGES given a pair of images and corresponding points,
draws the epipolar lines on the images
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    im1 - a HxW(xC) matrix that contains pixel values from the first image 
    im2 - a HxW(xC) matrix that contains pixel values from the second image 
    F - the fundamental matrix such that (points2)^T * F * points1 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    Nothing; instead, plots the two images with the matching points and
    their corresponding epipolar lines. See Figure 1 within the problem set
    handout for an example
'''
def plot_epipolar_lines_on_images(points1, points2, im1, im2, F):

    def plot_epipolar_lines_on_image(points1, points2, im, F):
        im_height = im.shape[0]
        im_width = im.shape[1]
        lines = F.T.dot(points2.T) # These should be also Fp' not F^Tp'
        plt.imshow(im, cmap='gray')
        for line in lines.T:
            a,b,c = line
            xs = [1, im.shape[1]-1]
            ys = [(-c-a*x)/b for x in xs]
            plt.plot(xs, ys, 'r')
        for i in range(points1.shape[0]):
            x,y,_ = points1[i]
            plt.plot(x, y, '*b')
        plt.axis([0, im_width, im_height, 0])

    # We change the figsize because matplotlib has weird behavior when
    # plotting images of different sizes next to each other. This
    # fix should be changed to something more robust.
    new_figsize = (8 * (float(max(im1.shape[1], im2.shape[1])) / min(im1.shape[1], im2.shape[1]))**2 , 6)
    fig = plt.figure(figsize=new_figsize)
    plt.subplot(121)
    plot_epipolar_lines_on_image(points1, points2, im1, F)
    plt.axis('off')
    plt.subplot(122)
    plot_epipolar_lines_on_image(points2, points1, im2, F.T)
    plt.axis('off')

'''
COMPUTE_DISTANCE_TO_EPIPOLAR_LINES  computes the average distance of a set a 
points to their corresponding epipolar lines. Compute just the average distance
from points1 to their corresponding epipolar lines (which you get from points2).
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    F - the fundamental matrix such that (points2)^T * F * points1 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    average_distance - the average distance of each point to the epipolar line
'''
def compute_distance_to_epipolar_lines(points1, points2, F):
    # F - the fundamental matrix such that (points2)^T * F * points1 = 0
    # l = Fp' and l' = F^Tp
    # so in this problem, l = F*points1 and l'=F^T*points2
    l = np.dot(F.T, points2.T)

    # Distance between point and line is |ax0+by0+c| / sqrt(a^2+b^2)
    a = l[0, :]; b = l[1, :]; c = l[2, :]
    x = points1[:, 0]; y = points1[:, 1]
    dist = np.abs(a * x + b * y + c) / np.sqrt(a ** 2 + b ** 2)
    avg_dist = np.mean(dist)
    return avg_dist


if __name__ == '__main__':
    for im_set in ['data/set1', 'data/set2']:
        print('-'*80)
        print("Set:", im_set)
        print('-'*80)

        # Read in the data
        im1 = imread(im_set+'/image1.jpg')
        im2 = imread(im_set+'/image2.jpg')
        points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
        points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
        assert (points1.shape == points2.shape)

        # Running the linear least squares eight point algorithm
        F_lls = lls_eight_point_alg(points1, points2)
        print("Fundamental Matrix from LLS  8-point algorithm:\n", F_lls)
        print("Distance to lines in image 1 for LLS:", \
            compute_distance_to_epipolar_lines(points1, points2, F_lls))
        print("Distance to lines in image 2 for LLS:", \
            compute_distance_to_epipolar_lines(points2, points1, F_lls.T))

        # Running the normalized eight point algorithm
        F_normalized = normalized_eight_point_alg(points1, points2)

        pFp = [points2[i].dot(F_normalized.dot(points1[i]))
            for i in range(points1.shape[0])]
        print("p'^T F p =", np.abs(pFp).max())
        print("Fundamental Matrix from normalized 8-point algorithm:\n", \
            F_normalized)
        print("Distance to lines in image 1 for normalized:", \
            compute_distance_to_epipolar_lines(points1, points2, F_normalized))
        print("Distance to lines in image 2 for normalized:", \
            compute_distance_to_epipolar_lines(points2, points1, F_normalized.T))

        # Plotting the epipolar lines
        plot_epipolar_lines_on_images(points1, points2, im1, im2, F_lls)
        plot_epipolar_lines_on_images(points1, points2, im1, im2, F_normalized)

        plt.show()

