import sys
import numpy as np
import os
from scipy.optimize import least_squares
import math
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.io import imread
from sfm_utils import *
np.set_printoptions(precision=6, suppress=True)

'''
ESTIMATE_INITIAL_RT from the Essential Matrix, we can compute 4 initial
guesses of the relative RT between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
Returns:
    RT: A 4x3x4 tensor in which the 3x4 matrix RT[i,:,:] is one of the
        four possible transformations
'''
def estimate_initial_RT(E):
    # First, define Z, W matrix
    Z = np.array([[0., 1., 0.],
                  [-1., 0., 0.],
                  [0., 0., 0.]])
    W = np.array([[0., -1., 0.],
                  [1., 0., 0.],
                  [0., 0., 1.]])

    # Get U, s, Vt using SVD
    U, s, Vt = np.linalg.svd(E, full_matrices=True)

    # E = MQ where M = UZU^T and Q = UWV^T or UW^TV^T
    M = np.dot(np.dot(U, Z), U.T)
    Q1 = np.dot(np.dot(U, W), Vt)
    Q2 = np.dot(np.dot(U, W.T), Vt)
    R1 = np.linalg.det(Q1) * Q1
    R2 = np.linalg.det(Q2) * Q2

    # T is simply u3 or -u3 (last column of U)
    T1 = U[:, -1]
    T2 = -U[:, -1]

    RT = np.zeros((4, 3, 4))
    RT[0, ...] = np.column_stack((R1, T1))
    RT[1, ...] = np.column_stack((R1, T2))
    RT[2, ...] = np.column_stack((R2, T1))
    RT[3, ...] = np.column_stack((R2, T2))

    return RT

'''
LINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point is the best linear estimate
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''
def linear_estimate_3d_point(image_points, camera_matrices):
    N = len(image_points)
    A = np.zeros((2 * N, 4))
    for i in range(N):
        p = image_points[i]
        M = camera_matrices[i]
        A[i * 2, :] = p[0] * M[2] - M[0]
        A[i * 2 + 1, :] = p[1] * M[2] - M[1]

    _, _, Vt = np.linalg.svd(A, full_matrices=True)
    V = Vt.T
    P = V[:, -1]

    # Divide last value since it is homogeneous coordinates.
    P = P / P[-1]
    point_3D = P[:-1]
    return point_3D

'''
REPROJECTION_ERROR given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    error - the 2Mx1 reprojection error vector
'''
def reprojection_error(point_3d, image_points, camera_matrices):
    N = image_points.shape[0]
    # Reprojection error vector's dimension is (2 * N, 1)
    e = np.zeros((2 * N, 1))
    # For matrix-vector product, we need homogeneous coordinate (X, Y, Z, 1)
    P = np.hstack((point_3d, 1))
    error_set = []
    for i in range(N):
        p = image_points[i]
        M = camera_matrices[i]
        y = np.dot(M, P)
        p_prime = 1.0 / y[2] * y[0:2]
        e[2 * i:2 * i + 2, 0] = p_prime - p

    return e

'''
JACOBIAN given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    jacobian - the 2Mx3 Jacobian matrix
'''
def jacobian(point_3d, camera_matrices):
    P = np.hstack((point_3d, 1))
    N = camera_matrices.shape[0]
    # Jacobian matrix has (2N, 3) shape
    J = np.zeros((2 * N, 3))
    # e_i = M\hat{P_i}-p_i
    # So M\hat{P_i} will be (M1\hat{P_i}/M1\hat{P_3}, M2\hat{P_i}/M1\hat{P_3})
    # Thus partial derivation's result will be like below code.
    for i in range(N):
        M = camera_matrices[i]
        p = np.dot(M, P)
        J[2 * i, :] = (M[0, :-1] * p[2] - M[2, :-1] * p[0]) / p[2] ** 2
        J[2 * i + 1, :] = (M[1, :-1] * p[2] - M[2, :-1] * p[1]) / p[2] ** 2

    return J

'''
NONLINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point that iteratively updates the points
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''
def nonlinear_estimate_3d_point(image_points, camera_matrices):
    # We need a good initialization which we have from our linear estimate
    P = linear_estimate_3d_point(image_points, camera_matrices)

    for i in range(10):
        e = reprojection_error(P, image_points, camera_matrices)
        J = jacobian(P, camera_matrices)
        P = P - np.dot(np.dot(np.linalg.inv(np.dot(J.T, J)), J.T), e).ravel()
    return P

'''
ESTIMATE_RT_FROM_E from the Essential Matrix, we can compute  the relative RT 
between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
    image_points - N measured points in each of the M images (NxMx2 matrix)
    K - the intrinsic camera matrix
Returns:
    RT: The 3x4 matrix which gives the rotation and translation between the 
        two cameras
'''
def estimate_RT_from_E(E, image_points, K):
    def cam1tocam2(P, RT):
        # Remind that moving reference system is
        # R^{-1}(P+t)
        R = RT[:, 0:3]
        T = RT[:, -1]
        P = P + T
        P = np.dot(np.linalg.inv(R), P)
        return P

    RT = estimate_initial_RT(E)
    N = RT.shape[0]
    cnt = [0] * N
    # First, we have to compute M_1
    # M_1 = K [I 0]
    I0 = np.hstack((np.eye(3), np.zeros((3, 1))))
    M1 = np.dot(K, I0)
    camera_matrices = np.zeros((2, 3, 4))
    camera_matrices[0] = M1
    # Find the best R, T among 4 cases
    for i in range(N):
        M2 = np.dot(K, RT[i])
        camera_matrices[1] = M2
        for j in range(image_points.shape[0]):
            P_hat = nonlinear_estimate_3d_point(image_points[j], camera_matrices)
            P_1 = P_hat
            P_2 = cam1tocam2(P_1, RT[i])
            if P_1[2] > 0 and P_2[2] > 0:
                cnt[i] += 1

    maxIndex = np.argmax(cnt)
    correctRT = RT[maxIndex]
    return correctRT


if __name__ == '__main__':
    run_pipeline = True

    # Load the data
    image_data_dir = 'data/statue/'
    unit_test_camera_matrix = np.load('data/unit_test_camera_matrix.npy')
    unit_test_image_matches = np.load('data/unit_test_image_matches.npy')
    image_paths = [os.path.join(image_data_dir, 'images', x) for x in
        sorted(os.listdir('data/statue/images')) if '.jpg' in x]
    focal_length = 719.5459
    matches_subset = np.load(os.path.join(image_data_dir,
        'matches_subset.npy'), allow_pickle=True, encoding='latin1')[0,:]
    dense_matches = np.load(os.path.join(image_data_dir, 'dense_matches.npy'), 
                               allow_pickle=True, encoding='latin1')
    fundamental_matrices = np.load(os.path.join(image_data_dir,
        'fundamental_matrices.npy'), allow_pickle=True, encoding='latin1')[0,:]

    # Part A: Computing the 4 initial R,T transformations from Essential Matrix
    print('-' * 80)
    print("Part A: Check your matrices against the example R,T")
    print('-' * 80)
    K = np.eye(3)
    K[0,0] = K[1,1] = focal_length
    E = K.T.dot(fundamental_matrices[0]).dot(K)
    im0 = imread(image_paths[0])
    im_height, im_width, _ = im0.shape
    example_RT = np.array([[0.9736, -0.0988, -0.2056, 0.9994],
        [0.1019, 0.9948, 0.0045, -0.0089],
        [0.2041, -0.0254, 0.9786, 0.0331]])
    print("Example RT:\n", example_RT)
    estimated_RT = estimate_initial_RT(E)
    print('')
    print("Estimated RT:\n", estimated_RT)

    # Part B: Determining the best linear estimate of a 3D point
    print('-' * 80)
    print('Part B: Check that the difference from expected point ')
    print('is near zero')
    print('-' * 80)
    camera_matrices = np.zeros((2, 3, 4))
    camera_matrices[0, :, :] = K.dot(np.hstack((np.eye(3), np.zeros((3,1)))))
    camera_matrices[1, :, :] = K.dot(example_RT)
    unit_test_matches = matches_subset[0][:,0].reshape(2,2)
    estimated_3d_point = linear_estimate_3d_point(unit_test_matches.copy(),
        camera_matrices.copy())
    expected_3d_point = np.array([0.6774, -1.1029, 4.6621])
    print("Difference: ", np.fabs(estimated_3d_point - expected_3d_point).sum())

    # Part C: Calculating the reprojection error and its Jacobian
    print('-' * 80)
    print('Part C: Check that the difference from expected error/Jacobian ')
    print('is near zero')
    print('-' * 80)
    estimated_error = reprojection_error(
            expected_3d_point, unit_test_matches, camera_matrices)
    estimated_jacobian = jacobian(expected_3d_point, camera_matrices)
    expected_error = np.array((-0.0095458, -0.5171407,  0.0059307,  0.501631))
    print("Error Difference: ", np.fabs(estimated_error - expected_error).sum())
    expected_jacobian = np.array([[ 154.33943931, 0., -22.42541691],
         [0., 154.33943931, 36.51165089],
         [141.87950588, -14.27738422, -56.20341644],
         [21.9792766, 149.50628901, 32.23425643]])
    print("Jacobian Difference: ", np.fabs(estimated_jacobian
        - expected_jacobian).sum())

    # Part D: Determining the best nonlinear estimate of a 3D point
    print('-' * 80)
    print('Part D: Check that the reprojection error from nonlinear method')
    print('is lower than linear method')
    print('-' * 80)
    estimated_3d_point_linear = linear_estimate_3d_point(
        unit_test_image_matches.copy(), unit_test_camera_matrix.copy())
    estimated_3d_point_nonlinear = nonlinear_estimate_3d_point(
        unit_test_image_matches.copy(), unit_test_camera_matrix.copy())
    error_linear = reprojection_error(
        estimated_3d_point_linear, unit_test_image_matches,
        unit_test_camera_matrix)
    print("Linear method error:", np.linalg.norm(error_linear))
    error_nonlinear = reprojection_error(
        estimated_3d_point_nonlinear, unit_test_image_matches,
        unit_test_camera_matrix)
    print("Nonlinear method error:", np.linalg.norm(error_nonlinear))

    # Part E: Determining the correct R, T from Essential Matrix
    print('-' * 80)
    print("Part E: Check your matrix against the example R,T")
    print('-' * 80)
    estimated_RT = estimate_RT_from_E(E,
        np.expand_dims(unit_test_image_matches[:2,:], axis=0), K)
    print("Example RT:\n", example_RT)
    print('')
    print("Estimated RT:\n", estimated_RT)

    # Part F: Run the entire Structure from Motion pipeline
    if not run_pipeline:
        sys.exit()
    print('-' * 80)
    print('Part F: Run the entire SFM pipeline')
    print('-' * 80)
    frames = [0] * (len(image_paths) - 1)
    for i in range(len(image_paths)-1):
        frames[i] = Frame(matches_subset[i].T, focal_length,
                fundamental_matrices[i], im_width, im_height)
        bundle_adjustment(frames[i])
    merged_frame = merge_all_frames(frames)

    # Construct the dense matching
    camera_matrices = np.zeros((2,3,4))
    dense_structure = np.zeros((0,3))
    for i in range(len(frames)-1):
        matches = dense_matches[i]
        camera_matrices[0,:,:] = merged_frame.K.dot(
            merged_frame.motion[i,:,:])
        camera_matrices[1,:,:] = merged_frame.K.dot(
                merged_frame.motion[i+1,:,:])
        points_3d = np.zeros((matches.shape[1], 3))
        use_point = np.array([True]*matches.shape[1])
        for j in range(matches.shape[1]):
            points_3d[j,:] = nonlinear_estimate_3d_point(
                matches[:,j].reshape((2,2)), camera_matrices)
        dense_structure = np.vstack((dense_structure, points_3d[use_point,:]))

    fig = plt.figure(figsize=(10,10))
    ax = fig.gca(projection='3d')
    ax.scatter(dense_structure[:,0], dense_structure[:,1], dense_structure[:,2],
        c='k', depthshade=True, s=2)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(0, 10)
    ax.view_init(-100, 90)

    plt.show()
