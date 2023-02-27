# CS231A Homework 1, Problem 3
import numpy as np
from utils import mat2euler
import math
np.set_printoptions(precision=3, suppress=True)

'''
COMPUTE_VANISHING_POINTS
Arguments:
    points - a list of all the points where each row is (x, y). 
            It will contain four points: two for each parallel line.
Returns:
    vanishing_point - the pixel location of the vanishing point
'''
def compute_vanishing_point(points):
    # BEGIN YOUR CODE HERE
    points = points.astype(float)
    p1 = points[0, :]
    p2 = points[1, :]
    p3 = points[2, :]
    p4 = points[3, :]

    # y = ax + b in 2D
    a1 = (p2[1] - p1[1]) / (p2[0] - p1[0])
    a2 = (p4[1] - p3[1]) / (p4[0] - p3[0])
    b1 = p1[1] - a1 * p1[0]
    b2 = p3[1] - a2 * p3[0]

    # intersects for y = a1*x+b1 and y = a2*x+b2
    x = (b2 - b1) / (a1 - a2)
    y = (a1 * b2 - a2 * b1) / (a1 - a2)
    vanishing_point = np.array([x, y])
    return vanishing_point
    # END YOUR CODE HERE

'''
COMPUTE_K_FROM_VANISHING_POINTS
Makes sure to make it so the bottom right element of K is 1 at the end.
Arguments:
    vanishing_points - a list of vanishing points

Returns:
    K - the intrinsic camera matrix (3x3 matrix)
'''

def compute_K_from_vanishing_points(vanishing_points):
    # BEGIN YOUR CODE HERE
    # We know v1^T*W*v2=v2^T*W*v3=v3^T*W*v1=0
    # So we can make Ax=0 shape for x = [w1, w4, w5, w6]
    # and use SVD to A and get last column of V for x
    v1 = vanishing_points[0]
    v2 = vanishing_points[1]
    v3 = vanishing_points[2]

    # A has 3x4 shape
    A0 = np.array([v1[0] * v2[0] + v1[1] * v2[1], v1[0] + v2[0], v1[1] + v2[1], 1])
    A1 = np.array([v2[0] * v3[0] + v2[1] * v3[1], v2[0] + v3[0], v2[1] + v3[1], 1])
    A2 = np.array([v3[0] * v1[0] + v3[1] * v1[1], v3[0] + v1[0], v3[1] + v1[1], 1])
    A = np.vstack((A0, A1, A2))
    _, _, Vt = np.linalg.svd(A)
    V = Vt.T
    x = V[:, -1]
    W = np.array([[x[0],    0, x[1]],
                  [   0, x[0], x[2]],
                  [x[1], x[2], x[3]]])
    # W = (KK^T)^{-1}
    # W = (K^T)^{-1}*K^{-1}
    K_tran_inv = np.linalg.cholesky(W)
    K = np.linalg.inv(K_tran_inv.T)
    # make the (3,3) element 1
    K = K / K[-1, -1]
    return K
    # END YOUR CODE HERE

'''
COMPUTE_ANGLE_BETWEEN_PLANES
Arguments:
    vanishing_pair1 - a list of a pair of vanishing points computed from lines within the same plane
    vanishing_pair2 - a list of another pair of vanishing points from a different plane than vanishing_pair1
    K - the camera matrix used to take both images

Returns:
    angle - the angle in degrees between the planes which the vanishing point pair comes from2
'''
def compute_angle_between_planes(vanishing_pair1, vanishing_pair2, K):
    # BEGIN YOUR CODE HERE
    # We need vanishing point's homogeneous coordinates.
    v1_1 = np.hstack((vanishing_pair1[0], 1))
    v1_2 = np.hstack((vanishing_pair1[1], 1))
    v2_1 = np.hstack((vanishing_pair2[0], 1))
    v2_2 = np.hstack((vanishing_pair2[1], 1))

    # find line by vanishing point's cross product.
    l1_horiz = np.cross(v1_1, v1_2)
    l2_horiz = np.cross(v2_1, v2_2)

    # We need Omega inverse for calculating theta.
    W_inv = np.dot(K, K.T)

    # Calculat the angle
    cos_theta = np.dot(np.dot(l1_horiz.T, W_inv), l2_horiz) \
        / (np.sqrt(np.dot(np.dot(l1_horiz.T, W_inv), l1_horiz)) * np.sqrt(np.dot(np.dot(l2_horiz.T, W_inv), l2_horiz)))
    theta = np.arccos(cos_theta) / np.pi * 180

    return theta
    # END YOUR CODE HERE

'''
COMPUTE_ROTATION_MATRIX_BETWEEN_CAMERAS
Arguments:
    vanishing_points1 - a list of vanishing points in image 1
    vanishing_points2 - a list of vanishing points in image 2
    K - the camera matrix used to take both images

Returns:
    R - the rotation matrix between camera 1 and camera 2
'''

def compute_rotation_matrix_between_cameras(vanishing_points1, vanishing_points2, K):
    # BEGIN YOUR CODE HERE
    # Remind v = Kd thus d = K^{-1}v / ||K^{-1}v||
    v1  = np.hstack((vanishing_points1[0], 1))
    v2  = np.hstack((vanishing_points1[1], 1))
    v3  = np.hstack((vanishing_points1[2], 1))
    v1b = np.hstack((vanishing_points2[0], 1))
    v2b = np.hstack((vanishing_points2[1], 1))
    v3b = np.hstack((vanishing_points2[2], 1))
    K_inv = np.linalg.inv(K)
    d1 = np.dot(K_inv,  v1)
    d1 = d1 / np.linalg.norm(d1)
    d2 = np.dot(K_inv,  v2)
    d2 = d2 / np.linalg.norm(d2)
    d3 = np.dot(K_inv,  v3)
    d3 = d3 / np.linalg.norm(d3)
    d1b = np.dot(K_inv, v1b)
    d1b = d1b / np.linalg.norm(d1b)
    d2b = np.dot(K_inv, v2b)
    d2b = d2b / np.linalg.norm(d2b)
    d3b = np.dot(K_inv, v3b)
    d3b = d3b / np.linalg.norm(d3b)

    # make matrix D = [d1 d2 d3] and Db = [d1b d2b d3b]
    D = np.zeros((3, 3))
    D  = np.column_stack(( d1,  d2,  d3))
    Db = np.column_stack((d1b, d2b, d3b))
    # Db = RD thus R = Db*D^{-1}
    R = np.dot(Db, np.linalg.inv(D))
    return R
    # END YOUR CODE HERE

if __name__ == '__main__':
    # Part A: Compute vanishing points
    v1 = compute_vanishing_point(np.array([[1080, 598],[1840, 478],[1094,1340],[1774,1086]]))
    v2 = compute_vanishing_point(np.array([[674,1826],[4, 878],[2456,1060],[1940,866]]))
    v3 = compute_vanishing_point(np.array([[1094,1340],[1080,598],[1774,1086],[1840,478]]))

    v1b = compute_vanishing_point(np.array([[714,614],[1474,494],[750,1378],[1438,1094]]))
    v2b = compute_vanishing_point(np.array([[314,1912],[36,1578],[2060,1040],[1598,882]]))
    v3b = compute_vanishing_point(np.array([[750,1378],[714,614],[1438,1094],[1474,494]]))

    # Part B: Compute the camera matrix
    vanishing_points = [v1, v2, v3]
    print("Intrinsic Matrix:\n",compute_K_from_vanishing_points(vanishing_points))

    K_actual = np.array([[2448.0, 0, 1253.0],[0, 2438.0, 986.0],[0,0,1.0]])
    print()
    print("Actual Matrix:\n", K_actual)

    # Part E: Estimate the angle between the box and floor
    floor_vanishing1 = compute_vanishing_point(np.array([[674,1826],[2456,1060],[1094,1340],[1774,1086]]))
    floor_vanishing2 = compute_vanishing_point(np.array([[674,1826],[126,1056],[2456,1060],[1940,866]]))
    box_vanishing1 = v3
    box_vanishing2 = compute_vanishing_point(np.array([[1094,1340],[1774,1086],[1080,598],[1840,478]]))
    angle = compute_angle_between_planes([floor_vanishing1, floor_vanishing2], [box_vanishing1, box_vanishing2], K_actual)
    print()
    print("Angle between floor and box:", angle)

    # Part E: Compute the rotation matrix between the two cameras
    rotation_matrix = compute_rotation_matrix_between_cameras(np.array([v1, v2, v3]), np.array([v1b, v2b, v3b]), K_actual)
    print()
    print("Rotation between two cameras:\n", rotation_matrix)
    z, y, x = mat2euler(rotation_matrix)
    print()
    print("Angle around z-axis (pointing out of camera): %f degrees" % (z * 180 / math.pi))
    print("Angle around y-axis (pointing vertically): %f degrees" % (y * 180 / math.pi))
    print("Angle around x-axis (pointing horizontally): %f degrees" % (x * 180 / math.pi))
