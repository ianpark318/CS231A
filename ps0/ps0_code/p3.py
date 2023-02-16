# CS231A Homework 0, Problem 3
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

def low_rank_approx(rank, u, s, v):
    pass # we recommend you implement this helper function for parts b and c

def part_a():
    # ===== Problem 4a =====
    # Take the singular value decomposition of the image.

    img1 = io.imread('image1.jpg', as_gray=True)
    u, s, v = None, None, None

    # BEGIN YOUR CODE HERE
    u, s, v = np.linalg.svd(img1, full_matrices=False)
    # 'full_matrices' parameter isn't important here.

    # END YOUR CODE HERE

    return u, s, v

def part_b(u, s, v):
    # ===== Problem 4b =====
    # Save and display the best rank 1 approximation 
    # of the (grayscale) image1.
    rank1approx = None

    # BEGIN YOUR CODE HERE
    u1 = u[:, 0:1]
    s1 = s[0]
    v1 = v[0:1, :]
    rank1approx = s1 * np.dot(u1, v1)
    # END YOUR CODE HERE

    plt.figure()
    plt.imshow(rank1approx, cmap = plt.get_cmap('gray'))
    return rank1approx
    
def part_c(u, s, v):
    # ===== Problem 4c =====
    # Save and display the best rank 20 approximation
    # of the (grayscale) image1.
    rank20approx = None

    # BEGIN YOUR CODE HERE
    u1_to_20 = u[:, :20]
    s1_to_20 = np.diag(s[:20])
    v1_to_20 = v[:20, :]
    rank20approx = np.dot(u1_to_20, np.dot(s1_to_20, v1_to_20))
    # END YOUR CODE HERE

    plt.figure()
    plt.imshow(rank20approx, cmap = plt.get_cmap('gray'))
    return rank20approx

if __name__ == '__main__':
    u, s, v = part_a()
    rank1approx = part_b(u, s, v)
    rank20approx = part_c(u, s, v)
    plt.show()