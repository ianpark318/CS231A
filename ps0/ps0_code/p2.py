# CS231A Homework 0, Problem 2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

def part_a():
    # ===== Problem 3a =====
    # Read in the images, image1.jpg and image2.jpg, as color images.
    # Hint: use io.imread to read in the files

    img1, img2 = None, None

    # BEGIN YOUR CODE HERE
    img1 = io.imread('./image1.jpg')
    img2 = io.imread('./image2.jpg')
    # END YOUR CODE HERE

    return img1, img2

def normalize_img(img):
    # TODO implement this helper function for parts b and c
    rgbmax = np.max(img, axis=(0, 1))   # (3,) for RGB
    rgbmin = np.min(img, axis=(0, 1))   # (3,) for RGB
    # use Min-Max Normalization
    return (img - rgbmin) / (rgbmax - rgbmin)

def part_b(img1, img2):
    # ===== Problem 3b =====
    # Convert the images to double precision and rescale them
    # to stretch from minimum value 0 to maximum value 1.

    # BEGIN YOUR CODE HERE
    img1 = img1.astype('float64')
    img2 = img2.astype('float64')

    img1 = normalize_img(img1)
    img2 = normalize_img(img2)
    # END YOUR CODE HERE
    
    return img1, img2
    
def part_c(img1, img2):
    # ===== Problem 3c =====
    # Add the images together and re-normalize them
    # to have minimum value 0 and maximum value 1.
    # Display this image.
    sumImage = None
    
    # BEGIN YOUR CODE HERE
    sumImage = normalize_img(img1 + img2)
    # END YOUR CODE HERE
    plt.figure()
    plt.imshow(sumImage)
    return sumImage

def part_d(img1, img2):
    # ===== Problem 3d =====
    # Create a new image such that the left half of
    # the image is the left half of image1 and the
    # right half of the image is the right half of image2.
    newImage1 = None

    # BEGIN YOUR CODE HERE
    row = img1.shape[0]
    col = img1.shape[1]
    half = col // 2
    newImage1 = np.zeros((row, col, 3), dtype=np.float64)
    newImage1[:, :half, :] = img1[:, :half, :]
    newImage1[:, half:, :] = img2[:, half:, :]
    # END YOUR CODE HERE

    plt.figure()
    plt.imshow(newImage1)
    return newImage1

def part_e(img1, img2):    
    # ===== Problem 3e =====
    # Using a for loop, create a new image such that every odd
    # numbered row is the corresponding row from image1 and the
    # every even row is the corresponding row from image2.
    # Hint: Remember that indices start at 0 and not 1 in Python.
    newImage2 = None

    # BEGIN YOUR CODE HERE
    row = img1.shape[0]
    col = img1.shape[1]
    newImage2 = np.zeros((row, col, 3), dtype=np.float64)
    for i in range(0, row, 2):
        newImage2[i, ...] = img2[i, ...]
    for i in range(1, row, 2):
        newImage2[i, ...] = img1[i, ...]
    # END YOUR CODE HERE

    plt.figure()
    plt.imshow(newImage2)
    return newImage2

def part_f(img1, img2):     
    # ===== Problem 3f =====
    # Accomplish the same task as part e without using a for-loop.
    # The functions reshape and tile may be helpful here.
    newImage3 = None

    # BEGIN YOUR CODE HERE
    row = img1.shape[0]
    col = img1.shape[1]
    evenRowMask = np.concatenate((np.ones((1, col, 3)), np.zeros((1, col, 3))), axis=0)
    oddRowMask = np.concatenate((np.zeros((1, col, 3)), np.ones((1, col, 3))), axis=0)

    evenRowFilter = np.tile(oddRowMask, (row//2, 1, 1))
    oddRowFilter = np.tile(evenRowMask, (row//2, 1, 1))

    newImage3 = np.multiply(img1, evenRowFilter) + np.multiply(img2, oddRowFilter)
    # END YOUR CODE HERE

    plt.figure()
    plt.imshow(newImage3)
    return newImage3

def part_g(img):         
    # ===== Problem 3g =====
    # Convert the result from part f to a grayscale image.
    # Display the grayscale image.
    # Hint: use np.dot and the standard formula for converting RGB to grey
    # greyscale = R*0.299 + G*0.587 + B*0.114
    grayImage = None

    # BEGIN YOUR CODE HERE
    grayImage = img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114
    # END YOUR CODE HERE

    plt.figure()
    plt.imshow(grayImage, cmap = plt.get_cmap('gray'))
    return img

if __name__ == '__main__':
    img1, img2 = part_a()
    img1, img2 = part_b(img1, img2)
    sumImage = part_c(img1, img2)
    newImage1 = part_d(img1, img2)
    newImage2 = part_e(img1, img2)
    newImage3 = part_f(img1, img2)
    grayImage = part_g(newImage3)
    plt.show()