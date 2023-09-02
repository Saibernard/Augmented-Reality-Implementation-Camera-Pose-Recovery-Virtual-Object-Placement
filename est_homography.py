import numpy as np

def est_homography(X, Y):
    """
    Calculates the homography H of two planes such that Y ~ H*X
    If you want to use this function for hw5, you need to figure out
    what X and Y should be.
    Input:
        X: 4x2 matrix of (x,y) coordinates
        Y: 4x2 matrix of (x,y) coordinates
    Returns:
        H: 3x3 homogeneours transformation matrix s.t. Y ~ H*X

    """

    ##### STUDENT CODE START #####
    A = np.ones((8, 9))

    A[0] = [-X[0][0], -X[0][1], -1, 0, 0, 0, X[0][0] * Y[0][0], X[0][1] * Y[0][0], Y[0][0]]
    A[1] = [0, 0, 0, -X[0][0], -X[0][1], -1, X[0][0] * Y[0][1], X[0][1] * Y[0][1], Y[0][1]]
    A[2] = [-X[1][0], -X[1][1], -1, 0, 0, 0, X[1][0] * Y[1][0], X[1][1] * Y[1][0], Y[1][0]]
    A[3] = [0, 0, 0, -X[1][0], -X[1][1], -1, X[1][0] * Y[1][1], X[1][1] * Y[1][1], Y[1][1]]
    A[4] = [-X[2][0], -X[2][1], -1, 0, 0, 0, X[2][0] * Y[2][0], X[2][1] * Y[2][0], Y[2][0]]
    A[5] = [0, 0, 0, -X[2][0], -X[2][1], -1, X[2][0] * Y[2][1], X[2][1] * Y[2][1], Y[2][1]]
    A[6] = [-X[3][0], -X[3][1], -1, 0, 0, 0, X[3][0] * Y[3][0], X[3][1] * Y[3][0], Y[3][0]]
    A[7] = [0, 0, 0, -X[3][0], -X[3][1], -1, X[3][0] * Y[3][1], X[3][1] * Y[3][1], Y[3][1]]

    [U, S, Vt] = np.linalg.svd(A)
    V = np.transpose(Vt)
    x = V[:, 8:]
    H = x.reshape(3, 3)



    ##### STUDENT CODE END #####

    return H
