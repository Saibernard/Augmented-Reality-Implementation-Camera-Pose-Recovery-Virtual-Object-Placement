import numpy as np

def est_pixel_world(pixels, R_wc, t_wc, K):
    """
    Estimate the world coordinates of a point given a set of pixel coordinates.
    The points are assumed to lie on the x-y plane in the world.
    Input:
        pixels: N x 2 coordiantes of pixels
        R_wc: (3, 3) Rotation of camera in world
        t_wc: (3, ) translation from world to camera
        K: 3 x 3 camara intrinsics
    Returns:
        Pw: N x 3 points, the world coordinates of pixels
    """

    ##### STUDENT CODE START #####
    pixels = np.concatenate((pixels,np.ones((pixels.shape[0],1))), axis = 1)
    R_cw = R_wc.T # rotation is the transpose between world and frame
    t_wc_reshaped = t_wc.reshape(-1,1) # reshaping to multiply
    R_T_mul = np.matmul(-R_cw,t_wc_reshaped)
    R_cw_sliced = R_cw[:,0:2]
    Rotational_matrix = np.concatenate((R_cw_sliced, R_T_mul),axis = 1)
    Homog_matrix = K @ Rotational_matrix
    H_inv = np.linalg.inv(Homog_matrix)
    Pw = (H_inv@pixels.T).T
    reshaped_pw = (Pw[:,-1].reshape(-1,1))
    Pw = Pw/reshaped_pw
    Pw[:, -1] = np.transpose(np.zeros((Pw.shape[0]))) # since z is zero, putting the last column elements to zero.
    ##### STUDENT CODE END #####
    return Pw
