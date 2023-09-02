from est_homography import est_homography
import numpy as np

def PnP(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-N-Point problem with collineation assumption, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3, ) numpy array describing camera translation in the world (t_wc)

    """

    ##### STUDENT CODE START #####

    # Homography Approach
    # Following slides: Pose from Projective Transformation
    Pw_1 = Pw[:,:2]
    H = est_homography(Pw_1, Pc)

    H = H / H[2][2]
    H_prime = np.dot(np.linalg.inv(K),H)
    H_1_prime = H_prime[:,0]
    H_2_prime = H_prime[:,1]
    H_3_prime = np.cross(H_1_prime, H_2_prime)
    H_updated = np.array([H_1_prime, H_2_prime, H_3_prime]).T #rotation matrix
    U, S, Vt = np.linalg.svd(H_updated)

    # new_fitted_h = U@S@Vt
    mul = U @ Vt
    r33 = np.linalg.det(mul)

    S = np.eye(3)
    S[2][2] = r33


    # R = np.array([[],[],[]])

    R_1 = U@(S@Vt)
    slice_3 = H_prime[:,2:] #h3
    slice_1 = H_prime[:,:1] # h1
    norm = np.linalg.norm(slice_1)
    T = slice_3/norm
    t = -np.transpose(R_1)@T
    t = t.reshape(3,) #reshaped to match the requirement
    R = np.linalg.inv(R_1)

    ##### STUDENT CODE END #####

    return R, t
