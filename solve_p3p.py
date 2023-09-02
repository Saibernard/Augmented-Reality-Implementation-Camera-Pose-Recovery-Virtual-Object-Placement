import numpy as np

def P3P(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-3-Point problem, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """

    ##### STUDENT CODE START #####

    # Invoke Procrustes function to find R, t
    # You may need to select the R and t that could transoform all 4 points correctly. 
    # R,t = Procrustes(Pc_3d, Pw[1:4])
    ##### STUDENT CODE END #####
    Pw = Pw[0:3]
    Pc = np.concatenate((Pc,np.ones((Pc.shape[0], 1))), axis=1)
    Pc_i = np.linalg.inv(K) @ Pc[0:3].T
    # f = int(np.mean(K[0][0], K[1][1]))  # Since k is a 3x3 matrix with all ones, f = 1, so using 1 directly
    # Pw_1 = Pw[0]
    # Pw_2 = Pw[1]
    # Pw_3 = Pw[2]
    # u_1, u_2, u_3 = Pc_i[0, :]
    # v_1, v_2, v_3 = Pc_i[1, :]
    u_1 = Pc_i[0][0]
    u_2 = Pc_i[0][1]
    u_3 = Pc_i[0][2]
    v_1 = Pc_i[1][0]
    v_2 = Pc_i[1][1]
    v_3 = Pc_i[1][2]
    q1 = np.array([u_1,v_1,1])
    q2 = np.array([u_2, v_2, 1])
    q3 = np.array([u_3, v_3, 1])

    a = np.linalg.norm(Pw[1] - Pw[2])
    b = np.linalg.norm(Pw[0] - Pw[2])
    c = np.linalg.norm(Pw[0] - Pw[1])

    # a = np.linalg.norm(Pw_2-Pw_3)
    # b = np.linalg.norm(Pw_1-Pw_3)
    # c = np.linalg.norm(Pw_1-Pw_2)

    # mat_1 = np.array([u_1],[v_1],[f])
    # mat_2 = np.array([u_2],[v_2],[f])
    # mat_3 = np.array([u_3],[v_3],[f])
    j1 = (1/np.linalg.norm(q1)) * q1       # norm is the sqrt of the distance
    j2 = (1/np.linalg.norm(q2)) * q2
    j3 = (1/np.linalg.norm(q3)) * q3
    # j1 = np.dot((1/ np.sqrt(pow(u_1,2) + pow(v_1,2) + pow(f,2))),q1)
    # j2 = np.dot((1/ np.sqrt(pow(u_2,2) + pow(v_2,2) + pow(f,2))),q2)
    # j3 = np.dot((1/ np.sqrt(pow(u_3,2) + pow(v_3,2) + pow(f,2))),q3)
    cos_alpha = np.dot(j2, j3.T)
    cos_beta = np.dot(j1, j3.T)
    cos_gamma = np.dot(j1, j2.T)
    # print(cos_a)
    # a_b_c_minus = ((pow(a,2) - pow(c,2))/(pow(b,2)))
    # a_b_c_plus = ((pow(a, 2) + pow(c, 2)) / (pow(b, 2)))

    # A4 = np.pow(((pow(a,2) - pow(c,2) / pow(b,2)) - 1),2) - ((4*pow(c,2)/pow(b,2))*pow(cos_alpha,2))
    # A3 = 4*((a_b_c_minus*(1-a_b_c_minus)*cos_beta) -
    #         ((1 - a_b_c_plus)*cos_alpha *cos_gamma) + (2*pow(c,2))/(pow(b,2))* pow(cos_alpha,2)*cos_beta)
    # A2 = 2 *(((np.square(a_b_c_minus)) - 1) + (2*(np.square(a_b_c_minus))*np.pow(cos_beta,2)) +
    #          (2*((pow(b,2) - pow(c,2))/(pow(b,2)))*(pow(cosine_alpha))) - (4*a_b_c_plus*cos_alpha*cos_beta*cos_gamma)
    #          + (2*((pow(b,2) - pow(a,2))/(pow(b,2))) * (pow(cos_gamma))))
    # A1 = 4*(-np.square(a_b_c_minus)*((1+a_b_c_minus)*cos_beta)+((2*pow(a,2)*(np.square(cos_gamma))*cos_beta)/(pow(b,2)))
    #         - ((1-(a_b_c_plus))*cos_alpha*cos_gamma))
    # A0 = (np.square(1+a_b_c_minus)-((4*pow(a,2))*np.square(cos_gamma)))

    A0 = ((1 + ((a ** 2 - c**2) / b**2))**2) - (4 * (a**2 / b**2) * (cos_gamma**2))
    A1 = 4 * ((-((a**2 - c**2) / b**2) * (1 + ((a**2 - c**2) / b**2)) * cos_beta) +
              (2 * (a**2 / b**2) * cos_gamma**2 * cos_beta) -
              ((1 - ((a**2 + c**2) / b**2)) * cos_alpha * cos_gamma))

    A2 = 2 * (((((a ** 2) - (c ** 2)) / (b ** 2)) ** 2) - 1 +
              2 * ((((a**2) - (c ** 2)) / (b ** 2)) ** 2) * (cos_beta ** 2) +
              2 * (((b ** 2) - (c ** 2)) / (b ** 2)) * (cos_alpha ** 2) -
              4 * (((a ** 2) + (c ** 2)) / (b ** 2)) * (cos_alpha) * (cos_beta) * (cos_gamma) +
              2 * (((b ** 2) - (a ** 2)) / (b ** 2)) * (cos_gamma ** 2))
    A3 = 4 * ((((a ** 2) - (c ** 2)) / (b ** 2)) * ((1 - (((a ** 2) - (c ** 2)) / (b ** 2))) * cos_beta) -
              ((1 - (((a ** 2) + (c ** 2)) / (b ** 2))) * cos_alpha * cos_gamma) +
              ((2 * (c ** 2) * (cos_alpha ** 2) * cos_beta) / (b ** 2)))
    A4 = (((((a ** 2) - (c**2)) / (b**2)) - 1)**2) - (((4*(c ** 2)) / (b**2)) * (cos_alpha ** 2))

    # beta = np.arccos(cos_beta)

    ## Poynomial equation
    # A4*pow(v,4) + A3*pow(v,3) + A2*pow(v,2) +A1*v +A0 = 0
    # Coef = [A4, A3, A2, A1, A0]
    # x = np.roots(Coef)
    roots = np.roots(np.array([A4, A3, A2, A1, A0]))
    print(roots)
    v = abs(roots[~np.iscomplex(roots)])
    print(v)

    # alpha =
    #
    # gamma =
    # eq_1 = (pow(a,2) - pow(c,2))/ pow(b,2)
    # eq_2 = (pow(a,2) - pow(c,2))/pow(b,2)
    # eq_3 = (pow(a,2) - pow(c,2)) / pow(b,2)
    # eq_4 = 2 *(cos_gamma - v*cos_alpha)

    # u = (((((-1 + eq_1)*pow(v,2)) - ((2 * eq_2)*np.cos_beta*v)) + 1 + (eq_3))/eq_4)

    u = ((((-1 + ((a ** 2 - c**2) / b**2)) *(v ** 2)) - ((2*(a ** 2 - c ** 2) / b ** 2) * cos_beta * v) + 1 +
         ((a ** 2 - c**2) / b ** 2)) / (2 * (cos_gamma - v * cos_alpha)))

    s1 = np.sqrt((c**2)/(1+(u**2)-(2*u*cos_gamma)))
    s2 = u * s1
    s3 = v * s1
    # print(s1)

    P1 = s1[0]*j1 # j1, j2 ,j3 respresents the division of camera coordinates and its norm
    P2 = s2[0]*j2
    P3 = s3[0]*j3
    Pc_3d = np.vstack((P1,P2,P3))
    # P1_1 = s1[1] * j1  # j1, j2 ,j3 respresents the division of camera coordinates and its norm
    # P2_1 = s2[1] * j2
    # P3_1 = s3[1] * j3
    # Pc_3d_2 = np.vstack((P1_1,P2_1,P3_1))
    # pc_1 = Pc[0:3]
    # x = np.sqrt((pc_1**2) - (Pc_3d**2))
    # y = np.sqrt((pc_1**2) - (Pc_3d_2**2))
    # print(x)
    # print(y)

    R, t = Procrustes(Pc_3d, Pw)
    # R1, t1 = Procrustes(Pc_3d_2, Pw)
    # print(R,t)
    # print(R1,t1)

    return R, t

def Procrustes(X, Y):
    """
    Solve Procrustes: Y = RX + t

    Input:
        X: Nx3 numpy array of N points in camera coordinate (returned by your P3P)
        Y: Nx3 numpy array of N points in world coordinate
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """
    ##### STUDENT CODE START #####

    # Following the slides

    centroid_x = np.mean(X, axis=0)
    centroid_y = np.mean(Y, axis=0)

    points_X = np.transpose(X - centroid_x)
    points_Y = np.transpose(Y - centroid_y)

    h = points_Y @ (np.transpose(points_X))
    [U, S, Vt] = np.linalg.svd(h)
    det_1 = np.linalg.det(U @ Vt)
    S_m = np.array([[1,0,0],[0,1,0],[0,0,det_1]])
    V = Vt.T
    U_T = U.T




    R = (U@S_m)@Vt
    t = centroid_y - (R@centroid_x)

    ##### STUDENT CODE END #####

    return R, t
