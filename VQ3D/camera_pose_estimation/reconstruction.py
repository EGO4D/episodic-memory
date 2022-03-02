import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix

from utils import Rotation2Quaternion
from utils import Quaternion2Rotation
from utils import skewsymm


def FindMissingReconstruction(X, track_i):
    """
    Find the points that will be newly added

    Parameters
    ----------
    X : ndarray of shape (F, 3)
        3D points
    track_i : ndarray of shape (F, 2)
        2D points of the newly registered image

    Returns
    -------
    new_point : ndarray of shape (F,)
        The indicator of new points that are valid for the new image and are
        not reconstructed yet
    """
    new_point = np.logical_and(X[:, 0] == -1, track_i[:, 0] != -1)
    return new_point


def Triangulation_nl(X, P1, P2, x1, x2):
    """
    Refine the triangulated points

    Parameters
    ----------
    X : ndarray of shape (n, 3)
        3D points
    P1 : ndarray of shape (3, 4)
        Camera projection matrix 1
    P2 : ndarray of shape (3, 4)
        Camera projection matrix 2
    x1 : ndarray of shape (n, 2)
        Point correspondences from pose 1
    x2 : ndarray of shape (n, 2)
        Point correspondences from pose 2

    Returns
    -------
    X_new : ndarray of shape (n, 3)
        The set of refined 3D points
    """
    R1 = P1[:, :3]
    C1 = -R1.T @ P1[:, 3]
    R2 = P2[:, :3]
    C2 = -R2.T @ P2[:, 3]

    p1 = np.concatenate([C1, Rotation2Quaternion(R1)])
    p2 = np.concatenate([C2, Rotation2Quaternion(R2)])

    lamb = 0.005
    n_iter = 10
    X_new = X.copy()
    for i in range(X.shape[0]):
        pt = X[i, :]
        for j in range(n_iter):
            proj1 = R1 @ (pt - C1)
            proj1 = proj1[:2] / proj1[2]
            proj2 = R2 @ (pt - C2)
            proj2 = proj2[:2] / proj2[2]

            dfdX1 = ComputePointJacobian(pt, p1)
            dfdX2 = ComputePointJacobian(pt, p2)

            H1 = dfdX1.T @ dfdX1 + lamb * np.eye(3)
            H2 = dfdX2.T @ dfdX2 + lamb * np.eye(3)

            J1 = dfdX1.T @ (x1[i, :] - proj1)
            J2 = dfdX2.T @ (x2[i, :] - proj2)

            delta_pt = np.linalg.inv(H1) @ J1 + np.linalg.inv(H2) @ J2
            pt += delta_pt

        X_new[i, :] = pt

    return X_new


def Triangulation_RANSAC(uv1, P, ransac_n_iter=30, threshold=1e-2):
    A = np.matmul(skewsymm(uv1), P)
    N = A.shape[0]

    best_inlier = -1
    best_C1_X = None
    best_inlier_ids = []
    best_outlier_ids = []
    for i in range(ransac_n_iter):
        rand_idx = np.random.choice(N, size=3, replace=False)
        A_select = A[rand_idx]
        A_select = np.reshape(A_select, (A_select.shape[0] * A_select.shape[1], A_select.shape[2]))
        u, s, vh = np.linalg.svd(A_select)
        if vh[3,3] == 0.0: continue
        C1_X = vh[3, :3] / vh[3, 3]

        inlier_num = 0
        inlier_ids = []
        outlier_ids = []
        for k in range(N):
            proj = P[k, :, :3] @ C1_X.reshape(3, 1) + P[k, :, 3:]
            proj = proj[:2] / proj[2]
            error = np.linalg.norm(proj.reshape(-1) - uv1[k, :2])
            if error < threshold:
                inlier_num += 1
                inlier_ids.append(k)
            else:
                outlier_ids.append(k)

        if inlier_num > best_inlier:
            best_inlier = inlier_num
            best_inlier_ids = inlier_ids
            best_outlier_ids = outlier_ids
            best_C1_X = C1_X

    return best_C1_X, best_inlier_ids, best_outlier_ids


def TriangulationD_RANSAC(Ci_f, P, ransac_n_iter=10, threshold=1e-2):
    N = P.shape[0]
    best_inlier = -1
    best_C1_X = None
    best_inlier_ids = []
    best_outlier_ids = []
    Ci_f = Ci_f.T
    Ci_uv = Ci_f[:2] / Ci_f[2]
    for i in range(N):
        C1_X = P[i, :, :3].T @ (Ci_f[:, i:(i + 1)] - P[i, :, 3:])

        inlier_num = 0
        inlier_ids = []
        outlier_ids = []

        for k in range(N):
            proj = P[k, :, :3] @ C1_X + P[k, :, 3:]
            proj = proj[:2] / proj[2]
            error = np.linalg.norm(proj - Ci_uv[:, k])
            if error < threshold:
                inlier_num += 1
                inlier_ids.append(k)
            else:
                outlier_ids.append(k)

        if inlier_num > best_inlier:
            best_inlier = inlier_num
            best_inlier_ids = inlier_ids
            best_outlier_ids = outlier_ids
            best_C1_X = C1_X

    return best_C1_X, best_inlier_ids, best_outlier_ids


def Triangulation_LS(uv1_inlier, P_inlier):
    A = np.matmul(skewsymm(uv1_inlier), P_inlier)
    A = np.reshape(A, (A.shape[0] * A.shape[1], A.shape[2]))
    u, s, vh = np.linalg.svd(A)
    C1_X = vh[3, :3] / vh[3, 3]

    return C1_X


def ComputePointJacobian(X, p):
    """
    Compute the point Jacobian

    Parameters
    ----------
    X : ndarray of shape (3,)
        3D point
    p : ndarray of shape (7,)
        Camera pose made of camera center and quaternion

    Returns
    -------
    dfdX : ndarray of shape (2, 3)
        The point Jacobian
    """
    R = Quaternion2Rotation(p[3:])
    C = p[:3]
    x = R @ (X - C)

    u = x[0]
    v = x[1]
    w = x[2]
    du_dc = R[0, :]
    dv_dc = R[1, :]
    dw_dc = R[2, :]

    dfdX = np.stack([
        (w * du_dc - u * dw_dc) / (w ** 2),
        (w * dv_dc - v * dw_dc) / (w ** 2)
    ], axis=0)

    return dfdX


def SetupBundleAdjustment(P, X, track, fixed_poses=None):
    """
    Setup bundle adjustment

    Parameters
    ----------
    P : ndarray of shape (K, 3, 4)
        Set of reconstructed camera poses
    X : ndarray of shape (J, 3)
        Set of reconstructed 3D points
    track : ndarray of shape (K, J, 2)
        Tracks for the reconstructed cameras

    Returns
    -------
    z : ndarray of shape (7K+3J,)
        The optimization variable that is made of all camera poses and 3D points
    b : ndarray of shape (2M,)
        The 2D points in track, where M is the number of 2D visible points
    S : ndarray of shape (2M, 7K+3J)
        The sparse indicator matrix that indicates the locations of Jacobian computation
    camera_index : ndarray of shape (M,)
        The index of camera for each measurement
    point_index : ndarray of shape (M,)
        The index of 3D point for each measurement
    """
    n_cameras = P.shape[0]
    n_points = X.shape[0]

    n_projs = np.sum(track[:, :, 0] != -1)
    b = np.zeros((2 * n_projs,))
    # S = np.zeros((2*n_projs, 7*n_cameras+3*n_points), dtype=bool)
    S_row = []
    S_col = []

    k = 0
    camera_index = []
    point_index = []
    optimized_poses = np.random.uniform(-1, 1, len(fixed_poses))
    for i in range(n_cameras):
        print('process camera ', i)
        for j in range(n_points):
            if track[i, j, 0] != -1 and track[i, j, 1] != -1:
                if not fixed_poses[i]:
                    # S[2*k : 2*(k+1), 7*i : 7*(i+1)] = 1
                    rows, cols = np.meshgrid(np.linspace(2 * k, 2 * (k + 1), endpoint=False, dtype=int),
                                             np.linspace(7 * i, 7 * (i + 1), endpoint=False, dtype=int))
                    rows = rows.reshape(-1)
                    cols = cols.reshape(-1)
                    S_row.append(rows)
                    S_col.append(cols)
                # else:
                #     if optimized_poses[i] > 0:
                #         rows, cols = np.meshgrid(np.linspace(2 * k, 2 * (k + 1), endpoint=False, dtype=int),
                #                                  np.linspace(7 * i, 7 * (i + 1), endpoint=False, dtype=int))
                #         rows = rows.reshape(-1)
                #         cols = cols.reshape(-1)
                #         S_row.append(rows)
                #         S_col.append(cols)

                # S[2*k : 2*(k+1), 7*n_cameras+3*j : 7*n_cameras+3*(j+1)] = 1
                rows, cols = np.meshgrid(np.linspace(2 * k, 2 * (k + 1), endpoint=False, dtype=int),
                                         np.linspace(7 * n_cameras + 3 * j, 7 * n_cameras + 3 * (j + 1), endpoint=False,
                                                     dtype=int))
                rows = rows.reshape(-1)
                cols = cols.reshape(-1)
                S_row.append(rows)
                S_col.append(cols)

                b[2 * k: 2 * (k + 1)] = track[i, j, :]
                camera_index.append(i)
                point_index.append(j)
                k += 1
    camera_index = np.asarray(camera_index)
    point_index = np.asarray(point_index)

    S_row = np.concatenate(S_row)
    S_col = np.concatenate(S_col)
    S_data = np.ones(S_row.shape[0], dtype=bool)
    S = csr_matrix((S_data, (S_row, S_col)), shape=(2 * n_projs, 7 * n_cameras + 3 * n_points))

    z = np.zeros((7 * n_cameras + 3 * n_points,))
    for i in range(n_cameras):
        R = P[i, :, :3]
        C = -R.T @ P[i, :, 3]
        q = Rotation2Quaternion(R)
        p = np.concatenate([C, q])
        z[7 * i: 7 * (i + 1)] = p
    for i in range(n_points):
        z[7 * n_cameras + 3 * i: 7 * n_cameras + 3 * (i + 1)] = X[i, :]

    return z, b, S, camera_index, point_index


def MeasureReprojection(z, b, n_cameras, n_points, camera_index, point_index):
    """
    Evaluate the reprojection error

    Parameters
    ----------
    z : ndarray of shape (7K+3J,)
        Optimization variable
    b : ndarray of shape (2M,)
        2D measured points
    n_cameras : int
        Number of cameras
    n_points : int
        Number of 3D points
    camera_index : ndarray of shape (M,)
        Index of camera for each measurement
    point_index : ndarray of shape (M,)
        Index of 3D point for each measurement

    Returns
    -------
    err : ndarray of shape (2M,)
        The reprojection error
    """
    n_projs = camera_index.shape[0]
    f = np.zeros((2 * n_projs,))
    for k, (i, j) in enumerate(zip(camera_index, point_index)):
        # Remove measurement error of fixed poses
        p = z[7 * i: 7 * (i + 1)]
        X = z[7 * n_cameras + 3 * j: 7 * n_cameras + 3 * (j + 1)]
        q = p[3:]
        q = q / np.linalg.norm(q)
        R = Quaternion2Rotation(q)
        C = p[:3]
        proj = R @ (X - C)
        proj = proj / proj[2]
        f[2 * k: 2 * (k + 1)] = proj[:2]
    err = b - f

    return err


def UpdatePosePoint(z0, z, n_cameras, n_points):
    """
    Update the poses and 3D points

    Parameters
    ----------
    z : ndarray of shape (7K+3J,)
        Optimization variable
    n_cameras : int
        Number of cameras
    n_points : int
        Number of 3D points

    Returns
    -------
    P_new : ndarray of shape (K, 3, 4)
        The set of refined camera poses
    X_new : ndarray of shape (J, 3)
        The set of refined 3D points
    """
    P_new = np.empty((n_cameras, 3, 4))
    for i in range(n_cameras):
        if i >= 2:
            p = z[7 * i: 7 * (i + 1)]
            q = p[3:]
        else:
            p = z0[7 * i: 7 * (i + 1)]
            q = p[3:]
        q = q / np.linalg.norm(q)
        R = Quaternion2Rotation(q)
        C = p[:3]
        P_new[i, :, :] = R @ np.hstack([np.eye(3), -C[:, np.newaxis]])

    X_new = np.empty((n_points, 3))
    for i in range(n_points):
        X_new[i, :] = z[7 * n_cameras + 3 * i: 7 * n_cameras + 3 * (i + 1)]

    return P_new, X_new


def RunBundleAdjustment(P, X, track, fixed_poses=None):
    """
    Run bundle adjustment

    Parameters
    ----------
    P : ndarray of shape (K, 3, 4)
        Set of reconstructed camera poses
    X : ndarray of shape (J, 3)
        Set of reconstructed 3D points
    track : ndarray of shape (K, J, 2)
        Tracks for the reconstructed cameras

    Returns
    -------
    P_new : ndarray of shape (K, 3, 4)
        The set of refined camera poses
    X_new : ndarray of shape (J, 3)
        The set of refined 3D points
    """
    n_cameras = P.shape[0]
    n_points = X.shape[0]

    z0, b, S, camera_index, point_index = SetupBundleAdjustment(P, X, track, fixed_poses)
    print('starting optimization')
    print('poses: ', P.shape[0])
    print('fixed poses: ', np.sum(fixed_poses))
    print('optimized poses: ', P.shape[0] - np.sum(fixed_poses))
    print('feature: ', track.shape[1])
    print('nnz: ', S.nnz)
    res = least_squares(
        lambda x: MeasureReprojection(x, b, n_cameras, n_points, camera_index, point_index),
        z0,
        jac_sparsity=S,
        verbose=2,
        ftol=1e-8,
        max_nfev=10,
        xtol=1e-15,
        loss='soft_l1',
        f_scale=0.2
    )
    # loss = 'soft_l1',
    # f_scale = 0.1
    z = res.x

    err0 = MeasureReprojection(z0, b, n_cameras, n_points, camera_index, point_index)
    err = MeasureReprojection(z, b, n_cameras, n_points, camera_index, point_index)
    print('Reprojection error {} -> {}'.format(np.linalg.norm(err0), np.linalg.norm(err)))

    P_new, X_new = UpdatePosePoint(z0, z, n_cameras, n_points)

    return P_new, X_new
