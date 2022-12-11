import numpy as np
import scipy


def compute_normals(V, F):

    # Sparse matrix that maps vertices to faces (and other way around)
    col_idx = np.repeat(np.arange(len(F)), 3)
    row_idx = F.reshape(-1)
    data = np.ones(len(col_idx), dtype=bool)
    vert2face = scipy.sparse.coo_matrix(
        (data, (row_idx, col_idx)),
        shape=(len(V), len(F)),
        dtype=data.dtype)

    # Compute face normals
    f0 = V[F[:, 0]]
    f1 = V[F[:, 1]]
    f2 = V[F[:, 2]]
    face_normals = np.cross(f1 - f0, f2 - f0)

    # For every vertex sum the normals of the faces its contained
    vertex_normals = vert2face.dot(face_normals)
    vertex_normals = vertex_normals / np.linalg.norm(vertex_normals, axis=1)[:, np.newaxis]

    return vertex_normals
