from typing import Iterable, Any
import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree

from ops.mesh.affine_transform import AffineTransform


def load_K_Rt(P: np.ndarray):

    dec = cv2.decomposeProjectionMatrix(P)
    K = dec[0]
    R = dec[1]
    t = dec[2]

    intrinsics = K / K[2, 2]

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


def transform_mesh(mesh: Iterable[Any], transform: np.ndarray):
    vertices = mesh[0].copy()
    vertices = AffineTransform(matrix=transform).transform(vertices)
    return [vertices, mesh[1]]


def perform_icp(
    mesh_source: Iterable[np.array],
    mesh_target: Iterable[np.array],
    mask_source: np.ndarray = None,
    mask_target: np.ndarray = None,
    **icp_args
) -> tuple:

    source_vertices = mesh_source[0].detach().numpy()
    target_vertices = mesh_target[0]
    points_source = source_vertices if mask_source is None else source_vertices[
        mask_source]
    points_target = target_vertices if mask_target is None else target_vertices[
        mask_target]
    transform, _, _ = trimesh.registration.icp(points_source, points_target, **icp_args)

    return transform_mesh([mesh_source[0].detach().numpy(), mesh_source[1].detach().numpy()], transform), transform


def unidirectional_chamfer_distance(source: np.ndarray, target: np.ndarray, p: int = 2):

    kdtree = cKDTree(target, leafsize=10)
    d, _ = kdtree.query(source, k=1, p=p)

    return d
