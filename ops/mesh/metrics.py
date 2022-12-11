from typing import Iterable
import numpy as np
import torch
import trimesh

from ops.mesh import numeric
from ops.mesh.load_obj import load_obj


def align_icp(mesh_gt, mesh_pred):
    # Perform fine alignment using ICP
    _, t_icp = numeric.perform_icp(mesh_gt, mesh_pred)
    mesh_pred = numeric.transform_mesh(mesh_pred, np.linalg.inv(t_icp))  # align the prediction with the ground truth
    return mesh_pred


def measure_chamfer(
    mesh_pred: Iterable[torch.FloatTensor],
    mesh_gt_path: str,
    p: int = 2,
):
    """
    Evaluates a predicted mesh with respect the ground truth scene. 
    ICP is performed to finely align the meshes, and finally the chamfer
    distance is computed in both ways.
    Finally, if a region identifier is provided, the scene is evaluated in that
    specific region. By default it evaluates with the whole head.
    See the README and the examples for more information
    Args:
        mesh_gt,       (Iterable[torch.FloatTensor]): GT mesh
        mesh_pred      (Iterable[torch.FloatTensor]): Predicted mesh
        region_id      (str): Region identifier
    Returns:
        np.array: Nx3 array with the chamfer distance gt->pred for each groundtruth vertex
        np.array: Mx3 array with the chamfer distance pred->gt for eacu predicted vertex
        Mesh    : Ground truth mesh from H3DS
        Mesh    : Finely aligned predicted mesh
    """
    mesh_gt = load_obj(mesh_gt_path)

    # Perform fine alignment using ICP
    mesh_pred = align_icp(mesh_gt, mesh_pred)

    chamfer_gt_pred = numeric.unidirectional_chamfer_distance(
        mesh_gt[0], mesh_pred[0], p=p) * 100  # in cm
    chamfer_pred_gt = numeric.unidirectional_chamfer_distance(
        mesh_pred[0], mesh_gt[0], p=p) * 100  # in cm

    return chamfer_gt_pred, chamfer_pred_gt, mesh_gt, mesh_pred


def measure_v2s(mesh_pred, mesh_gt_path: str, device='cuda'):
    """
    Measure v2s distance (cm) between gt and predicted mesh.
    """
    gt = load_obj(mesh_gt_path)
    gt_v = gt[0].numpy().copy()
    gt_f = gt[1].numpy().copy()

    # Perform fine alignment using ICP
    mesh_pred = align_icp(gt, mesh_pred)
    mesh_v = mesh_pred[0].copy()

    if torch.cuda.is_available():
        from kaolin.ops.mesh import index_vertices_by_faces
        from kaolin.metrics.trianglemesh import point_to_mesh_distance

        face_vertices_gt = index_vertices_by_faces(
            torch.tensor(gt_v, device=device).unsqueeze(0),
            torch.tensor(gt_f, device=device))
        v2s = torch.sqrt(
            point_to_mesh_distance(
                torch.tensor(mesh_v, device=device).double().unsqueeze(0),
                face_vertices_gt.double()
            )[0]
        )
        v2s = v2s.detach().cpu().numpy()
    else:
        v2s = trimesh.proximity.closest_point(
            trimesh.Trimesh(gt_v, gt_f), mesh_v)[1]

    v2s *= 100  # cm
    return np.mean(v2s), v2s
