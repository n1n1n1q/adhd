"""
Point cloud data structure
"""

from dataclasses import dataclass
from typing import Optional, Dict, List
import numpy as np
import re
from open3d import geometry
from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm

from recon3D.data.utils import visualize_pcds

debug_mode = False


@dataclass
class CloudWithViews:
    pcd: geometry.PointCloud
    views: list
    poses: list
    interesting_clouds: Optional[Dict] = None
    obj_frames: Optional[Dict] = None

    def __matmul__(self, B):
        """
        Multiply PCD's by a 4x4 matrix
        """
        self.pcd.transform(B)
        if self.interesting_clouds is not None:
            for i, interesting_cloud in self.interesting_clouds.items():
                self.interesting_clouds[i] = interesting_cloud.transform(B)
        return self


def compute_hausdorff_distance(
    pcd1: geometry.PointCloud, pcd2: geometry.PointCloud
) -> float:
    """
    Compute Hausdorff distance between two point clouds using SciPy's efficient implementation.

    Args:
        pcd1, pcd2: Open3D PointCloud objects

    Returns:
        Hausdorff distance as float
    """
    # Convert to numpy arrays
    points1 = np.array(pcd1.points)
    points2 = np.array(pcd2.points)

    if len(points1) == 0 or len(points2) == 0:
        return float("inf")

    # Compute directed Hausdorff distances in both directions
    dist1 = directed_hausdorff(points1, points2)[0]
    dist2 = directed_hausdorff(points2, points1)[0]

    # Hausdorff distance is the maximum of the two directed distances
    hausdorff = max(dist1, dist2)
    return hausdorff


def extract_base_name(object_name: str) -> str:
    """
    Extract base name from object name by removing trailing numbers.

    Args:
        object_name: Object name like "chair1", "table23", etc.

    Returns:
        Base name like "chair", "table", etc.
    """
    # Remove trailing digits
    return re.sub(r"\d+$", "", object_name)


def compare_objects_hausdorff(
    cloud1: CloudWithViews, cloud2: CloudWithViews, threshold: float = 1.0
) -> List:
    """
    Compare interesting objects between two CloudWithViews using Hausdorff distance.

    Args:
        cloud1, cloud2: CloudWithViews instances to compare
        threshold: Maximum Hausdorff distance to consider objects as "close"

    Returns:
        List of dictionaries with 'obj_name' and 'obj_frame' for objects in cloud1
        that don't have close matches in cloud2
    """
    missing_obj_frames = []

    if cloud1.interesting_clouds is None or cloud2.interesting_clouds is None:
        return missing_obj_frames

    if cloud1.obj_frames is None:
        return missing_obj_frames

    cloud2_by_base_name = {}
    for obj_name, obj_pcd in cloud2.interesting_clouds.items():
        base_name = extract_base_name(obj_name)
        if base_name not in cloud2_by_base_name:
            cloud2_by_base_name[base_name] = []
        cloud2_by_base_name[base_name].append(obj_pcd)

    for obj_name1, obj_pcd1 in tqdm(
        cloud1.interesting_clouds.items(), desc="Comparing objects"
    ):
        base_name1 = extract_base_name(obj_name1)

        if base_name1 not in cloud2_by_base_name:
            if obj_name1 in cloud1.obj_frames:
                missing_obj_frames.append(
                    {"obj_name": obj_name1, "obj_frame": cloud1.obj_frames[obj_name1]}
                )
            continue

        min_distance = float("inf")
        best_match_idx = -1
        for idx, obj_pcd2 in enumerate(cloud2_by_base_name[base_name1]):
            distance = compute_hausdorff_distance(obj_pcd1, obj_pcd2)
            print(
                f"Comparing {obj_name1} with object {idx} -> distance: {distance:.4f}"
            )
            if distance < min_distance:
                min_distance = distance
                best_match_idx = idx

        print(f"{obj_name1} -> min distance: {min_distance:.4f}")

        if min_distance <= threshold and best_match_idx != -1:

            if debug_mode:
                obj_pcd1.paint_uniform_color([1.0, 0.0, 0.0])
                cloud2_by_base_name[base_name1][best_match_idx].paint_uniform_color(
                    [0.0, 1.0, 0.0]
                )
                visualize_pcds(
                    cloud1.pcd,
                    obj_pcd1,
                    cloud2_by_base_name[base_name1][best_match_idx],
                    window_name=f"{obj_name1} vs {base_name1} match {min_distance}",
                )

            cloud2_by_base_name[base_name1].pop(best_match_idx)
            print(
                f"Matched {obj_name1} with object {best_match_idx}, removing from available matches"
            )
        else:
            print(
                "______________________________DEBUG______________________________",
                cloud1.obj_frames.keys(),
            )
            if obj_name1 in cloud1.obj_frames:
                missing_obj_frames.append(
                    {"obj_name": obj_name1, "obj_frame": cloud1.obj_frames[obj_name1]}
                )

    return missing_obj_frames
