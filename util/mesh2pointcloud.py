import torch
from torch.nn.functional import normalize

def compute_normals_with_knn(point_cloud, k=20):
    """
    Computes the point cloud normals using the k-nearest neighbors.

    Args:
        point_cloud (torch.Tensor): Input point cloud tensor of shape (B, N, 3).
        k (int): The number of nearest neighbors to consider.

    Returns:
        torch.Tensor: Computed point cloud normals tensor of shape (B, N, 3).
    """
    device = point_cloud.device
    B, N, _ = point_cloud.size()

    # Compute pairwise distances between points
    # shape: (B, N, N)
    dists = torch.cdist(point_cloud, point_cloud)

    # Get indices of k-nearest neighbors
    # shape: (B, N, k)
    knn_inds = dists.topk(k=k+1, largest=False, sorted=True)[1][:,:,1:]

    # Get nearest neighbor points
    # shape: (B, N, k, 3)
    nn_pts = torch.gather(point_cloud.unsqueeze(2).repeat(1,1,k+1,1), 1, knn_inds.unsqueeze(-1).repeat(1,1,1,3))

    # Compute pairwise differences between points
    # shape: (B, N, k, k, 3)
    diff = nn_pts.unsqueeze(3) - nn_pts.unsqueeze(2)

    # Compute the cross product between pairwise differences
    # shape: (B, N, k, k, 3)
    cross = torch.cross(diff[:, :, :, :, :3], diff[:, :, :, :, 3:], dim=-1)

    # Compute the normals by averaging the cross products
    # shape: (B, N, k, 3)
    normals = normalize(cross.sum(dim=3), dim=-1)

    # Weight the normals by the cosine of the angle between the normal and the vector to the query point
    # shape: (B, N, k)
    angle_weights = torch.sum(normals * diff[:, :, :, :, :3], dim=-1)
    angle_weights /= torch.norm(diff[:, :, :, :, :3], dim=-1)
    angle_weights = torch.nn.functional.relu(angle_weights)
    angle_weights /= torch.sum(angle_weights, dim=-1, keepdim=True)

    # Weight the normals by the cosine weights and sum them
    # shape: (B, N, 3)
    normals_weighted = torch.sum(normals * angle_weights.unsqueeze(-1), dim=2)

    # Normalize the resulting vectors
    # shape: (B, N, 3)
    normals_weighted = normalize(normals_weighted, dim=-1)

    return normals_weighted
