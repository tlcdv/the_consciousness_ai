import torch
import torch.nn.functional as F


def _build_inverse_distance_matrix(H, W):
    # type: (int, int) -> torch.Tensor
    """
    Precompute inverse pairwise spatial distance for an HxW grid.

    Returns:
        [H*W, H*W] matrix where entry (i, j) = 1 / (dist(i, j) + 1)
    """
    coords_h = torch.arange(H, dtype=torch.float32)
    coords_w = torch.arange(W, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(coords_h, coords_w, indexing='ij')
    coords = torch.stack([grid_y.reshape(-1), grid_x.reshape(-1)], dim=-1)  # [N, 2]

    # Pairwise Euclidean distance
    dist = torch.cdist(coords.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)  # [N, N]
    inv_dist = 1.0 / (dist + 1.0)
    return inv_dist


# Cache for distance matrices (grid sizes are small and reused)
_inv_dist_cache = {}


def topographic_spatial_loss(feature_map, alpha=0.25):
    # type: (torch.Tensor, float) -> torch.Tensor
    """
    TDANN-style topographic spatial smoothness loss (Margalit et al. 2024, Neuron).

    Encourages nearby spatial locations in the feature map to have correlated
    activations, reproducing the topographic organization observed in primate
    visual cortex.

    Loss = -alpha * pearson_correlation(response_similarity, inverse_distance)

    When nearby cells respond similarly (high correlation with inverse distance),
    the loss is low (negative of a positive correlation). When responses are
    spatially disorganized, the loss is high.

    Args:
        feature_map: [B, C, H, W] feature activations from the tectum
        alpha: weight for the topographic loss term (default 0.25)

    Returns:
        Scalar loss tensor (differentiable)
    """
    B, C, H, W = feature_map.shape
    N = H * W

    # Get or build inverse distance matrix
    cache_key = (H, W)
    if cache_key not in _inv_dist_cache:
        _inv_dist_cache[cache_key] = _build_inverse_distance_matrix(H, W)
    inv_dist = _inv_dist_cache[cache_key].to(feature_map.device)  # [N, N]

    # Flatten spatial dims: [B, C, N]
    features_flat = feature_map.reshape(B, C, N)

    # Pairwise cosine similarity between spatial locations
    # Normalize along channel dim
    features_normed = F.normalize(features_flat, dim=1)  # [B, C, N]
    # sim[b, i, j] = cosine similarity between location i and j
    sim = torch.bmm(features_normed.permute(0, 2, 1), features_normed)  # [B, N, N]

    # Average similarity across batch
    sim_mean = sim.mean(dim=0)  # [N, N]

    # Pearson correlation between flattened similarity and inverse distance
    sim_flat = sim_mean.reshape(-1)
    inv_dist_flat = inv_dist.reshape(-1)

    # Pearson r = cov(x,y) / (std(x) * std(y))
    sim_centered = sim_flat - sim_flat.mean()
    dist_centered = inv_dist_flat - inv_dist_flat.mean()

    cov = (sim_centered * dist_centered).mean()
    std_sim = sim_centered.std() + 1e-8
    std_dist = dist_centered.std() + 1e-8
    pearson_r = cov / (std_sim * std_dist)

    # Negative correlation = loss (we want to maximize correlation)
    return -alpha * pearson_r
