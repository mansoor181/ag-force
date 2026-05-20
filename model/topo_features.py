"""Topological Binding Signatures via Persistent Homology.

Extracts topological invariants (connected components, loops, voids) from
CDR-epitope interfaces that capture global shape complementarity.

Key insight: Topology is orthogonal to sequence. If a CDR must form a loop
threading through an antigen cavity, this topological constraint is captured
by persistent homology but missed by distance-based features.

References:
- PMC11891663: PH for protein complex interface quality assessment
- arxiv:2505.22786: Protein-nucleic acid binding affinity via PH
- Nature Comms 2025: Topological determinants capture binding sites
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import gudhi, fall back to placeholder if not available
try:
    import gudhi
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False
    print("Warning: gudhi not installed. Topological features will use placeholder.")


def compute_persistence_diagram(coords: np.ndarray, max_edge_length: float = 12.0,
                                 max_dimension: int = 2) -> dict:
    """Compute persistence diagram for a point cloud.

    Args:
        coords: (N, 3) numpy array of 3D coordinates
        max_edge_length: Maximum edge length for Rips complex (Angstroms)
        max_dimension: Maximum homology dimension (0=components, 1=loops, 2=voids)

    Returns:
        dict with keys 'H0', 'H1', 'H2' containing (birth, death) arrays
    """
    if not GUDHI_AVAILABLE or len(coords) < 4:
        # Return empty diagrams
        return {f'H{d}': np.zeros((0, 2)) for d in range(max_dimension + 1)}

    # Build Rips complex
    rips = gudhi.RipsComplex(points=coords.tolist(), max_edge_length=max_edge_length)
    simplex_tree = rips.create_simplex_tree(max_dimension=max_dimension + 1)

    # Compute persistence
    simplex_tree.compute_persistence()

    # Extract diagrams per dimension
    diagrams = {}
    for dim in range(max_dimension + 1):
        intervals = simplex_tree.persistence_intervals_in_dimension(dim)
        # Filter out infinite deaths (use max_edge_length as proxy)
        finite_intervals = []
        for birth, death in intervals:
            if death == float('inf'):
                death = max_edge_length
            finite_intervals.append([birth, death])
        diagrams[f'H{dim}'] = np.array(finite_intervals) if finite_intervals else np.zeros((0, 2))

    return diagrams


def persistence_image(diagram: np.ndarray, resolution: int = 20,
                      sigma: float = 0.5, birth_range: tuple = (0, 12),
                      persist_range: tuple = (0, 12)) -> np.ndarray:
    """Convert persistence diagram to persistence image (vectorized representation).

    A persistence image is a stable, fixed-size vectorization of a persistence diagram.
    Each (birth, death) pair is mapped to (birth, persistence=death-birth) and
    convolved with a Gaussian, then discretized on a grid.

    Args:
        diagram: (N, 2) array of (birth, death) pairs
        resolution: Grid resolution (output is resolution x resolution)
        sigma: Gaussian kernel bandwidth
        birth_range: (min, max) for birth axis
        persist_range: (min, max) for persistence axis

    Returns:
        (resolution, resolution) persistence image
    """
    if len(diagram) == 0:
        return np.zeros((resolution, resolution))

    # Convert to (birth, persistence) coordinates
    births = diagram[:, 0]
    persistences = diagram[:, 1] - diagram[:, 0]

    # Create grid
    birth_grid = np.linspace(birth_range[0], birth_range[1], resolution)
    persist_grid = np.linspace(persist_range[0], persist_range[1], resolution)

    # Compute persistence image via Gaussian kernel
    image = np.zeros((resolution, resolution))
    for b, p in zip(births, persistences):
        if p <= 0:
            continue
        # Weight by persistence (more persistent = more important)
        weight = p
        # Add Gaussian centered at (b, p)
        for i, bg in enumerate(birth_grid):
            for j, pg in enumerate(persist_grid):
                dist_sq = (b - bg) ** 2 + (p - pg) ** 2
                image[j, i] += weight * np.exp(-dist_sq / (2 * sigma ** 2))

    return image


def persistence_statistics(diagram: np.ndarray) -> np.ndarray:
    """Compute summary statistics from persistence diagram.

    Args:
        diagram: (N, 2) array of (birth, death) pairs

    Returns:
        (10,) statistics: [n_features, mean_birth, std_birth, mean_death, std_death,
                          mean_persist, std_persist, max_persist, total_persist, entropy]
    """
    if len(diagram) == 0:
        return np.zeros(10)

    births = diagram[:, 0]
    deaths = diagram[:, 1]
    persistences = deaths - births

    # Filter valid persistences
    valid = persistences > 0
    if not valid.any():
        return np.zeros(10)

    persistences = persistences[valid]
    births = births[valid]
    deaths = deaths[valid]

    # Compute statistics
    n = len(persistences)
    stats = [
        n,                                    # Number of features
        births.mean(),                        # Mean birth
        births.std() if n > 1 else 0,         # Std birth
        deaths.mean(),                        # Mean death
        deaths.std() if n > 1 else 0,         # Std death
        persistences.mean(),                  # Mean persistence
        persistences.std() if n > 1 else 0,   # Std persistence
        persistences.max(),                   # Max persistence
        persistences.sum(),                   # Total persistence
        _persistence_entropy(persistences),   # Persistence entropy
    ]
    return np.array(stats)


def _persistence_entropy(persistences: np.ndarray) -> float:
    """Compute entropy of persistence distribution."""
    if len(persistences) == 0 or persistences.sum() == 0:
        return 0.0
    p = persistences / persistences.sum()
    return -np.sum(p * np.log(p + 1e-10))


class TopologicalFeatureExtractor:
    """Extract topological features from protein structures.

    Computes persistent homology for:
    1. CDR point cloud (captures CDR shape)
    2. Epitope point cloud (captures epitope shape)
    3. Interface point cloud (captures binding geometry)
    4. Combined CDR+epitope (captures complementarity)
    """

    def __init__(self, max_edge_length: float = 12.0, max_dimension: int = 2,
                 image_resolution: int = 16, sigma: float = 1.0):
        self.max_edge_length = max_edge_length
        self.max_dimension = max_dimension
        self.image_resolution = image_resolution
        self.sigma = sigma

    def extract(self, cdr_coords: np.ndarray, epitope_coords: np.ndarray,
                return_images: bool = True) -> dict:
        """Extract topological features from CDR-epitope interface.

        Args:
            cdr_coords: (N_cdr, 3) CDR CA coordinates
            epitope_coords: (N_epi, 3) epitope CA coordinates
            return_images: If True, return persistence images; else return statistics

        Returns:
            dict with topological features
        """
        features = {}

        # 1. CDR topology
        cdr_diag = compute_persistence_diagram(
            cdr_coords, self.max_edge_length, self.max_dimension)

        # 2. Epitope topology
        epi_diag = compute_persistence_diagram(
            epitope_coords, self.max_edge_length, self.max_dimension)

        # 3. Interface topology (combined)
        if len(cdr_coords) > 0 and len(epitope_coords) > 0:
            interface_coords = np.concatenate([cdr_coords, epitope_coords], axis=0)
        else:
            interface_coords = cdr_coords if len(cdr_coords) > 0 else epitope_coords
        interface_diag = compute_persistence_diagram(
            interface_coords, self.max_edge_length, self.max_dimension)

        if return_images:
            # Return persistence images (suitable for CNN processing)
            for name, diag in [('cdr', cdr_diag), ('epi', epi_diag), ('interface', interface_diag)]:
                for dim in range(self.max_dimension + 1):
                    img = persistence_image(
                        diag[f'H{dim}'],
                        resolution=self.image_resolution,
                        sigma=self.sigma
                    )
                    features[f'{name}_H{dim}_image'] = img
        else:
            # Return statistics (fixed-size vectors)
            for name, diag in [('cdr', cdr_diag), ('epi', epi_diag), ('interface', interface_diag)]:
                for dim in range(self.max_dimension + 1):
                    stats = persistence_statistics(diag[f'H{dim}'])
                    features[f'{name}_H{dim}_stats'] = stats

        return features

    def get_feature_dim(self, return_images: bool = True) -> int:
        """Get total feature dimension."""
        n_sources = 3  # cdr, epi, interface
        n_dims = self.max_dimension + 1  # H0, H1, H2
        if return_images:
            return n_sources * n_dims * (self.image_resolution ** 2)
        else:
            return n_sources * n_dims * 10  # 10 statistics per diagram


class TopologicalEncoder(nn.Module):
    """Neural network encoder for topological features.

    Takes persistence images or statistics and encodes them into
    a fixed-size embedding that can be used for conditioning.
    """

    def __init__(self, output_dim: int = 64, image_resolution: int = 16,
                 max_dimension: int = 2, use_images: bool = True):
        super().__init__()
        self.output_dim = output_dim
        self.image_resolution = image_resolution
        self.max_dimension = max_dimension
        self.use_images = use_images

        n_sources = 3  # cdr, epi, interface
        n_dims = max_dimension + 1

        if use_images:
            # CNN for persistence images
            n_channels = n_sources * n_dims  # 9 channels (3 sources x 3 dimensions)
            self.encoder = nn.Sequential(
                nn.Conv2d(n_channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(64 * (image_resolution // 4) ** 2, 128),
                nn.ReLU(),
                nn.Linear(128, output_dim),
            )
        else:
            # MLP for statistics
            input_dim = n_sources * n_dims * 10
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, output_dim),
            )

    def forward(self, topo_features: dict) -> torch.Tensor:
        """Encode topological features.

        Args:
            topo_features: dict from TopologicalFeatureExtractor.extract()

        Returns:
            (output_dim,) encoded topological embedding
        """
        if self.use_images:
            # Stack images into tensor: (n_channels, H, W)
            images = []
            for source in ['cdr', 'epi', 'interface']:
                for dim in range(self.max_dimension + 1):
                    key = f'{source}_H{dim}_image'
                    img = topo_features.get(key, np.zeros((self.image_resolution, self.image_resolution)))
                    images.append(torch.tensor(img, dtype=torch.float32))
            x = torch.stack(images, dim=0)  # (n_channels, H, W)
            x = x.unsqueeze(0)  # (1, n_channels, H, W)
            return self.encoder(x).squeeze(0)
        else:
            # Concatenate statistics
            stats = []
            for source in ['cdr', 'epi', 'interface']:
                for dim in range(self.max_dimension + 1):
                    key = f'{source}_H{dim}_stats'
                    s = topo_features.get(key, np.zeros(10))
                    stats.append(torch.tensor(s, dtype=torch.float32))
            x = torch.cat(stats, dim=0)  # (n_sources * n_dims * 10,)
            return self.encoder(x)


class TopologicalBindingSignature(nn.Module):
    """Full topological binding signature module.

    Computes persistent homology features for CDR-epitope interface
    and provides them as conditioning signal for sequence prediction.
    """

    def __init__(self, hidden_dim: int = 256, topo_dim: int = 64,
                 max_edge_length: float = 12.0, use_images: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.topo_dim = topo_dim

        # Feature extractor (non-parametric)
        self.extractor = TopologicalFeatureExtractor(
            max_edge_length=max_edge_length,
            max_dimension=2,
            image_resolution=16 if use_images else 0,
        )

        # Neural encoder
        self.encoder = TopologicalEncoder(
            output_dim=topo_dim,
            image_resolution=16,
            max_dimension=2,
            use_images=use_images,
        )

        # Project to hidden dim for conditioning
        self.proj = nn.Linear(topo_dim, hidden_dim)

    def forward(self, cdr_coords: torch.Tensor, epitope_coords: torch.Tensor) -> torch.Tensor:
        """Compute topological binding signature.

        Args:
            cdr_coords: (N_cdr, 3) CDR CA coordinates
            epitope_coords: (N_epi, 3) epitope CA coordinates

        Returns:
            (hidden_dim,) topological conditioning vector
        """
        # Move to numpy for gudhi
        cdr_np = cdr_coords.detach().cpu().numpy()
        epi_np = epitope_coords.detach().cpu().numpy()

        # Extract topological features
        use_images = self.encoder.use_images
        topo_features = self.extractor.extract(cdr_np, epi_np, return_images=use_images)

        # Move features to same device as input
        device = cdr_coords.device

        # Encode
        topo_emb = self.encoder(topo_features).to(device)

        # Project to hidden dimension
        return self.proj(topo_emb)

    def compute_topo_loss(self, pred_coords: torch.Tensor, true_coords: torch.Tensor,
                          epitope_coords: torch.Tensor) -> torch.Tensor:
        """Compute topological similarity loss between predicted and true CDR.

        This encourages the predicted CDR to have similar topological properties
        to the true CDR in the context of the epitope.

        Args:
            pred_coords: (N_cdr, 3) predicted CDR coordinates
            true_coords: (N_cdr, 3) true CDR coordinates
            epitope_coords: (N_epi, 3) epitope coordinates

        Returns:
            Scalar topological loss
        """
        # Get topological embeddings for both
        pred_topo = self.forward(pred_coords, epitope_coords)
        true_topo = self.forward(true_coords, epitope_coords)

        # MSE loss on topological embeddings
        return F.mse_loss(pred_topo, true_topo)
