import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rescale
from skimage import io as skio
from skimage.morphology import (
    binary_dilation,
    binary_erosion,
    skeletonize,
    cube
)
from skimage import exposure, util                              # ← NEW
from skimage.morphology import (
    binary_dilation,
    binary_erosion,
    cube,
    ball                                                    # ← NEW
)
import os
from skimage.filters import threshold_local, threshold_otsu
import tifffile as tiff
from scipy.ndimage import distance_transform_edt, map_coordinates
from sklearn.decomposition import PCA
import sys
import logging
from scipy.spatial import cKDTree  # For efficient spatial queries
from scipy.io import loadmat
from collections import defaultdict
import math
import random
import matplotlib.ticker as ticker

try:
    import czifile
except ImportError:
    czifile = None

########################################################
#                Logging Configuration
########################################################
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
logging.info("Logging reconfigured after imports.")

########################################################
#             Global Parameters / Settings
########################################################

# ------- Input / Output ----------
input_file = "NPN02_MD.tif"  # .mat or .tif or .czi
# input_file = "fiber_network_2.mat"
mat_variable_name = "fiber_network"  # variable name if .mat
max_fibers_to_process = 2000
slice_selection = 9

# ------- Diam. thresholds (microns) -------
diameter_threshold_max = 10.0
diameter_threshold_min = 0

# ------- Sampling / Spacing --------
# If your .mat or .tif is isotropic (Z=Y=X=1), just keep these =1.
# For .tif with different z-spacing, set scaling_factor_z accordingly.
scaling_factor_z = 5.0
scaling_factor_y = 1.2044
scaling_factor_x = 1.2044


# scaling_factor_z = 1.0
# scaling_factor_y = 1.0
# scaling_factor_x = 1.0
sampling = np.array([scaling_factor_z, scaling_factor_y, scaling_factor_x])  # (sz, sy, sx)

# ------- Overlay Output ----------
overlay_filename = "fiber_overlay.tif"

########################################################
#                  Utility Functions
########################################################

# --- Robust MAT-file reader (works for v7.3 HDF5 files) -----------------------
import warnings
try:
    import h5py                      # HDF5 backend for -v7.3 files
except ImportError:
    h5py = None                      # handled later

from scipy.io import loadmat         # still fine for classic MAT files

def load_mat_any(filepath, variable=None):
    """
    Load a MATLAB file saved with *any* version.
      • If `variable` is given, return that dataset (else raise).
      • Otherwise, return the first non-meta dataset found.
    Returned array is always transposed to (Z, Y, X) to keep the rest of the
    pipeline unchanged.

    Examples
    --------
    z_stack = load_mat_any("fiber_network_4.mat", "fiber_network")
    z_stack = load_mat_any("legacy.mat")            # auto-detects the dataset
    """
    # ---------- 1) try the fast/classic route --------------------------------
    try:
        mat = loadmat(filepath, simplify_cells=True)
        datasets = {k: v for k, v in mat.items() if not k.startswith("__")}
        if not datasets:
            raise KeyError("No datasets inside MAT file")

        if variable:
            if variable not in datasets:
                raise KeyError(f"'{variable}' not found in {filepath}")
            arr = datasets[variable]
        else:
            if len(datasets) > 1:
                warnings.warn(f"Multiple variables found; using '{next(iter(datasets))}'.")
            arr = next(iter(datasets.values()))

        return np.asarray(arr).transpose(2, 1, 0)   # (Z,Y,X)

    # ---------- 2) fall back to HDF5 route (v7.3) -----------------------------
    except Exception as e:
        if h5py is None:
            raise RuntimeError(
                "File looks like MATLAB v7.3 (HDF5) but h5py is missing. "
                "Please `pip install h5py`."
            ) from e

        with h5py.File(filepath, "r") as f:
            keys = list(f.keys())
            if not keys:
                raise RuntimeError("No datasets found in HDF5 MAT file")

            if variable:
                if variable not in f:
                    raise KeyError(f"'{variable}' not found in {filepath}")
                data = f[variable][()]
            else:
                if len(keys) > 1:
                    warnings.warn(f"Multiple variables found; using '{keys[0]}'.")
                data = f[keys[0]][()]

        # np.ndarray comes out in Fortran order: axes = (X,Y,Z)
        # Transpose to (Z,Y,X) for consistency with tif/3-D logic
        return np.asarray(data).transpose(2, 1, 0)
    
from matplotlib import cm


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def truncate_colormap(cmap_name, minval=0.2, maxval=1.0, n=256):
    """
    Return a new colormap based on `cmap_name`, but “cut off” below `minval`.
    - cmap_name: string name (e.g. 'Blues', 'Greens', 'Reds') or a Colormap instance.
    - minval, maxval: floats in [0,1] specifying the fraction of the original cmap to keep.
      minval=0.2 means the bottom 20% of the original colormap (the whitest‐looking part)
      is removed, so 0→0.2 of the original is never used.
    - n: number of discrete color samples (256 is usually fine).
    """
    if isinstance(cmap_name, str):
        base = plt.get_cmap(cmap_name)
    else:
        base = cmap_name
    
    # Sample `n` points between minval and maxval from the original colormap
    colors = base(np.linspace(minval, maxval, n))
    new_cmap = LinearSegmentedColormap.from_list(f"{base.name}_trunc", colors)
    return new_cmap

def plot_colored_fibers(traced_fibers, fiber_props,
                        color_by='elevation',
                        cmap_name='Greens',
                        vmin=0,
                        vmax=90,
                        title=""):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    if isinstance(cmap_name, str):
        cmap = cm.get_cmap(cmap_name)
    else:
        cmap = cmap_name

    norm = Normalize(vmin=vmin, vmax=vmax)
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')

    for voxels, props in zip(traced_fibers, fiber_props):
        value = props[color_by]
        color = cmap(norm(value))
        scaled_voxels = voxels * sampling
        X = scaled_voxels[:, 2]
        Y = scaled_voxels[:, 1]
        Z = scaled_voxels[:, 0]
        ax.plot(X, Y, Z, color=color, linewidth=2)

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = plt.colorbar(mappable, ax=ax, pad=0.1)
    cbar.set_label(f"{color_by.capitalize()}")

    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Y (µm)")
    ax.set_zlabel("Z (µm)")
    ax.set_title(title)
    set_axes_equal(ax)
    plt.tight_layout()
    plt.show()

def plot_colored_fibers_2d_projection(traced_fibers, fiber_props,
                                      color_by='azimuth',
                                      cmap_name='Blues',  # can be string or Colormap
                                      vmin=0,
                                      vmax=180,
                                      title="2D Max Projection Colored by Azimuth (°)"):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    # If user passed a string, turn it into a Colormap. Otherwise assume it's already a Colormap.
    if isinstance(cmap_name, str):
        cmap = cm.get_cmap(cmap_name)
    else:
        # Already a Colormap object (e.g. from truncate_colormap)
        cmap = cmap_name

    norm = Normalize(vmin=vmin, vmax=vmax)
    fig, ax = plt.subplots(figsize=(9, 8))

    for voxels, props in zip(traced_fibers, fiber_props):
        value = props[color_by]
        color = cmap(norm(value))
        scaled_voxels = voxels * sampling
        X = scaled_voxels[:, 2]  # x
        Y = scaled_voxels[:, 1]  # y
        ax.plot(X, Y, color=color, linewidth=1.5)

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = plt.colorbar(mappable, ax=ax, pad=0.01)
    cbar.set_label(f"{color_by.capitalize()} (°)")

    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Y (µm)")
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()

def compute_local_thickness(fiber_mask, max_radius=None):
    """
    Computes local thickness (largest inscribed diameter) at each voxel
    in a 3D binary mask via iterative spherical erosion.
    """
    from scipy.ndimage import distance_transform_edt
    fiber_mask = fiber_mask.astype(bool)
    if max_radius is None:
        max_radius = int(np.ceil(distance_transform_edt(fiber_mask).max()))
    thickness_map = np.zeros(fiber_mask.shape, dtype=np.float32)
    for r in range(max_radius, 0, -1):
        eroded = binary_erosion(fiber_mask, ball(r))
        mask_new = eroded & (thickness_map == 0)
        thickness_map[mask_new] = 2 * r
    return thickness_map


def thickness_map_iso(fiber_mask, sampling, iso=0.5):
    """
    Local-thickness map in µm with sub-µm precision.
      iso : target isotropic voxel size (µm)  (0.5 → 0.5 µm³ voxels)
    """
    # a) upscale to cubic voxels of size `iso`
    zoom = sampling / iso                   # e.g. [10.0, 2.4088, 2.4088]
    mask_iso = rescale(fiber_mask.astype(float), zoom,
                       order=0, preserve_range=True).astype(bool)

    # b) local thickness on that grid  (vox units)
    #    -- identical algorithm but much smaller quantisation step
    max_r = int(np.ceil(np.sqrt(mask_iso.shape).max()))   # safe upper bound
    thick_vox = np.zeros_like(mask_iso, dtype=np.uint16)
    for r in range(max_r, 0, -1):
        survive = binary_erosion(mask_iso, ball(r))
        thick_vox[(survive) & (thick_vox == 0)] = 2*r      # diameter in vox

    thick_iso_um = thick_vox.astype(np.float32) * iso      # convert to µm

    # c) back-resample to original grid
    back = 1 / zoom
    thick_um = rescale(thick_iso_um, back,
                       order=0, preserve_range=True).astype(np.float32)
    # crop in case the rescale is 1-2 voxels larger
    z,y,x = fiber_mask.shape
    return thick_um[:z,:y,:x]


def plot_colored_fibers(traced_fibers, fiber_props, color_by='azimuth', cmap_name='twilight', vmin=0, vmax=180, title=""):
    """
    Plot traced fibers one-by-one in 3D colored by azimuth, elevation, or diameter.
    
    traced_fibers: list of (N_i,3) arrays of voxel coordinates
    fiber_props: list of dicts from compute_fiber_properties
    color_by: 'azimuth', 'elevation', or 'diameter'
    cmap_name: matplotlib colormap name
    vmin, vmax: color normalization limits
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import numpy as np

    fig = plt.figure(figsize=(9,8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Prepare colormap
    cmap = cm.get_cmap(cmap_name)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    for voxels, props in zip(traced_fibers, fiber_props):
        # Get color value
        value = props[color_by]
        color = cmap(norm(value))
        
        # Scale voxels
        scaled_voxels = voxels * sampling
        X = scaled_voxels[:,2]
        Y = scaled_voxels[:,1]
        Z = scaled_voxels[:,0]
        
        ax.plot(X, Y, Z, color=color, linewidth=2)

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = plt.colorbar(mappable, ax=ax, pad=0.1)
    cbar.set_label(f"{color_by.capitalize()}")

    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Y (µm)")
    ax.set_zlabel("Z (µm)")
    ax.set_title(title)
    set_axes_equal(ax)
    plt.tight_layout()
    plt.show()

def validate_input_data(z_stack):
    logging.info("Validating input data...")
    if not isinstance(z_stack, np.ndarray):
        logging.error("Input data is not a numpy array.")
        sys.exit(1)
    if z_stack.ndim != 3:
        logging.error(f"Expected 3D image stack, got {z_stack.ndim}D.")
        sys.exit(1)
    if not np.issubdtype(z_stack.dtype, np.number):
        logging.error("Image stack contains non-numeric data.")
        sys.exit(1)
    logging.info("Input data validation passed.")

def adaptive_threshold_3d(z_stack, block_size=15, offset=0):
    """
    Applies local adaptive thresholding on each Z-slice
    for grayscale images in shape (Z, Y, X).
    """
    binary_stack = np.zeros_like(z_stack, dtype=bool)
    for z in range(z_stack.shape[0]):
        local_thresh = threshold_local(z_stack[z], block_size=block_size, offset=offset)
        binary_stack[z] = z_stack[z] >= local_thresh
    return binary_stack

def simple_threshold_3d(z_stack, threshold_value):
    """
    Simple fixed threshold across entire volume.
    """
    z_stack = z_stack.astype(np.float32)
    return (z_stack >= threshold_value).astype(np.uint8)

########################################################
#  Diameter and Curvature Computations
########################################################

def compute_local_fiber_orientation(fiber_vox):
    """
    Compute tangent vectors via central differences.
    fiber_vox: Nx3 array of voxel coordinates (z,y,x).
    Returns: Nx3 array of unit tangent vectors in voxel space.
    """
    N = len(fiber_vox)
    if N < 2:
        return np.zeros((N, 3))
    
    tangents = np.zeros_like(fiber_vox, dtype=np.float32)
    if N > 2:
        tangents[1:-1] = (fiber_vox[2:] - fiber_vox[:-2]) / 2.0
    tangents[0] = fiber_vox[1] - fiber_vox[0]
    tangents[-1] = fiber_vox[-1] - fiber_vox[-2]

    norms = np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-12
    return tangents / norms

def compute_diameters(fiber_vox, dt_image, fiber_segment):
    """
    Compute diameters along the fiber in real units (microns).
    fiber_vox: Nx3 array of voxel coordinates (z,y,x).
    dt_image: distance transform (same shape as fiber_segment),
              but *values* are in microns (if called with sampling).
    fiber_segment: boolean fiber mask.
    Returns: Nx diameters in microns.
    """
    tangents = compute_local_fiber_orientation(fiber_vox)
    diameters = []
    for i, point_vox in enumerate(fiber_vox):
        tangent_vox = tangents[i]
        d = compute_cross_section_diameter(point_vox, tangent_vox,
                                           dt_image, fiber_segment)
        diameters.append(d)
    return np.array(diameters)

def compute_cross_section_diameter(voxel_point, tangent_vox,
                                   dt_image, fiber_segment, half_width=10):
    """
    Compute the local fiber diameter at a given point in voxel coords.
    dt_image stores distances in microns. The indexing is [z, y, x].
    tangent_vox is the unit tangent in voxel space.
    """

    # Ensure tangent is normalized
    tangent_vox = tangent_vox / (np.linalg.norm(tangent_vox) + 1e-12)
    
    # Construct a local frame: find n1, n2 perpendicular to tangent.
    # We'll pick an arbitrary vector that isn't parallel:
    arbitrary = np.array([1,0,0], dtype=float)
    cross_val = np.cross(arbitrary, tangent_vox)
    if np.allclose(cross_val, 0, atol=1e-6):
        arbitrary = np.array([0,1,0], dtype=float)
    n1 = np.cross(tangent_vox, arbitrary)
    n1 /= (np.linalg.norm(n1) + 1e-12)
    n2 = np.cross(tangent_vox, n1)
    n2 /= (np.linalg.norm(n2) + 1e-12)

    # Create a grid of cross-sectional points in voxel space.
    i_indices = np.arange(-half_width, half_width+1)
    j_indices = np.arange(-half_width, half_width+1)
    I, J = np.meshgrid(i_indices, j_indices, indexing='ij')

    I_flat = I.ravel()
    J_flat = J.ravel()

    # Each step i or j is 1 voxel in the local n1 / n2 directions:
    grid_coords = (voxel_point[None, :] +
                   I_flat[:, None]*n1 +
                   J_flat[:, None]*n2)
    # grid_coords is shape (N, 3) in (z,y,x) voxel coords.

    # map_coordinates expects array ordering [z, y, x].
    # We'll sample dt_image & fiber_segment in "nearest" mode for safety.
    dt_patch = map_coordinates(dt_image,
                               [grid_coords[:,0],  # z
                                grid_coords[:,1],  # y
                                grid_coords[:,2]], # x
                               order=1, mode='nearest')
    seg_patch = map_coordinates(fiber_segment.astype(float),
                                [grid_coords[:,0],
                                 grid_coords[:,1],
                                 grid_coords[:,2]],
                                order=0, mode='nearest')

    size = 2*half_width + 1
    dt_patch = dt_patch.reshape(size, size)
    seg_patch = seg_patch.reshape(size, size)

    # The center voxel corresponds to the fiber point
    center_dt = dt_patch[half_width, half_width]

    # dt_patch values are in microns (distance in real space).
    # Let's find a "robust radius" from the patch within the fiber.
    fiber_mask = seg_patch > 0.5
    if np.any(fiber_mask):
        masked_dt_values = dt_patch[fiber_mask]
        robust_radius = np.max(masked_dt_values)  # in microns
    else:
        # Fallback if no fiber pixels found
        robust_radius = center_dt
    
    diameter = robust_radius * 2.0  # in microns
    return diameter

def compute_curvature(fiber_scaled):
    """
    Computes the curvature of a scaled fiber in microns.
    fiber_scaled: Nx3 array of coordinates in real space (microns).
    Returns: average curvature (1/microns).
    """
    if len(fiber_scaled) < 3:
        return 0.0
    diffs = np.diff(fiber_scaled, axis=0)
    norms = np.linalg.norm(diffs, axis=1)
    epsilon = 1e-8
    tangents = diffs / (norms[:, np.newaxis] + epsilon)
    dtangents = np.diff(tangents, axis=0)
    curvature = np.mean(np.linalg.norm(dtangents, axis=1))
    return curvature

########################################################
#         Skeletonization and Path Segmentation
########################################################

def segment_skeleton(skeleton, coords, angle_threshold_degs=70):
    """
    Segments the skeleton (3D binary image, shape (Z,Y,X)) into paths.
    'coords' is Nx3 array of voxel coordinates (z,y,x).
    """
    adjacency = defaultdict(list)
    coord_to_index = {tuple(c): i for i, c in enumerate(coords)}

    neighbor_offsets = []
    for dz in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                neighbor_offsets.append((dz, dy, dx))

    def get_neighbors(cz, cy, cx):
        neighs = []
        for (dz, dy, dx) in neighbor_offsets:
            nz, ny, nx = cz + dz, cy + dy, cx + dx
            if (0 <= nz < skeleton.shape[0]
                and 0 <= ny < skeleton.shape[1]
                and 0 <= nx < skeleton.shape[2]):
                if skeleton[nz, ny, nx]:
                    neighs.append((nz, ny, nx))
        return neighs

    # Build adjacency
    for i, c in enumerate(coords):
        z, y, x = c
        for nb in get_neighbors(z, y, x):
            j = coord_to_index[nb]
            adjacency[i].append(j)

    endpoints = set()
    junctions = set()
    visited = set()
    paths = []

    # Identify endpoints (degree <=1) and junctions (degree >2)
    for idx, nbrs in adjacency.items():
        if len(nbrs) <= 1:
            endpoints.add(idx)
        elif len(nbrs) > 2:
            junctions.add(idx)

    def angle_between(u, v):
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu < 1e-12 or nv < 1e-12:
            return 180.0
        dot = np.dot(u, v) / (nu * nv)
        dot = np.clip(dot, -1.0, 1.0)
        return math.degrees(math.acos(dot))

    # Depth-first style path trace
    def trace_path(start):
        path = []
        current = start
        prev = -1
        while True:
            path.append(current)
            visited.add(current)
            nbrs = adjacency[current]
            unvisited = [n for n in nbrs if n != prev and n not in visited]
            if len(unvisited) == 0:
                break
            if len(unvisited) == 1:
                nxt = unvisited[0]
                if prev != -1:
                    vec_in = coords[current] - coords[prev]
                    vec_out = coords[nxt] - coords[current]
                    # If angle is too large, break
                    if angle_between(vec_in, vec_out) <= angle_threshold_degs:
                        prev, current = current, nxt
                        continue
                    else:
                        break
                else:
                    prev, current = current, nxt
                    continue
            break
        return path

    # Start from endpoints, then junctions, then leftover nodes
    for ep in endpoints:
        if ep not in visited:
            p = trace_path(ep)
            if len(p) > 1:
                paths.append(p)
    for jn in junctions:
        if jn not in visited:
            p = trace_path(jn)
            if len(p) > 1:
                paths.append(p)
    for node in set(range(len(coords))) - visited:
        p = trace_path(node)
        if len(p) > 1:
            paths.append(p)

    return paths

def compute_single_track_properties(track_vox, dt_image, fiber_segment, sampling):
    """
    Given a single skeleton path (list of voxel coords (z,y,x)),
    compute path length in microns and average diameter (microns).
    """
    if len(track_vox) < 2:
        return dict(length=0.0, avg_diameter=0.0)

    # track_vox is Nx3 in voxel coords.
    # Convert to physical units for length:
    diffs = np.diff(track_vox * sampling, axis=0)
    path_length = np.sum(np.linalg.norm(diffs, axis=1))

    # For diameter, use the dt_image. We simply read dt at each path voxel:
    track_vox_rounded = np.round(track_vox).astype(int)
    zmax, ymax, xmax = dt_image.shape
    track_vox_rounded[:, 0] = np.clip(track_vox_rounded[:, 0], 0, zmax-1)
    track_vox_rounded[:, 1] = np.clip(track_vox_rounded[:, 1], 0, ymax-1)
    track_vox_rounded[:, 2] = np.clip(track_vox_rounded[:, 2], 0, xmax-1)
    radii = dt_image[
        track_vox_rounded[:, 0],
        track_vox_rounded[:, 1],
        track_vox_rounded[:, 2]
    ]
    diameters = 2.0 * radii
    avg_diameter = np.mean(diameters) if len(diameters) else 0.0

    return dict(length=path_length, avg_diameter=avg_diameter)

def compute_initial_tortuosity(coords, paths, sampling):
    """
    Evaluate tortuosity on the skeleton paths to pick a threshold
    for polygonal track simplification.
    coords: Nx3 voxel coords
    """
    tortuosities = []
    for path in paths:
        path_vox = coords[path]  # voxel coords
        if len(path_vox) < 2:
            continue
        diffs = np.diff(path_vox * sampling, axis=0)
        path_len = np.sum(np.linalg.norm(diffs, axis=1))
        net_dist = np.linalg.norm((path_vox[-1] - path_vox[0]) * sampling)
        tort = path_len / net_dist if net_dist > 0 else np.inf
        tortuosities.append(tort)

    if not tortuosities:
        logging.warning("No tortuosity computed; using fallback = 4.0")
        return 4.0

    med = np.median(tortuosities)
    thr = med * 2
    logging.info(f"Median tortuosity: {med:.2f}, threshold: {thr:.2f}")
    return thr

def transform_to_polygonal_tracks(coords, paths, tortuosity_value, sampling):
    """
    Splits each skeleton path if it becomes too tortuous when extended,
    yielding a simplified polygonal representation in voxel coords.
    """
    logging.info("Transforming skeleton paths into polygonal tracks...")
    polygonal_tracks = []
    for path in paths:
        path_vox = coords[path]  # Nx3 voxel coords
        if len(path_vox) < 2:
            continue
        track_points = [path_vox[0]]
        start_idx = 0
        while start_idx < len(path_vox) - 1:
            for nxt in range(start_idx+1, len(path_vox)):
                seg = path_vox[start_idx:nxt+1]
                seg_diffs = np.diff(seg * sampling, axis=0)
                seg_len = np.sum(np.linalg.norm(seg_diffs, axis=1))
                direct = np.linalg.norm((seg[-1]-seg[0]) * sampling)
                tort = seg_len / direct if direct > 0 else np.inf
                if tort > tortuosity_value:
                    # too tortuous → break one step earlier
                    track_points.append(path_vox[nxt-1])
                    start_idx = nxt - 1
                    break
            else:
                track_points.append(path_vox[-1])
                break
        polygonal_tracks.append(np.array(track_points))
    return polygonal_tracks

def track_meets_criteria(track_prop,
                         dia_min=diameter_threshold_min,
                         dia_max=diameter_threshold_max):
    """
    Decide if a track is valid based on length and diameter constraints.
    """
    length = track_prop['length']
    avg_diam = track_prop['avg_diameter']
    if length < avg_diam * 3:
        return False
    if avg_diam < dia_min or avg_diam > dia_max:
        return False
    return True

def fiber_tracing(polygonal_tracks, track_props, sampling,
                  max_distance=5.0, angle_thresh=70.0):
    """
    Merges polygonal tracks into final fibers if they lie
    within 'max_distance' (in microns) and under 'angle_thresh' difference.
    We build a cKDTree in real-space (microns) so sampling is applied.
    """
    logging.info("Tracing fibers by connecting polygonal tracks...")
    if len(polygonal_tracks) == 0:
        logging.warning("No polygonal tracks available for fiber tracing.")
        return []

    # Convert track endpoints to real space for the KD-tree.
    # We'll store them in 3D arrays shaped (N,3).
    start_points = []
    end_points = []
    for trk in polygonal_tracks:
        start_points.append(trk[0] * sampling)  # in microns
        end_points.append(trk[-1] * sampling)   # in microns

    start_points = np.array(start_points)
    end_points   = np.array(end_points)

    start_tree = cKDTree(start_points)  # KD-tree in microns
    end_tree   = cKDTree(end_points)

    used = set()
    final_fibers = []

    for i, track_vox in enumerate(polygonal_tracks):
        if i in used or not track_meets_criteria(track_props[i]):
            used.add(i)
            continue
        # Start a new fiber with this track
        fiber_vox = track_vox.copy()
        used.add(i)
        extended = True

        while extended:
            extended = False
            # Merge from fiber's end:
            end_pt_microns = fiber_vox[-1] * sampling
            candidates = start_tree.query_ball_point(end_pt_microns, r=max_distance)
            candidates = [c for c in candidates if c not in used and c != i]
            for c in candidates:
                # Check if track c meets diameter/length criteria
                if not track_meets_criteria(track_props[c]):
                    continue
                # Check angle
                vect1_vox = fiber_vox[-1] - fiber_vox[-2] if len(fiber_vox) > 1 else np.array([0,0,0])
                cand = polygonal_tracks[c]
                vect2_vox = cand[1] - cand[0] if len(cand) > 1 else np.array([0,0,0])
                v1 = vect1_vox / (np.linalg.norm(vect1_vox) + 1e-12)
                v2 = vect2_vox / (np.linalg.norm(vect2_vox) + 1e-12)
                dotval = np.dot(v1, v2)
                dotval = np.clip(dotval, -1.0, 1.0)
                angle = np.degrees(np.arccos(dotval))
                if angle <= angle_thresh or (180 - angle) <= angle_thresh:
                    # Merge them
                    fiber_vox = np.concatenate([fiber_vox, cand], axis=0)
                    used.add(c)
                    extended = True
                    break

            # Merge from fiber's start:
            if not extended:
                start_pt_microns = fiber_vox[0] * sampling
                candidates = end_tree.query_ball_point(start_pt_microns, r=max_distance)
                candidates = [c for c in candidates if c not in used and c != i]
                for c in candidates:
                    if not track_meets_criteria(track_props[c]):
                        continue
                    cand = polygonal_tracks[c]
                    vect1_vox = fiber_vox[1] - fiber_vox[0] if len(fiber_vox) > 1 else np.array([0,0,0])
                    vect2_vox = cand[-1] - cand[-2] if len(cand) > 1 else np.array([0,0,0])
                    v1 = vect1_vox / (np.linalg.norm(vect1_vox) + 1e-12)
                    v2 = vect2_vox / (np.linalg.norm(vect2_vox) + 1e-12)
                    dotval = np.dot(v1, v2)
                    dotval = np.clip(dotval, -1.0, 1.0)
                    angle = np.degrees(np.arccos(dotval))
                    if angle <= angle_thresh or (180 - angle) <= angle_thresh:
                        fiber_vox = np.concatenate([cand, fiber_vox], axis=0)
                        used.add(c)
                        extended = True
                        break

        final_fibers.append(fiber_vox)

    logging.info(f"Traced {len(final_fibers)} fibers.")
    return final_fibers

########################################################
#      Fiber Measurements: Length, Orientation, etc.
########################################################

def compute_mean_fiber_radius(dt_image):
    """
    Mean radius from the distance transform (values in microns).
    """
    vals = dt_image[dt_image > 0]
    if len(vals) == 0:
        logging.warning("No non-zero DT values found.")
        return 0.0
    return vals.mean()

import numpy as np
import logging
from sklearn.decomposition import PCA

def compute_fiber_properties(
    traced_fibers,
    dt_image,
    fiber_segment,
    sampling,
    dia_min=diameter_threshold_min,
    dia_max=diameter_threshold_max
):
    """
    Computes properties for each traced fiber using advanced diameter computation.

    traced_fibers: list of arrays in voxel coords (z, y, x).
    dt_image: 3D array (shape: Z x Y x X) storing distance‐transform values in microns.
    fiber_segment: 3D boolean array of the segmented volume (Z x Y x X).
    sampling: array [sz, sy, sx] giving microns per voxel in each dimension.
    dia_min, dia_max: diameter thresholds (in µm).

    Returns:
      fiber_properties: list of dicts with keys:
        - 'length'       : total path length in µm
        - 'diameter'     : average fiber diameter in µm
        - 'tortuosity'   : path_length / straight_line_distance
        - 'orientation'  : principal axis vector in (Z, Y, X) order (unit length)
        - 'azimuth'      : angle in XY‐plane, in [0°, 180°]
        - 'elevation'    : angle from XY‐plane up to Z, in [0°, 90°]
        - 'curvature'    : average curvature (1/µm)
    """
    logging.info("Computing fiber properties...")
    fiber_properties = []

    # These lists collect per‐fiber az/el/diameter for summary statistics
    all_diameters = []
    all_elevations = []
    all_azimuths = []
    total_length = 0.0
    weighted_azimuth = 0.0
    weighted_elevation = 0.0
    weighted_diameter = 0.0

    for idx, fiber_vox in enumerate(traced_fibers):
        # Skip very short fibers
        if len(fiber_vox) < 2:
            logging.warning(f"Fiber {idx+1} is too short and will be skipped.")
            continue

        # STEP 1: Compute length in µm
        # fiber_vox is Nx3 array of (z, y, x). Convert to physical coords via sampling.
        fiber_scaled = fiber_vox.astype(float) * sampling  # shape (N, 3); order currently [Z, Y, X] in µm
        # The first column is Z_phys, second is Y_phys, third is X_phys.

        # Path length = sum of Euclidean distances between consecutive points
        diffs = np.diff(fiber_scaled, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        path_length = np.sum(segment_lengths)

        # STEP 2: Compute diameters at each sampled point
        diameters = compute_diameters(fiber_vox, dt_image, fiber_segment)
        if diameters.size == 0:
            logging.warning(f"Fiber {idx+1} has no valid diameters. Skipping.")
            continue
        avg_diameter = np.mean(diameters)
        all_diameters.append(avg_diameter)

        # STEP 3: Basic diameter filtering
        if (avg_diameter < dia_min) or (avg_diameter > dia_max):
            logging.info(f"Fiber {idx+1} excluded: diameter {avg_diameter:.2f} µm out of range [{dia_min}, {dia_max}].")
            all_diameters.pop()
            continue
        if path_length <= np.max(sampling) - 1:
            logging.info(f"Fiber {idx+1} excluded: diameter >= length (d*3 >= L).")
            all_diameters.pop()
            continue

        # STEP 4: Compute tortuosity = path_length / straight_line_distance
        start_pt = fiber_scaled[0]
        end_pt = fiber_scaled[-1]
        straight_line_distance = np.linalg.norm(end_pt - start_pt)
        if straight_line_distance <= 0:
            tortuosity = np.inf
        else:
            tortuosity = path_length / straight_line_distance
        if np.isnan(tortuosity) or np.isinf(tortuosity):
            logging.warning(f"Fiber {idx+1} has invalid tortuosity = {tortuosity}. Skipping.")
            all_diameters.pop()
            continue

        # STEP 5: PCA‐based Orientation
        # We want to run PCA in a conventional (X, Y, Z) Cartesian frame.
        # Currently fiber_scaled columns are [Z_phys, Y_phys, X_phys]. We reorder to [X, Y, Z].
        pts_xyz = fiber_scaled[:, [2, 1, 0]]  # shape (N, 3): columns = [X_phys, Y_phys, Z_phys]

        try:
            # Perform PCA on the Nx3 array pts_xyz
            pca = PCA(n_components=1)
            pca.fit(pts_xyz)
            principal_axis = pca.components_[0]  # unit vector in (X, Y, Z) order

            # Ensure the Z component is non‐negative for a consistent "upward" direction
            if principal_axis[2] < 0:
                principal_axis = -principal_axis

            # Extract components
            vx, vy, vz = principal_axis  # X‐component, Y‐component, Z‐component

            # Compute azimuth in [0°, 180°]:
            if (abs(vx) < 1e-12) and (abs(vy) < 1e-12):
                # Fiber is (nearly) purely vertical in Z. No well‐defined azimuth in XY plane.
                azimuth = 0.0
            else:
                raw_az = np.degrees(np.arctan2(vy, vx))  # in [–180°, +180°]
                if raw_az < 0:
                    raw_az += 360.0
                # Fold into [0°, 180°] to treat opposite directions as the same line
                if raw_az >= 180.0:
                    raw_az -= 180.0
                azimuth = raw_az

            # Compute elevation in [0°, 90°]:
            # Elevation = angle between the vector and the XY plane
            horizontal_length = np.sqrt(vx**2 + vy**2)
            if horizontal_length < 1e-12:
                # Purely vertical fiber
                elevation = 90.0
            else:
                elevation = np.degrees(np.arctan2(vz, horizontal_length))
            print(f"Fiber {idx + 1}:  Diameter = {avg_diameter:.2f} µm, Azimuth = {azimuth:.2f}°, Elevation = {elevation:.2f}°")
            # Collect in the lists
            all_azimuths.append(azimuth)
            all_elevations.append(elevation)

        except Exception as e:
            logging.error(f"Fiber {idx+1}: PCA failed with error: {e}. Skipping.")
            all_diameters.pop()
            continue

        # STEP 6: Compute curvature in µm⁻¹
        curvature = compute_curvature(fiber_scaled)

        # Store properties
        fiber_properties.append({
            'length': path_length,
            'diameter': avg_diameter,
            'tortuosity': tortuosity,
            # We can store the PCA axis in (X, Y, Z) or in (Z, Y, X). Here we keep the PCA result as (X, Y, Z).
            'orientation': principal_axis,  
            'azimuth': azimuth,
            'elevation': elevation,
            'curvature': curvature
        })

        # Accumulate weighted sums for length‐weighted averages
        total_length += path_length
        weighted_azimuth += path_length * azimuth
        weighted_elevation += path_length * elevation
        weighted_diameter += path_length * avg_diameter

    # -------------------------------------------------------------------------
    # PRINT SUMMARY STATISTICS
    if fiber_properties:
        # Unweighted mean ± std
        mean_diam = np.mean(all_diameters)
        std_diam = np.std(all_diameters)
        mean_el = np.mean(all_elevations)
        std_el = np.std(all_elevations)
        mean_az = np.mean(all_azimuths)
        std_az = np.std(all_azimuths)

        print("\nFiber Property Summary:")
        print(f"  Average Diameter:  {mean_diam:.2f} ± {std_diam:.2f} µm")
        print(f"  Average Elevation: {mean_el:.2f} ± {std_el:.2f}°")
        print(f"  Average Azimuth:   {mean_az:.2f} ± {std_az:.2f}°")

        # Length‐weighted averages
        if total_length > 0:
            wa = weighted_azimuth / total_length
            we = weighted_elevation / total_length
            wd = weighted_diameter / total_length
            print("\nLength‐Weighted Averages:")
            print(f"  Azimuth:   {wa:.2f}°")
            print(f"  Elevation: {we:.2f}°")
            print(f"  Diameter:  {wd:.2f} µm")

    logging.info(f"Computed properties for {len(fiber_properties)} valid fibers.")
    return fiber_properties

def save_fiber_analysis_to_csv(output_filename, z_stack, volume_fraction, mean_radius, tort_threshold, traced_fibers, fiber_props):
    """
    Save all fiber analysis data to a CSV file.
    """
    import pandas as pd
    import os
    from datetime import datetime

    # Create a dictionary for global metrics
    global_metrics = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'input_file': input_file,
        'image_shape_z': z_stack.shape[0],
        'image_shape_y': z_stack.shape[1],
        'image_shape_x': z_stack.shape[2],
        'volume_fraction': volume_fraction,
        'mean_radius_microns': mean_radius,
        'mean_diameter_microns': 2 * mean_radius,
        'tortuosity_threshold': tort_threshold,
        'total_fibers': len(traced_fibers)
    }

    # Calculate summary statistics from fiber properties
    if fiber_props:
        lengths = [p['length'] for p in fiber_props]
        diameters = [p['diameter'] for p in fiber_props]
        tortuosities = [p['tortuosity'] for p in fiber_props if not np.isinf(p['tortuosity'])]
        azimuths = [p['azimuth'] for p in fiber_props if not np.isnan(p['azimuth'])]
        elevations = [p['elevation'] for p in fiber_props if not np.isnan(p['elevation'])]
        curvatures = [p['curvature'] for p in fiber_props if not np.isnan(p['curvature'])]

        # Add summary statistics to global metrics
        global_metrics.update({
            'avg_fiber_length': np.mean(lengths),
            'std_fiber_length': np.std(lengths),
            'avg_fiber_diameter': np.mean(diameters),
            'std_fiber_diameter': np.std(diameters),
            'avg_tortuosity': np.mean(tortuosities) if tortuosities else np.nan,
            'std_tortuosity': np.std(tortuosities) if tortuosities else np.nan,
            'avg_azimuth': np.mean(azimuths) if azimuths else np.nan,
            'std_azimuth': np.std(azimuths) if azimuths else np.nan,
            'avg_elevation': np.mean(elevations) if elevations else np.nan,
            'std_elevation': np.std(elevations) if elevations else np.nan,
            'avg_curvature': np.mean(curvatures) if curvatures else np.nan,
            'std_curvature': np.std(curvatures) if curvatures else np.nan
        })

    # Save global metrics to one CSV file
    global_df = pd.DataFrame([global_metrics])
    global_df.to_csv(f'{output_filename}_summary.csv', index=False)
    logging.info(f"Saved summary metrics to {output_filename}_summary.csv")

    # Save individual fiber data to another CSV file
    if fiber_props:
        fiber_data = []
        for i, prop in enumerate(fiber_props):
            fiber_data.append({
                'fiber_id': i + 1,
                'length': prop['length'],
                'diameter': prop['diameter'],
                'tortuosity': prop['tortuosity'],
                'azimuth': prop['azimuth'],
                'elevation': prop['elevation'],
                'curvature': prop['curvature']
            })
        fiber_df = pd.DataFrame(fiber_data)
        fiber_df.to_csv(f'{output_filename}_individual_fibers.csv', index=False)
        logging.info(f"Saved individual fiber data to {output_filename}_individual_fibers.csv")


########################################################
#               Overlay TIFF Creation
########################################################

def draw_line_2d(image_slice, y0, x0, y1, x1, color=(255,0,0)):
    """
    Draw a line on a 2D RGB image slice using Bresenham's algorithm.
    image_slice shape: (Y,X,3).
    """
    y0, x0, y1, x1 = int(y0), int(x0), int(y1), int(x1)
    dy = abs(y1 - y0)
    dx = abs(x1 - x0)
    sy = 1 if y0 < y1 else -1
    sx = 1 if x0 < x1 else -1
    err = dx - dy
    while True:
        if 0 <= y0 < image_slice.shape[0] and 0 <= x0 < image_slice.shape[1]:
            image_slice[y0, x0] = color
        if y0 == y1 and x0 == x1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

def save_overlay_tiff_stack(z_stack, traced_fibers, output_filename="fiber_overlay.tif"):
    """
    Creates an RGB overlay of the traced fibers on the original grayscale z-stack.
    traced_fibers: list of Nx3 voxel coords (z,y,x).
    """
    num_slices, height, width = z_stack.shape

    # Normalize original to uint8 if not already
    if z_stack.dtype != np.uint8:
        z_stack_norm = (z_stack - z_stack.min()) / (z_stack.max() - z_stack.min() + 1e-12)
        z_stack_uint8 = (z_stack_norm * 255).astype(np.uint8)
    else:
        z_stack_uint8 = z_stack.copy()

    # Make an RGB stack
    rgb_stack = np.stack([z_stack_uint8]*3, axis=-1)  # (Z, Y, X, 3)

    # Draw each fiber slice-by-slice in red
    for fiber_vox in traced_fibers:
        fiber_rounded = np.round(fiber_vox).astype(int)
        for i in range(len(fiber_rounded) - 1):
            z0, y0, x0 = fiber_rounded[i]
            z1, y1, x1 = fiber_rounded[i+1]
            if z0 == z1 and 0 <= z0 < num_slices:
                draw_line_2d(rgb_stack[z0], y0, x0, y1, x1, color=(255,0,0))

    tiff.imwrite(output_filename, rgb_stack, photometric='rgb')
    logging.info(f"Overlay TIFF saved as {output_filename}.")



########################################################
#         3D Visualization with Equal Axis Scaling
########################################################

def set_axes_equal(ax):
    """
    Sets equal scaling on 3D axes for a matplotlib Axes3D object.
    """
    extents = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
    centers = np.mean(extents, axis=1)
    max_range = np.max(np.abs(extents[:, 1] - extents[:, 0])) / 2.0
    ax.set_xlim(centers[0] - max_range, centers[0] + max_range)
    ax.set_ylim(centers[1] - max_range, centers[1] + max_range)
    ax.set_zlim(centers[2] - max_range, centers[2] + max_range)

def visualize_3d_skeleton(skeleton, sampling):
    """
    Shows the 3D skeleton point cloud in real space (microns).
    skeleton: 3D boolean array (Z,Y,X).
    sampling: (sz, sy, sx).
    """
    logging.info("Visualizing 3D skeleton...")
    coords = np.column_stack(np.where(skeleton > 0))  # Nx3 (z,y,x)
    coords_phys = coords * sampling  # convert to microns
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')

    # reorder to (x, y, z)
    plot_xyz = coords_phys[:, [2,1,0]]
    ax.scatter(plot_xyz[:,0], plot_xyz[:,1], plot_xyz[:,2],
               s=1, c='green', alpha=0.6)
    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Y (µm)")
    ax.set_zlabel("Z (µm)")
    ax.set_xlim(0, ax.get_xlim()[1])
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.set_zlim(0, ax.get_zlim()[1])
    ax.set_title("3D Skeleton Visualization")
    set_axes_equal(ax)
    plt.tight_layout()
    plt.show()
    logging.info("3D skeleton visualization done.")

def visualize_traced_fibers_3d(traced_fibers, dt_image, fiber_segment, sampling,
                               diameter_min, diameter_max):
    """
    3D plot of traced fibers that pass diameter checks, for quick debugging.
    All coords are in voxel space; we convert to microns for plotting.
    """
    logging.info("Visualizing traced fibers in 3D (subset passing diameter checks)...")
    valid_fibers = []
    for fiber_vox in traced_fibers:
        if len(fiber_vox) < 2:
            continue
        # get length in microns
        fiber_scaled = fiber_vox * sampling
        path_length = np.sum(np.linalg.norm(np.diff(fiber_scaled, axis=0), axis=1))

        # compute average diameter from dt_image
        fiber_rounded = np.round(fiber_vox).astype(int)
        fiber_rounded[:,0] = np.clip(fiber_rounded[:,0], 0, dt_image.shape[0]-1)
        fiber_rounded[:,1] = np.clip(fiber_rounded[:,1], 0, dt_image.shape[1]-1)
        fiber_rounded[:,2] = np.clip(fiber_rounded[:,2], 0, dt_image.shape[2]-1)
        radii = dt_image[
            fiber_rounded[:,0],
            fiber_rounded[:,1],
            fiber_rounded[:,2]
        ]
        avg_diam = np.mean(2 * radii)

        # quick filters
        if not (diameter_min <= avg_diam <= diameter_max):
            continue
        if avg_diam >= path_length:
            continue

        valid_fibers.append(fiber_vox)

    if not valid_fibers:
        logging.warning("No valid fibers to display in 3D.")
        return

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')

    for fiber_vox in valid_fibers:
        plot_xyz = (fiber_vox * sampling)[:, [2,1,0]]
        ax.plot(plot_xyz[:,0], plot_xyz[:,1], plot_xyz[:,2],
                marker='o', markersize=2, linewidth=2.5, alpha=0.7)

    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Y (µm)")
    ax.set_zlabel("Z (µm)")
    ax.set_xlim(0, ax.get_xlim()[1])
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.set_zlim(0, ax.get_zlim()[1])
    ax.set_title("Traced Fibers (Filtered by Diameter)")
    set_axes_equal(ax)
    plt.tight_layout()
    plt.show()
    logging.info("Traced fibers visualization done.")

########################################################
#  Side-by-Side Comparison of a Random Slice
########################################################

def display_random_slice_comparison(z_stack, thresh_segment, filtered_segment, overlay_filename):
    """
    Choose a random slice and show:
      1) Original slice
      2) Thresholded (binary) slice
      3) Filtered slice (after morphological ops)
      4) Overlay from saved TIFF
    """
    num_slices = z_stack.shape[0]
    
    if slice_selection is None:
        slice_idx = random.randint(0, num_slices - 1)
    else:
        slice_idx = min(max(0, slice_selection - 1), num_slices - 1)
    logging.info(f"Displaying comparison for slice {slice_idx}")

    # Original
    orig_slice = z_stack[slice_idx]

    # Thresholded
    thresh_slice = thresh_segment[slice_idx]

    # Filtered
    filtered_slice = filtered_segment[slice_idx]

    # Overlay
    overlay_stack = tiff.imread(overlay_filename)
    if slice_idx >= overlay_stack.shape[0]:
        logging.error("Overlay stack has fewer slices than expected.")
        return
    overlay_slice = overlay_stack[slice_idx]

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    axs[0].imshow(orig_slice, cmap='gray')
    axs[0].set_title("Original Slice")
    axs[0].axis("off")

    axs[1].imshow(thresh_slice, cmap='gray')
    axs[1].set_title("Thresholded/Binary")
    axs[1].axis("off")

    axs[2].imshow(filtered_slice, cmap='gray')
    axs[2].set_title("Filtered Slice")
    axs[2].axis("off")

    axs[3].imshow(overlay_slice)
    axs[3].set_title("Overlay Slice")
    axs[3].axis("off")

    plt.tight_layout()
    plt.show()

def plot_azimuth_and_elevation_distributions(fiber_properties, bins=30):
    """
    Histograms of azimuth & elevation using colored bars.
    Azimuth in [0..180] → blue, Elevation in [0..90] → green.
    """
    azimuths = [fp['azimuth'] for fp in fiber_properties if not np.isnan(fp['azimuth'])]
    elevations = [fp['elevation'] for fp in fiber_properties if not np.isnan(fp['elevation'])]

    if not azimuths or not elevations:
        logging.warning("Insufficient data to plot azimuth/elevation.")
        return

    def plot_colored_histogram(values, cmap_name, title, xlabel, vmin, vmax):
        cmap = plt.get_cmap(cmap_name)
        counts, bin_edges = np.histogram(values, bins=bins)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

        fig, ax = plt.subplots(figsize=(8, 6))
        for count, center, left, right in zip(counts, bin_centers, bin_edges[:-1], bin_edges[1:]):
            color = cmap(norm(center))
            ax.bar(left, count, width=right-left, color=color, edgecolor='black', align='edge')

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        plt.tight_layout()
        plt.show()

    # Azimuth: blue gradient
    plot_colored_histogram(
        azimuths,
        cmap_name='Blues',
        title="Azimuth Distribution (0..180)",
        xlabel="Azimuth (degrees)",
        vmin=0,
        vmax=180
    )

    # Elevation: green gradient
    plot_colored_histogram(
        elevations,
        cmap_name='Greens',
        title="Elevation Distribution (0..90)",
        xlabel="Elevation (degrees)",
        vmin=0,
        vmax=90
    )

def plot_diameter_distribution(fiber_properties, bins=30):
    """
    Plots a histogram of fiber diameters using a red color gradient.
    """
    diameters = [fp['diameter'] for fp in fiber_properties
                 if 'diameter' in fp and not np.isnan(fp['diameter'])]

    if not diameters:
        logging.warning("No valid diameter data found for plotting.")
        return

    cmap = plt.get_cmap('Reds')
    counts, bin_edges = np.histogram(diameters, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    norm = plt.Normalize(vmin=min(diameters), vmax=max(diameters))

    fig, ax = plt.subplots(figsize=(8, 6))
    for count, center, left, right in zip(counts, bin_centers, bin_edges[:-1], bin_edges[1:]):
        color = cmap(norm(center))
        ax.bar(left, count, width=right-left, color=color, edgecolor='black', align='edge')

    ax.set_title("Fiber Diameter Distribution")
    ax.set_xlabel("Diameter (µm)")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.show()

def plot_colored_fibers_2d_projection(traced_fibers, fiber_props,
                                      color_by='azimuth',
                                      cmap_name='Blues',
                                      vmin=0,
                                      vmax=180,
                                      title="2D Max Projection Colored by Azimuth (°)"):
    """
    Plot traced fibers in 2D (X-Y projection) colored by azimuth, elevation, or diameter.

    traced_fibers: list of (N_i,3) arrays of voxel coordinates (z,y,x)
    fiber_props: list of dicts from compute_fiber_properties
    color_by: property to color by ('azimuth', 'elevation', or 'diameter')
    cmap_name: colormap name (e.g. 'Blues' for azimuth)
    vmin, vmax: limits for colormap normalization
    title: plot title
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize

    fig, ax = plt.subplots(figsize=(9, 8))

    cmap = cm.get_cmap(cmap_name)
    norm = Normalize(vmin=vmin, vmax=vmax)

    for voxels, props in zip(traced_fibers, fiber_props):
        value = props[color_by]
        color = cmap(norm(value))

        scaled_voxels = voxels * sampling
        X = scaled_voxels[:, 2]  # x
        Y = scaled_voxels[:, 1]  # y

        ax.plot(X, Y, color=color, linewidth=1.5)

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = plt.colorbar(mappable, ax=ax, pad=0.01)
    cbar.set_label(f"{color_by.capitalize()} (°)" if color_by in ['azimuth', 'elevation'] else f"{color_by.capitalize()}")

    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Y (µm)")
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()

########################################################
#                     Main Pipeline
########################################################

def main():
    logging.info("Loading data...")
    try:
        if input_file.endswith('.mat'):
            z_stack = load_mat_any(input_file, mat_variable_name)
            logging.info(f"Loaded '{input_file}' (shape {z_stack.shape}).")
            # These MAT files are already binary (0/1), so:
            thresh_segment = (z_stack > 0).astype(bool)


        elif input_file.endswith('.czi'):
            # 2) LOAD .czi (if needed)
            if czifile is None:
                raise ImportError("czifile not installed.")
            z_stack = czifile.imread(input_file)
            # You may want an adaptive or Otsu threshold depending on intensity:
            thresh_segment = (z_stack > 0).astype(bool)

        else:
            # 3) LOAD TIF or other image
            z_stack = skio.imread(input_file)
            z_stack = np.squeeze(z_stack)
            # (Optional) thresholding
            # Example using adaptive threshold:
            offset_value = -23
            thresh_segment = adaptive_threshold_3d(z_stack, block_size=21, offset=offset_value)

    except Exception as e:
        logging.error(f"Error loading file: {e}")
        sys.exit(1)

    # Ensure shape is (Z, Y, X) & numeric
    z_stack = np.squeeze(z_stack)
    validate_input_data(z_stack)
    logging.info(f"Image shape = {z_stack.shape}, intensity range=({z_stack.min()}, {z_stack.max()})")

    from skimage.morphology import binary_opening, binary_closing, cube
    from skimage.measure import label, regionprops


    def remove_small_objects(binary_image, min_size=50):
        labeled = label(binary_image)
        filtered = np.zeros_like(binary_image)
        for region in regionprops(labeled):
            if region.area >= min_size:  # Keep only large-enough objects
                filtered[labeled == region.label] = 1
        return filtered

    filtered_segment = remove_small_objects(thresh_segment, min_size=3)

    # se = ball(1)
    # filtered_segment = binary_opening(filtered_segment, se)
    
    # filtered_segment = binary_closing(filtered_segment, se)

    # filtered_segment = thresh_segment
    # filtered_segment = binary_closing(filtered_segment, ball(1))
    # filtered_segment = binary_dilation(filtered_segment, ball(1))


    volume_fraction = np.sum(filtered_segment) / filtered_segment.size
    logging.info(f"Volume Fraction is {volume_fraction}")


    # Skeletonize
    skeleton = skeletonize(filtered_segment)

    # Distance transform. dt_image has the same shape,
    # but the *values* are in microns because sampling is passed.
    dt_image = distance_transform_edt(filtered_segment, sampling=sampling)
    mean_rad = compute_mean_fiber_radius(dt_image)
    logging.info(f"Mean radius in DT= {mean_rad:.2f} µm; mean diameter= {2*mean_rad:.2f} µm")

    # Segment skeleton -> paths
    coords = np.column_stack(np.where(skeleton))  # (N,3) voxel coords
    paths = segment_skeleton(skeleton, coords, angle_threshold_degs=70)

    if len(paths) > max_fibers_to_process:
        logging.info(f"Limiting initial paths to {max_fibers_to_process} out of {len(paths)} total")
        paths = paths[:max_fibers_to_process]

    # Compute a tortuosity threshold
    tort_threshold = compute_initial_tortuosity(coords, paths, sampling)
    poly_tracks = transform_to_polygonal_tracks(coords, paths, tort_threshold, sampling)

    # Preliminary track properties
    track_props = [
        compute_single_track_properties(trk, dt_image, filtered_segment, sampling)
        for trk in poly_tracks
    ]

    # Fiber merging/tracing in real space (microns)
    traced_fibers = fiber_tracing(poly_tracks, track_props, sampling,
                                  max_distance=10.0,  # 8 microns search radius
                                  angle_thresh=70.0)
    logging.info(f"Traced {len(traced_fibers)} fibers after merging.")

    # Save overlay
    save_overlay_tiff_stack(z_stack, traced_fibers, output_filename=overlay_filename)

    # Final fiber properties
    fiber_props = compute_fiber_properties(
        traced_fibers, dt_image, filtered_segment, sampling,
        dia_min=diameter_threshold_min,
        dia_max=diameter_threshold_max
    )

    # Plot angle distributions
    plot_azimuth_and_elevation_distributions(fiber_props)
    plot_diameter_distribution(fiber_props)

    # Visualizations
    visualize_3d_skeleton(skeleton, sampling)
    visualize_traced_fibers_3d(
        traced_fibers, dt_image, filtered_segment, sampling,
        diameter_threshold_min, diameter_threshold_max
    )
    coords_all, az_all, el_all, diam_all = [], [], [], []

    for vox, prop in zip(traced_fibers, fiber_props):
        n = len(vox)
        coords_all.append(vox * sampling)                                   # (n,3)
        az_all.append( np.full(n, prop['azimuth']) )
        el_all.append( np.full(n, prop['elevation']) )
        # pick local vs average diameter – here: average
        diam_all.append( np.full(n, prop['diameter']) )

    coords_all = np.vstack(coords_all)
    az_all     = np.concatenate(az_all)
    el_all     = np.concatenate(el_all)
    diam_all   = np.concatenate(diam_all)

    print(coords_all.shape, az_all.shape)

    blues_trunc  = truncate_colormap('Blues',  minval=0.2, maxval=1.0)
    greens_trunc = truncate_colormap('Greens', minval=0.2, maxval=1.0)
    reds_trunc   = truncate_colormap('Reds',   minval=0.2, maxval=1.0)


    plot_colored_fibers_2d_projection(
        traced_fibers,
        fiber_props,
        color_by='azimuth',
        cmap_name= blues_trunc,  # blue gradient for azimuth
        vmin=0,
        vmax=180,
        title="2D Max Projection Colored by Azimuth (°)"
    )

    plot_colored_fibers(
        traced_fibers,
        fiber_props,
        color_by='elevation',
        cmap_name=greens_trunc,
        vmin=0,
        vmax=90,
        title="Skeleton colored by Elevation (°)"
    )

    plot_colored_fibers(
        traced_fibers,
        fiber_props,
        color_by='diameter',
        cmap_name=reds_trunc,
        vmin=min(fp['diameter'] for fp in fiber_props),
        vmax=max(fp['diameter'] for fp in fiber_props),
        title="Skeleton colored by Diameter (µm)"
    )
    # output_base = os.path.splitext(input_file)[0]  # Use input filename as base
    output_base = "fiber_stats" # Use input filename as base
    save_fiber_analysis_to_csv(
        output_base,
        z_stack,
        volume_fraction,
        mean_rad,
        tort_threshold,
        traced_fibers,
        fiber_props
    )
    # Display random slice comparison (4 distinct images)
    display_random_slice_comparison(z_stack, thresh_segment, filtered_segment, overlay_filename)

    logging.info("Processing complete.")

if __name__ == "__main__":
    main()
