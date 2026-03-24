# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Dental Image Preprocessing Module for Stage2 Lesion Detection.

This module transforms Stage1 segmentation masks into multi-channel tensors
that provide rich prior information for the YOLO11-Dental detector.

Input Channels:
    Ch0: I_raw - Original grayscale image (normalized)
    Ch1: M_virtual - Virtual fusion mask (complete tooth shape)
    Ch2: DistanceMap - Distance transform of M_virtual (core/boundary guidance)
    Ch3: M_metal - Metal prior mask (implant + prosthesis = high-risk regions)
"""

import cv2
import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects, h_maxima, label


class DentalStage1Processor:
    """
    Processes Stage1 VM-UNet segmentation output into 4-channel input tensor for Stage2 YOLO.

    Stage1 classes (0-4):
        0: background
        1: normal_teeth (healthy)
        2: defect_teeth (disease)
        3: implant (metal)
        4: prosthesis (metal)
    """

    def __init__(self,
                 normalize_distance=True,
                 use_watershed=True,
                 min_tooth_area=100,
                 distance_sigma=1.5,
                 marker_threshold=8.0,
                 marker_min_area=50,
                 marker_h_value=3.0):
        """
        Initialize the processor.

        Args:
            normalize_distance: Whether to normalize distance transform to [0, 1]
            use_watershed: Whether to use watershed for instance separation
            min_tooth_area: Minimum area threshold for valid tooth regions
            distance_sigma: Gaussian smoothing sigma for distance transform (before watershed)
            marker_threshold: Absolute distance threshold (pixels) for initial marker detection.
                              Points must be at least this far from boundary to be considered.
            marker_min_area: Minimum area (pixels) for valid marker regions.
                             Removes small noise markers.
            marker_h_value: Height parameter for h_maxima. Only keeps local maxima that are
                            at least h_value higher than surrounding. Higher = fewer markers.
        """
        self.normalize_distance = normalize_distance
        self.use_watershed = use_watershed
        self.min_tooth_area = min_tooth_area
        self.distance_sigma = distance_sigma
        self.marker_threshold = marker_threshold
        self.marker_min_area = marker_min_area
        self.marker_h_value = marker_h_value

    def __call__(self, raw_image: np.ndarray, stage1_mask: np.ndarray) -> np.ndarray:
        """
        Process raw image and Stage1 mask into 4-channel tensor.

        Args:
            raw_image: Original image, shape (H, W) or (H, W, 3)
            stage1_mask: Stage1 segmentation mask with class indices 0-4, shape (H, W)

        Returns:
            4-channel tensor, shape (H, W, 4) with dtype float32, values in [0, 1]
        """
        # Ensure grayscale
        if len(raw_image.shape) == 3:
            raw_gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
        else:
            raw_gray = raw_image.copy()

        # Normalize raw image to [0, 1]
        ch0_raw = raw_gray.astype(np.float32) / 255.0

        # Extract class masks from Stage1 output
        m_healthy = (stage1_mask == 1).astype(np.uint8)    # normal_teeth
        m_disease = (stage1_mask == 2).astype(np.uint8)    # defect_teeth
        m_implant = (stage1_mask == 3).astype(np.uint8)    # implant
        m_prosthesis = (stage1_mask == 4).astype(np.uint8)  # prosthesis

        # Step 1: Virtual Fusion - Merge all tooth regions
        m_virtual = self._virtual_fusion(m_healthy, m_disease, m_implant, m_prosthesis)
        ch1_virtual = m_virtual.astype(np.float32)

        # Step 2: Distance Transform with optional instance separation
        ch2_distance = self._compute_distance_field(m_virtual)

        # Step 3: Metal Prior - High-risk regions (implant + prosthesis)
        m_metal = np.logical_or(m_implant, m_prosthesis).astype(np.float32)
        ch3_metal = m_metal

        # Stack into 4-channel tensor
        tensor_4ch = np.stack([ch0_raw, ch1_virtual, ch2_distance, ch3_metal], axis=-1)

        return tensor_4ch

    def _virtual_fusion(self,
                        m_healthy: np.ndarray,
                        m_disease: np.ndarray,
                        m_implant: np.ndarray,
                        m_prosthesis: np.ndarray) -> np.ndarray:
        """
        Virtual fusion: Merge all tooth-related masks to form complete tooth shape.
        This fills gaps caused by metal artifacts.

        Args:
            m_healthy: Normal teeth mask
            m_disease: Defective teeth mask
            m_implant: Implant mask
            m_prosthesis: Prosthesis mask

        Returns:
            Fused binary mask representing complete tooth regions
        """
        # Union of all tooth regions
        m_fused = np.logical_or.reduce([m_healthy, m_disease, m_implant, m_prosthesis])
        m_fused = m_fused.astype(np.uint8)

        # Morphological closing to fill small gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        m_fused = cv2.morphologyEx(m_fused, cv2.MORPH_CLOSE, kernel)

        # Remove small noise regions
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m_fused, connectivity=8)
        m_clean = np.zeros_like(m_fused)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= self.min_tooth_area:
                m_clean[labels == i] = 1

        return m_clean

    def _compute_distance_field(self, m_virtual: np.ndarray) -> np.ndarray:
        """
        Compute distance transform field for tooth regions.
        Optionally performs watershed-based instance separation.

        The distance field implicitly encodes:
        - Core regions (high values near center)
        - Boundary regions (low values near edges)

        Args:
            m_virtual: Binary mask of fused tooth regions

        Returns:
            Normalized distance field, values in [0, 1]
        """
        if m_virtual.sum() == 0:
            return np.zeros_like(m_virtual, dtype=np.float32)

        # Compute distance transform (raw pixel distances)
        distance = ndimage.distance_transform_edt(m_virtual)

        if self.use_watershed:
            # Instance separation using watershed
            # This returns PER-INSTANCE normalized values [0, 1]
            # Each tooth instance is normalized independently, which:
            # - Ensures consistent representation regardless of tooth size
            # - Front teeth and molars both have apex at ~1.0, boundary at ~0.0
            # Note: Gaussian smoothing is done inside _watershed_separation
            distance = self._watershed_separation(m_virtual, distance)
        else:
            # No watershed: apply smoothing and global normalization
            # WARNING: Global normalization makes values dependent on image content
            if self.distance_sigma > 0:
                distance = ndimage.gaussian_filter(distance, sigma=self.distance_sigma)
            if self.normalize_distance and distance.max() > 0:
                distance = distance / distance.max()

        return distance.astype(np.float32)

    def _watershed_separation(self,
                              m_virtual: np.ndarray,
                              distance: np.ndarray) -> np.ndarray:
        """
        Use watershed algorithm to separate touching teeth instances.
        Returns per-instance normalized distance field.

        Standard morphological watershed pipeline:
        1. Smooth distance transform
        2. Threshold to get initial marker regions
        3. Remove small objects (noise)
        4. h_maxima to keep only significant local maxima
        5. Label connected components as markers
        6. Watershed segmentation
        7. Per-instance normalization

        Args:
            m_virtual: Binary mask of tooth regions
            distance: Distance transform of mask

        Returns:
            Instance-aware distance field with per-instance normalization
        """
        # Step 1: Smooth distance transform for stability
        d_smooth = ndimage.gaussian_filter(distance, sigma=self.distance_sigma)

        # Step 2: Absolute threshold - pixels at least marker_threshold from boundary
        # This is stable across different tooth sizes (front teeth vs molars)
        markers = d_smooth > self.marker_threshold

        # Step 3: Remove small marker regions (noise)
        markers = remove_small_objects(markers, min_size=self.marker_min_area)

        # Step 4: h_maxima - keep only "significant" local maxima
        # A maximum is significant if it's at least h_value higher than surrounding
        # This prevents multiple markers in one tooth due to noise/texture
        if self.marker_h_value > 0:
            h_max = h_maxima(d_smooth, h=self.marker_h_value)
            markers = markers & (h_max > 0)

        # Step 5: Label connected marker regions
        markers_labeled = label(markers)

        if markers_labeled.max() == 0:
            # No valid markers found, return original distance (normalized)
            if distance.max() > 0:
                return (distance / distance.max()).astype(np.float32)
            return distance.astype(np.float32)

        # Step 6: Watershed segmentation
        # Use negative distance so watershed flows from high (center) to low (boundary)
        labels = watershed(-d_smooth, markers_labeled, mask=m_virtual)

        # Step 7: Compute per-instance normalized distance field
        # Each tooth instance is normalized to [0, 1] independently
        # This ensures consistent representation regardless of tooth size
        result = np.zeros_like(distance, dtype=np.float32)
        for label_id in range(1, labels.max() + 1):
            instance_mask = (labels == label_id).astype(np.uint8)
            if instance_mask.sum() > 0:
                instance_dist = ndimage.distance_transform_edt(instance_mask)
                if instance_dist.max() > 0:
                    instance_dist = instance_dist / instance_dist.max()
                result[instance_mask > 0] = instance_dist[instance_mask > 0]

        return result


class DentalPreprocessTransform:
    """
    Transform wrapper for use in data augmentation pipeline.
    Applies DentalStage1Processor to image-mask pairs.
    """

    def __init__(self, **processor_kwargs):
        """Initialize with processor configuration."""
        self.processor = DentalStage1Processor(**processor_kwargs)

    def __call__(self, image: np.ndarray, stage1_mask: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing transform.

        Args:
            image: Raw input image
            stage1_mask: Stage1 segmentation mask

        Returns:
            4-channel preprocessed tensor
        """
        return self.processor(image, stage1_mask)


def visualize_4ch_tensor(tensor_4ch: np.ndarray, save_path: str = None):
    """
    Visualize the 4-channel tensor for debugging.

    Args:
        tensor_4ch: 4-channel tensor from processor, shape (H, W, 4)
        save_path: Optional path to save visualization
    """
    import matplotlib.pyplot as plt

    channel_names = ['Raw Image', 'Virtual Fusion', 'Distance Field', 'Metal Prior']

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for i, (ax, name) in enumerate(zip(axes, channel_names)):
        ax.imshow(tensor_4ch[:, :, i], cmap='gray' if i == 0 else 'jet')
        ax.set_title(name)
        ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# Convenience function for quick testing
def process_dental_image(raw_image_path: str,
                         stage1_mask_path: str,
                         output_path: str = None) -> np.ndarray:
    """
    Quick function to process a single dental image.

    Args:
        raw_image_path: Path to raw dental X-ray image
        stage1_mask_path: Path to Stage1 segmentation mask (class indices 0-4)
        output_path: Optional path to save visualization

    Returns:
        4-channel preprocessed tensor
    """
    raw_image = cv2.imread(raw_image_path)
    stage1_mask = cv2.imread(stage1_mask_path, cv2.IMREAD_GRAYSCALE)

    processor = DentalStage1Processor()
    tensor_4ch = processor(raw_image, stage1_mask)

    if output_path:
        visualize_4ch_tensor(tensor_4ch, output_path)

    return tensor_4ch
