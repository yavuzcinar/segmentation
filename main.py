import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


def imshow(img, title="Image", ax=None):
    """Display image using matplotlib (handles BGR or grayscale)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    if img.ndim == 2:
        ax.imshow(img, cmap="gray")
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
    ax.set_title(title)
    ax.axis("off")
    return ax


def watershed_segmentation(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)
    img_watershed = img.copy()
    img_watershed[markers == -1] = [0, 0, 255]
    return img_watershed, markers


def grabcut_segmentation(img):
    mask = np.zeros(img.shape[:2], np.uint8)
    height, width = img.shape[:2]
    margin = max(10, int(min(height, width) * 0.08))
    rect = (margin, margin, width - 2 * margin, height - 2 * margin)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    img_grabcut = img * mask2[:, :, np.newaxis]
    return img_grabcut, mask2


def color_range_segmentation(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    img_green = cv2.bitwise_and(img, img, mask=mask_green)

    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    img_red = cv2.bitwise_and(img, img, mask=mask_red)

    return img_green, mask_green, img_red


def kmeans_segmentation(img, k=5):
    pixel_values = img.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    img_kmeans = segmented_data.reshape(img.shape)

    kmeans_sklearn = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_sklearn = kmeans_sklearn.fit_predict(img.reshape((-1, 3)))
    segmented_sklearn = kmeans_sklearn.cluster_centers_[labels_sklearn]
    segmented_sklearn = segmented_sklearn.reshape(img.shape).astype(np.uint8)

    return img_kmeans, segmented_sklearn


def main():
    parser = argparse.ArgumentParser(description="Lab 4: Classical image segmentation")
    parser.add_argument("--image", default="flower1.jpg", help="Path to input image")
    parser.add_argument("--k", type=int, default=5, help="Number of clusters for k-means")
    args = parser.parse_args()

    img_path = Path(args.image)
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Error: Could not load image from {img_path}")
        print("Make sure the image exists in the project directory or pass --image.")
        return

    print(f"Image loaded successfully: {img.shape}")
    fig = plt.figure(figsize=(20, 12))

    ax1 = plt.subplot(3, 3, 1)
    imshow(img, "Original Image", ax1)

    print("\n=== Watershed Segmentation ===")
    img_watershed, _ = watershed_segmentation(img)
    ax2 = plt.subplot(3, 3, 2)
    imshow(img_watershed, "Watershed Segmentation", ax2)

    print("\n=== GrabCut Segmentation ===")
    img_grabcut, mask_grabcut = grabcut_segmentation(img)
    ax3 = plt.subplot(3, 3, 3)
    imshow(img_grabcut, "GrabCut Segmentation", ax3)

    print("\n=== Color-Range Segmentation ===")
    img_green, mask_green, img_red = color_range_segmentation(img)
    ax4 = plt.subplot(3, 3, 4)
    imshow(img_green, "Color-Range Segmentation (Green)", ax4)
    ax5 = plt.subplot(3, 3, 5)
    imshow(img_red, "Color-Range Segmentation (Red)", ax5)

    print("\n=== K-Means Clustering Segmentation ===")
    img_kmeans, img_kmeans_sklearn = kmeans_segmentation(img, k=args.k)
    ax6 = plt.subplot(3, 3, 6)
    imshow(img_kmeans, f"K-Means Segmentation (k={args.k})", ax6)
    ax7 = plt.subplot(3, 3, 7)
    imshow(img_kmeans_sklearn, f"K-Means (sklearn) Segmentation (k={args.k})", ax7)

    ax8 = plt.subplot(3, 3, 8)
    ax8.imshow(mask_grabcut, cmap="gray")
    ax8.set_title("GrabCut Mask")
    ax8.axis("off")

    ax9 = plt.subplot(3, 3, 9)
    ax9.imshow(mask_green, cmap="gray")
    ax9.set_title("Color-Range Mask (Green)")
    ax9.axis("off")

    plt.tight_layout()
    plt.savefig("segmentation_results.png", dpi=150, bbox_inches="tight")
    print("\nResults saved to 'segmentation_results.png'")
    plt.show()

    print("\n=== Segmentation Complete ===")
    print("\nSummary:")
    print("1. Watershed: Distance transform + markers")
    print("2. GrabCut: Graph-cut foreground extraction")
    print("3. Color-Range: HSV thresholding")
    print("4. K-Means: Pixel clustering by color")


if __name__ == "__main__":
    main()
