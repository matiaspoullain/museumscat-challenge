import cv2
import numpy as np

# Furniture exclusion: x-coordinate of the fixed vertical bar (logo/color-chart/
# barcode cards), measured via EDA (tables_figures/eda/layout_notes.md).
FURNITURE_X = 2200

BRIGHTNESS_THRESHOLD = 140
OPEN_KERNEL = 21
PADDING = 30


def find_search_roi(image_path):
    """Return (x1, y1, x2, y2) bounding the mount surface where label cards
    are pinned, excluding the fixed furniture column and the specimen area.

    The L-shaped mount (furniture bar + card-bearing bar) is one physically
    connected bright blob against a darker background, easily separated by a
    plain brightness threshold (unlike individual cards vs. the mount surface,
    which have near-identical brightness/texture — see layout_notes.md for why
    per-card detection was abandoned in favor of this coarser ROI). The
    specimen's small pinned card is a separate, disconnected bright blob and
    is excluded by only keeping the largest connected component.
    """
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((OPEN_KERNEL, OPEN_KERNEL), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return (FURNITURE_X, 0, w, h)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mount = labels == largest
    mount[:, :FURNITURE_X] = False

    ys, xs = np.where(mount)
    if len(xs) == 0:
        return (FURNITURE_X, 0, w, h)

    x1 = max(FURNITURE_X, int(xs.min()) - PADDING)
    y1 = max(0, int(ys.min()) - PADDING)
    x2 = min(w, int(xs.max()) + PADDING)
    y2 = min(h, int(ys.max()) + PADDING)
    return (x1, y1, x2, y2)
