import cv2
import numpy as np

# DaSSCo card color signature (HSV) — solid saturated blue, easily separable
# from the blue-grey background (sat ~59) and white/speckled mount (sat ~18).
# See tables_figures/eda/dassco_anchor_test/notes.md.
DASSCO_HSV_LOW = (95, 95, 40)
DASSCO_HSV_HIGH = (130, 255, 255)
DASSCO_OPEN_KERNEL = 15

# Furniture-column cut, calibrated as a multiple of the DaSSCo card's own
# detected width (real, per-image size — not a fixed pixel constant, which
# broke on images where the mount is shifted in-frame; see notes.md).
FURNITURE_X_RATIO = 1.21

BRIGHTNESS_THRESHOLD = 140
OPEN_KERNEL = 21
PADDING = -50

# Below this, don't bother rotating — noise in the DaSSCo blob's own edges
# produces angle jitter of this order even on a perfectly straight photo.
MIN_ROTATION_DEG = 0.05


def _dassco_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, DASSCO_HSV_LOW, DASSCO_HSV_HIGH)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((DASSCO_OPEN_KERNEL, DASSCO_OPEN_KERNEL), np.uint8))


def _dassco_bbox(img):
    mask = _dassco_mask(img)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    x, y, w, h, _ = stats[largest]
    return (x, y, w, h)


def _dassco_rotation_angle(img):
    """Degrees to rotate `img` (cv2.getRotationMatrix2D convention, positive
    = counterclockwise) so the DaSSCo card - and with it, the whole rigid
    mount - is horizontal. Calibrated empirically: applying a rotation R to a
    straight image shifts the measured angle by -R, so undoing a measured
    angle A means applying a rotation of A itself (verified against synthetic
    +/-3 degree test rotations, see tables_figures/eda/dassco_anchor_test/notes.md).
    """
    mask = _dassco_mask(img)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return 0.0
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    ys, xs = np.where(labels == largest)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    _, _, angle = cv2.minAreaRect(pts)
    while angle <= -45:
        angle += 90
    while angle > 45:
        angle -= 90
    return angle


def _deskew(img):
    """Return (rotated_img, angle_applied). No-op below MIN_ROTATION_DEG."""
    angle = _dassco_rotation_angle(img)
    if abs(angle) < MIN_ROTATION_DEG:
        return img, 0.0
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    return rotated, angle


def find_search_roi(image_path):
    """Return (angle, (x1, y1, x2, y2)): the degrees to rotate the original
    image by (cv2.getRotationMatrix2D convention) to straighten the mount,
    and the ROI box valid on that rotated image — bounding the mount surface
    where label cards are pinned, furniture column and specimen area excluded.

    Rotation and the furniture-column cut are both derived from the DaSSCo
    card's detection (color-based, per-image), so both stay correct when the
    mount is shifted or tilted in-frame. See
    tables_figures/eda/dassco_anchor_test/notes.md for why.
    """
    img = cv2.imread(image_path)
    img, angle = _deskew(img)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    dassco = _dassco_bbox(img)
    # Two different cuts from the same anchor: `safe_cut` is deliberately
    # generous so the connected-component search below never re-touches the
    # furniture bar (see notes.md — undershooting the true corner by even
    # 30-100px reconnects to it). `report_x1` is the DaSSco card's own right
    # edge, snug against it, so cards pinned close to the corner don't get
    # clipped — the extra strip between the two is blank mount surface,
    # already known to be safe to include (Task 6's OCR won't find text there).
    safe_cut = int(dassco[0] + FURNITURE_X_RATIO * dassco[2]) if dassco else w // 3
    report_x1 = (dassco[0] + dassco[2]) if dassco else safe_cut

    _, mask = cv2.threshold(gray, BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((OPEN_KERNEL, OPEN_KERNEL), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return angle, (report_x1, 0, w, h)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mount = labels == largest
    mount[:, :safe_cut] = False

    ys, xs = np.where(mount)
    if len(xs) == 0:
        return angle, (report_x1, 0, w, h)

    x1 = max(0, report_x1 - PADDING)
    y1 = max(0, int(ys.min()) - PADDING)
    x2 = min(w, int(xs.max()) + PADDING)
    y2 = min(h, int(ys.max()) + PADDING)
    return angle, (x1, y1, x2, y2)


def get_search_crop(image_path):
    """Deskew + crop in one step: the actual image data ready for OCR."""
    img = cv2.imread(image_path)
    img, _ = _deskew(img)
    _, (x1, y1, x2, y2) = find_search_roi(image_path)
    return img[y1:y2, x1:x2]
