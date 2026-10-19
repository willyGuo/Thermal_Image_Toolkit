"""OpenCV availability shared by image and camera modules."""

try:
    import cv2
except ImportError:
    cv2 = None
