import cv2
import numpy as np
from typing import Tuple, List
from abc import ABCMeta, abstractmethod


class KeyPoint(metaclass=ABCMeta):
    @property
    @abstractmethod
    def pt(self) -> Tuple[float, float]:
        return NotImplemented

    @property
    @abstractmethod
    def size(self) -> float:
        return NotImplemented

    @property
    @abstractmethod
    def angle(self) -> float:
        return NotImplemented

    @property
    @abstractmethod
    def response(self) -> float:
        return NotImplemented

    @property
    @abstractmethod
    def octave(self) -> int:
        return NotImplemented


class SIFT:
    def __init__(self):
        self.cv2sift = cv2.SIFT_create()

    def __call__(self, img: cv2.Mat) -> Tuple[List[KeyPoint], List[np.ndarray]]:
        return self.cv2sift.detectAndCompute(img, None)

    def detect(self, img: cv2.Mat) -> List[KeyPoint]:
        return self.cv2sift.detect(img, None)

    def compute(self, img: cv2.Mat, kp: List[KeyPoint]) -> Tuple[List[KeyPoint], List[np.ndarray]]:
        return self.cv2sift.compute(img, kp)
