import cv2
import argparse
import os

ARUCO_DICT = cv2.aruco.DICT_5X5_250
SQUARES_VERTICALLY = 14
SQUARES_HORIZONTALLY = 9
SQUARE_LENGTH = 0.04
MARKER_LENGTH = 0.03

# based on https://medium.com/@ed.twomey1/using-charuco-boards-in-opencv-237d8bc9e40d

class SimpleCalibrator:
    def __init__(self, image_path):
        self.image_path = image_path
        
        self.images = []
        for image in os.listdir(self.image_path):
            self.images.append(cv2.imread(f"{self.image_path}/{image}"))
        
    def calibrate(self):
        dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)

        all_charuco_corners = []
        all_charuco_ids = []

        for image in self.images:
            image_copy = image.copy()
            marker_corners, marker_ids, rejected_corners = cv2.aruco.detectMarkers(image, dictionary)
            marker_corners, marker_ids, _, _ = cv2.aruco.refineDetectedMarkers(image, board, marker_corners, marker_ids, rejected_corners)

            if len(marker_ids) > 0:
                cv2.aruco.drawDetectedMarkers(image_copy, marker_corners, marker_ids)
                charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, image, board)
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)
        
        retval, K, D, _, _ = cv2.aruco.calibrateCameraCharuco(all_charuco_corners, all_charuco_ids, board, image.shape[:2], None, None)

        print("Result:", retval, K, D)

        for image in self.images:
            undistorted_image = cv2.undistort(image, K, D)
            cv2.imshow('Undistorted Image', undistorted_image)
            cv2.waitKey(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    args = parser.parse_args()
    calibrator = SimpleCalibrator(args.input_dir)
    calibrator.calibrate()