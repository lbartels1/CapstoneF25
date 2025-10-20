import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import time

camera_mtx_path = r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Calibration_Data\camera_mtx.txt"
camera_dist_path = r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Calibration_Data\camera_dist.txt"
image_points_path = r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Calibration_Data\image_points.npy"
world_points_path = r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Calibration_Data\world_points.npy"
cam_rvec_path = r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Calibration_Data\camera_rvec.npy"
cam_tvec_path = r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Calibration_Data\camera_tvec.npy"

map_path = r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Perimeter\map.pgm"
map_scale_path = r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Perimeter\map_scale.txt"

""" Class which contains map data and scale information"""
class Map:
    def __init__(self, map_path, map_scale_path):
        self.map_image = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        if self.map_image is None:
            raise FileNotFoundError(f"Map image not found: {map_path}")
        
        with open(map_scale_path, 'r') as f:
            self.scale = float(f.read().strip())  # meters per pixel

""" Class to update and store robot pose estimation"""
class Robot:
    def __init__(self):
        self.position = np.array([0.0, 0.0])  # x, y in meters
        self.orientation = 0.0  # theta in radians


""" Uses map and robot classes to navigate the robot """
class Navigator:
    def __init__(self, map_data, robot):
        self.map = map_data
        self.robot = robot
