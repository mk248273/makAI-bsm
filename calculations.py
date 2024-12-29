import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras import preprocessing

def euclidean_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

def convert_to_real_measurements(pixel_measurement, pixel_height, real_height_cm):
    height_ratio = real_height_cm / pixel_height
    return pixel_measurement * height_ratio

def measure_body_sizes(side_colored_mask, front_colored_mask, sideposes, frontposes, real_height_cm, rainbow):
    """Measure various body sizes based on detected poses."""
    measurements = []
    
    for pose in frontposes:
        # Assuming each `pose` is a dictionary with 'keypoints' that are already in the required format
        keypoints = pose[0]  # This should directly give us the dictionary

        # Extract positions directly from keypoints
        left_eye = keypoints[1].position
        right_eye = keypoints[2].position
        nose = keypoints[3].position
        right_ear = keypoints[4].position
        left_shoulder = keypoints[5].position
        right_shoulder = keypoints[6].position
        left_elbow = keypoints[7].position
        right_elbow = keypoints[8].position
        left_wrist = keypoints[9].position
        right_wrist = keypoints[10].position
        left_hip = keypoints[11].position
        right_hip = keypoints[12].position
        left_knee = keypoints[13].position
        right_knee = keypoints[14].position
        left_ankle = keypoints[15].position
        right_ankle = keypoints[16].position
        
        # Calculate pixel height (from the top of the head to the bottom of the ankle)
        pixel_height = euclidean_distance((left_eye.x, left_eye.y), (left_ankle.x, left_ankle.y))
        
        
        shoulder_width_cm = convert_to_real_measurements(
            euclidean_distance((left_shoulder.x, left_shoulder.y),(right_shoulder.x, right_shoulder.y)),
            pixel_height, real_height_cm
        )

        # arm_length_cm = convert_to_real_measurements(
        #     euclidean_distance((right_shoulder.x, right_shoulder.y), (right_elbow.x, right_elbow.y)),
        #     pixel_height, real_height_cm
        # ) + convert_to_real_measurements(
        #     euclidean_distance((right_elbow.x, right_elbow.y), (right_wrist.x, right_wrist.y)),
        #     pixel_height, real_height_cm
        # )
        
        # leg_length_cm = convert_to_real_measurements(
        #     euclidean_distance((left_hip.x, left_hip.y), (left_knee.x, left_knee.y)),
        #     pixel_height, real_height_cm
        # ) + convert_to_real_measurements(
        #     euclidean_distance((left_knee.x, left_knee.y), (left_ankle.x, left_ankle.y)),
        #     pixel_height, real_height_cm
        # )
        
        arm_length_cm = convert_to_real_measurements(
            euclidean_distance((left_shoulder.x, left_shoulder.y), (left_wrist.x, left_wrist.y)),
            pixel_height, real_height_cm
        ) 
        
        leg_length_cm = convert_to_real_measurements(
            euclidean_distance((left_hip.x, left_hip.y), (left_ankle.x, right_ankle.y)),
            pixel_height, real_height_cm
        )

        shoulder_to_waist_cm = convert_to_real_measurements(
            euclidean_distance((left_shoulder.x, left_shoulder.y), (left_hip.x, left_hip.y)),
            pixel_height, real_height_cm
        )

        # Calculate waist circumference using the ellipse circumference formula
        a = euclidean_distance((left_hip.x, left_hip.y), (right_hip.x, right_hip.y)) / 2
        # b = euclidean_distance((), ()) / 2

        # Use Ramanujan's approximation for the circumference of an ellipse
        # waist_circumference_px = math.pi * (3*(a + b) - math.sqrt((3*a + b)*(a + 3*b)))
        waist_circumference_cm = 90 #convert_to_real_measurements(waist_circumference_px, pixel_height, real_height_cm)

        
        # Convert pixel measurements to real measurements using the height ratio
        measurements.append({
            "height_cm": real_height_cm,
            "arm_length_cm":  arm_length_cm,
            "shoulder_to_waist_cm": shoulder_to_waist_cm,
            "shoulder_width_cm": shoulder_width_cm,
            "leg_length_cm": leg_length_cm,
        })

    return measurements
