import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
import tf_bodypix
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths
from tf_bodypix.draw import draw_poses
from tensorflow.keras import preprocessing
import cv2
import sys

import json
from matplotlib import pyplot as plt
import numpy as np
from calculations import measure_body_sizes
import gradio as gr
import pandas as pd

# Load BodyPix model
bodypix_model = load_model(download_model(BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16))

rainbow = [
  [110, 64, 170], [143, 61, 178], [178, 60, 178], [210, 62, 167],
  [238, 67, 149], [255, 78, 125], [255, 94, 99],  [255, 115, 75],
  [255, 140, 56], [239, 167, 47], [217, 194, 49], [194, 219, 64],
  [175, 240, 91], [135, 245, 87], [96, 247, 96],  [64, 243, 115],
  [40, 234, 141], [28, 219, 169], [26, 199, 194], [33, 176, 213],
  [47, 150, 224], [65, 125, 224], [84, 101, 214], [99, 81, 195]
]

def process_images(front_img, side_img, real_height_cm):
    fimage_array = preprocessing.image.img_to_array(front_img)
    simage_array = preprocessing.image.img_to_array(side_img)
    
    # bodypix prediction
    frontresult = bodypix_model.predict_single(fimage_array)
    sideresult = bodypix_model.predict_single(simage_array)
    
    front_mask = frontresult.get_mask(threshold=0.75)
    side_mask = sideresult.get_mask(threshold=0.75)
    
    front_colored_mask = frontresult.get_colored_part_mask(front_mask, rainbow)
    side_colored_mask = sideresult.get_colored_part_mask(side_mask, rainbow)
    
    frontposes = frontresult.get_poses()
    front_image_with_poses = draw_poses(
        fimage_array.copy(),
        frontposes,
        keypoints_color=(255, 100, 100),
        skeleton_color=(100, 100, 255)
    )
    
    sideposes = sideresult.get_poses()
    side_image_with_poses = draw_poses(
        simage_array.copy(),
        sideposes,
        keypoints_color=(255, 100, 100),
        skeleton_color=(100, 100, 255)
    )
    
    body_sizes = measure_body_sizes(side_colored_mask, front_colored_mask, sideposes, frontposes, real_height_cm, rainbow)
    measurements_df = pd.DataFrame([body_sizes]) if isinstance(body_sizes, dict) else pd.DataFrame(body_sizes)
    csv_file = "Body-measurement.csv"
    if not os.path.exists(csv_file):
        # Save as a new file if it doesn't exist
        measurements_df.to_csv(csv_file, index=False)
    else:
        # Append to the existing file
        measurements_df.to_csv(csv_file, mode='a', header=False, index=False)
        
    return "Your body part measured and saved to the database."

def create_interface():
    system = BodyMeasurementSystem()

    with gr.Blocks(title="Body Sizing System") as interface:
        gr.Markdown("# Body Sizing System")
        gr.Markdown("""
        ## Instructions:
        1. Click the camera icon in each image box to access your device camera
        2. Capture your front and side poses
        3. Enter your height in centimeters
        4. Click Process Images
        """)

        with gr.Row():
            with gr.Column():
                front_image = gr.Image(
                    label="Front Pose",
                    sources=["webcam"],
                    type="numpy",
                    interactive=True
                )
            
            with gr.Column():
                side_image = gr.Image(
                    label="Side Pose",
                    sources=["webcam"],
                    type="numpy",
                    interactive=True
                )
        
        height_input = gr.Number(
            label="Enter Your Height (cm)",
            minimum=100,
            maximum=250,
            value=170
        )
        
        process_btn = gr.Button("Process Images", variant="primary")
        output_text = gr.Textbox(label="Status")
        
        process_btn.click(
            fn=system.process_images,
            inputs=[front_image, side_image, height_input],
            outputs=[output_text]
        )
    
    return interface
if __name__ == "__main__":
    try:
        interface = create_interface()
        
        # Launch Gradio app with HTTPS configuration
        interface.launch(
            share=True,  # This enables external access
            server_name="0.0.0.0",
            server_port=443,
            auth=None,  # Remove if you want authentication
            favicon_path=None, 
            prevent_thread_lock=False,
        )

    except Exception as e:
        sys.exit(1)






import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
import tf_bodypix
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths
from tf_bodypix.draw import draw_poses
import cv2
import numpy as np
from calculations import measure_body_sizes
import gradio as gr
import ssl

import pandas as pd

class BodyMeasurementSystem:
    def __init__(self):
        self.bodypix_model = load_model(download_model(BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16))
        self.rainbow = [
            [110, 64, 170], [143, 61, 178], [178, 60, 178], [210, 62, 167],
            [238, 67, 149], [255, 78, 125], [255, 94, 99],  [255, 115, 75],
            [255, 140, 56], [239, 167, 47], [217, 194, 49], [194, 219, 64],
            [175, 240, 91], [135, 245, 87], [96, 247, 96],  [64, 243, 115],
            [40, 234, 141], [28, 219, 169], [26, 199, 194], [33, 176, 213],
            [47, 150, 224], [65, 125, 224], [84, 101, 214], [99, 81, 195]
        ]

    def process_images(self, front_img, side_img, real_height_cm):
        try:
            if front_img is None or side_img is None:
                return "Please capture both front and side images."
            
            # Convert uploaded images to numpy arrays if they aren't already
            if not isinstance(front_img, np.ndarray):
                front_img = np.array(front_img)
            if not isinstance(side_img, np.ndarray):
                side_img = np.array(side_img)
                
            frontresult = self.bodypix_model.predict_single(front_img)
            sideresult = self.bodypix_model.predict_single(side_img)
            
            front_mask = frontresult.get_mask(threshold=0.75)
            side_mask = sideresult.get_mask(threshold=0.75)
            
            front_colored_mask = frontresult.get_colored_part_mask(front_mask, self.rainbow)
            side_colored_mask = sideresult.get_colored_part_mask(side_mask, self.rainbow)
            
            frontposes = frontresult.get_poses()
            sideposes = sideresult.get_poses()
            
            if not frontposes or not sideposes:
                return "No body poses detected. Please ensure your full body is visible in both images."
            
            body_sizes = measure_body_sizes(
                side_colored_mask,
                front_colored_mask,
                sideposes,
                frontposes,
                real_height_cm,
                self.rainbow
            )
            
            measurements_df = pd.DataFrame([body_sizes]) if isinstance(body_sizes, dict) else pd.DataFrame(body_sizes)
            csv_file = f"Body-measurement_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            measurements_df.to_csv(csv_file, index=False)
            
            return "Your body measurements have been processed and saved successfully!"
            
        except Exception as e:
            logger.error(f"Error during processing: {str(e)}")
            return f"Error: {str(e)}"


def create_interface():
    system = BodyMeasurementSystem()

    with gr.Blocks(title="Body Sizing System") as interface:
        gr.Markdown("# Body Sizing System")
        gr.Markdown("""
        ## Instructions:
        1. Click the camera icon in each image box to access your device camera
        2. Capture your front and side poses
        3. Enter your height in centimeters
        4. Click Process Images
        """)

        with gr.Row():
            with gr.Column():
                front_image = gr.Image(
                    label="Front Pose",
                    sources=["webcam"],
                    type="numpy",
                    interactive=True
                )
            
            with gr.Column():
                side_image = gr.Image(
                    label="Side Pose",
                    sources=["webcam"],
                    type="numpy",
                    interactive=True
                )
        
        height_input = gr.Number(
            label="Enter Your Height (cm)",
            minimum=100,
            maximum=250,
            value=170
        )
        
        process_btn = gr.Button("Process Images", variant="primary")
        output_text = gr.Textbox(label="Status")
        
        process_btn.click(
            fn=system.process_images,
            inputs=[front_image, side_image, height_input],
            outputs=[output_text]
        )
    
    return interface
if __name__ == "__main__":
    try:
        interface = create_interface()
        
        # Launch Gradio app with HTTPS configuration
        interface.launch(
            share=True,  # This enables external access
            server_name="0.0.0.0",
            server_port=443,
            auth=None,  # Remove if you want authentication
            favicon_path=None, 
            prevent_thread_lock=False,
        )

    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        sys.exit(1)