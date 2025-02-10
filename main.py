import os
import logging
from Detector import Detector

# Configure logging for better traceability
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_model_paths():
    """
    Returns the model paths for configuration, model, and class labels.
    """
    base_path = "model_SSD/ssd_mobilenet_v3_large_coco_2020_01_14"
    config_path = os.path.join(base_path, "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    model_path = os.path.join(base_path, "frozen_inference_graph.pb")
    classes_path = os.path.join(base_path, "coco.names")
    
    return config_path, model_path, classes_path

def validate_video_path(video_path):
    """
    Validate the video file path.
    Returns True if the file exists, False otherwise.
    """
    if not os.path.isfile(video_path):
        logging.error(f"Video file '{video_path}' not found!")
        return False
    return True

def main():
    # Define the path to the video file
    # video_path = "videos/input/multiObject.avi"
    video_path = "videos/input/ball.mp4"  # Unused for now, can be tested later

    # Get the model paths
    config_path, model_path, classes_path = get_model_paths()

    # Validate the video file before processing
    if not validate_video_path(video_path):
        return  # Exit if the video file is not found

    try:
        # Initialize the detector with paths
        detector = Detector(video_path, config_path, model_path, classes_path)
        
        # Start processing the video
        detector.onVideo()
        logging.info(f"Started video processing for {video_path}")
    except Exception as e:
        logging.error(f"Error while processing the video: {e}")

if __name__ == '__main__':
    main()
