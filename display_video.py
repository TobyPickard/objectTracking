from dis import dis
from math import floor, dist
from typing import NoReturn

import cv2
import json
import random
import numpy
import time 

def open_video(path: str) -> cv2.VideoCapture:
    """Opens a video file.

    Args:
        path: the location of the video file to be opened

    Returns:
        An opencv video capture file.
    """
    video_capture = cv2.VideoCapture(path)
    if not video_capture.isOpened():
        raise RuntimeError(f'Video at "{path}" cannot be opened.')
    return video_capture


def get_frame_dimensions(video_capture: cv2.VideoCapture) -> tuple[int, int]:
    """Returns the frame dimension of the given video.

    Args:
        video_capture: an opencv video capture file.

    Returns:
        A tuple containing the height and width of the video frames.

    """
    return video_capture.get(cv2.CAP_PROP_FRAME_WIDTH), video_capture.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )


def get_frame_display_time(video_capture: cv2.VideoCapture) -> int:
    """Returns the number of milliseconds each frame of a VideoCapture should be displayed.

    Args:
        video_capture: an opencv video capture file.

    Returns:
        The number of milliseconds each frame should be displayed for.
    """
    frames_per_second = video_capture.get(cv2.CAP_PROP_FPS)
    return floor(1000 / frames_per_second)


def is_window_open(title: str) -> bool:
    """Checks to see if a window with the specified title is open."""

    # all attempts to get a window property return -1 if the window is closed
    return cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) >= 1

def get_detection_data(video_path: str) -> dict:
    """automatically gets the correct detection data based on which video is being used"""
    
    file_prefix = video_path.split('.')[0]
    with open(f'{file_prefix}_detections.json', 'r') as file:
        data = json.load(file)
    return data

def show_overlay(frame: numpy.ndarray, frame_data: dict, class_colours: dict) -> None:
    """Adds the boxe data around, centroid and id to all people detected in the video
    
    Args: 
        frame: pixel data for the frame of the video being looked at.
        frame_data: data containing objects detected in the video.
        class_colours: a dictionary containing colours for each unique class type that has been detected thus far.

    Returns:
        None
    """
    bounding_boxes = frame_data['bounding boxes']
    detection_classes = frame_data['detected classes']
    centroids = frame_data['centroids']
    ids = frame_data['ids']
    
    font = cv2.FONT_HERSHEY_COMPLEX
    
    # Loop through adding the detection data to the overlay on the video
    for box in bounding_boxes:
        start_point = (box[0]//2,box[1]//2)
        end_point = ((box[0] + box[2])//2, (box[1] + box[3])//2)
        detected_object_index = bounding_boxes.index(box)
        
        # Get a random color for every unique detected class
        if detection_classes[detected_object_index] not in class_colours:
            class_colours[detection_classes[detected_object_index]] = \
                (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        
        # only add to overlay if the detected class is person
        if detection_classes[detected_object_index] == 'person':
            cv2.rectangle(img=frame, 
                          pt1=start_point, 
                          pt2=end_point, 
                          color=class_colours[detection_classes[detected_object_index]],
                          thickness=3)
            cv2.circle(img=frame, 
                       center=tuple(coordinate//2 for coordinate in centroids[detected_object_index]),
                       radius=5,
                       color=(255, 0,0),
                       thickness=-1)
            cv2.putText(img=frame, 
                        text=ids[detected_object_index],
                        org=tuple(coordinate//2 for coordinate in centroids[detected_object_index]), 
                        fontFace=font,
                        fontScale=1,
                        color=(255,0,0),
                        thickness=2)

def caluculate_centroids(frame_data: dict) -> None:
    """Calcultes the center point of each object detected in the video"""
    for box in frame_data['bounding boxes']:
        frame_data['centroids'].append(((box[0] + (box[2])//2), (box[1] + (box[3])//2)))

def set_initial_id(frame_data: dict, id_counter: int) -> None:
    """Setting the ID value for each person detected"""
    for i in frame_data['detected classes']:
        if i == 'person':
            frame_data['ids'].append(f'ID:{id_counter}')
            id_counter += 1
        else:
            frame_data['ids'].append(None)
    return id_counter

def track_objects(frame_data: dict, last_frame:dict) -> None:
    """Compares current and last frame data to see if there are any objects that are likley to be the same object."""
    if last_frame != {}:
        for centroid in frame_data['centroids']:
            for reference_centroid in last_frame['centroids']:
                if dist(centroid,reference_centroid) < 50:
                    frame_data['ids'][frame_data['centroids'].index(centroid)] = \
                        last_frame['ids'][last_frame['centroids'].index(reference_centroid)]

def main(video_path: str, title: str) -> NoReturn:
    """Displays a video at half size until it is complete or the 'q' key is pressed.

    Args:
        video_path: the location of the video to be displayed
        title: the title to display in the video window
    """

    video_capture = open_video(video_path)
    width, height = get_frame_dimensions(video_capture)
    wait_time = get_frame_display_time(video_capture)
    detection_data = get_detection_data(video_path)
    
    last_frame = {}
    class_colours = {}
    frame_num = 0
    id_counter = 0
    try:
        # read the first frame
        success, frame = video_capture.read()
        frame_num = frame_num + 1 

        # create the window
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)

        # run whilst there are frames and the window is still open
        while success and is_window_open(title):
            # get detection data for the current frame
            frame_data = detection_data[str(frame_num)]
            frame_data['centroids'] = []
            frame_data['ids'] = []
            caluculate_centroids(frame_data)

            # Set the ID value of each object classified as a person
            id_counter = set_initial_id(frame_data, id_counter)
            
            # Compare this frame to last frame to track detected objects
            track_objects(frame_data, last_frame)
            
            # shrink it            
            smaller_image = cv2.resize(frame, (floor(width // 2), floor(height // 2)))

            # overlay 
            show_overlay(smaller_image, frame_data, class_colours)

            # display it
            cv2.imshow(title, smaller_image)

            # test for quit key
            if cv2.waitKey(wait_time) == ord("q"):
                break

            # read the next frame
            success, frame = video_capture.read()
            frame_num = frame_num + 1
            last_frame = frame_data
    finally:
        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    VIDEO_PATH = "resources/video_1.mp4"
    main(VIDEO_PATH, "My Video")
