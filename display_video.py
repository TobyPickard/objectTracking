from math import floor
from typing import NoReturn

import cv2
import json
import random
import numpy

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
    file_prefix = video_path.split('.')[0]
    with open(f'{file_prefix}_detections.json', 'r') as file:
        data = json.load(file)
    return data

def show_overlay(frame:numpy.ndarray, frame_data: dict, class_colours: dict) -> None:
    bounding_boxes = frame_data['bounding boxes']
    detection_classes = frame_data['detected classes']
    
    for i in bounding_boxes:
        start_point = (i[0]//2,i[1]//2)
        end_point = ((i[0] + i[2])//2, (i[1] + i[3])//2)
        centroid = ((i[0] + (i[2])//2)//2, (i[1] + (i[3])//2)//2)
        detected_object_index = bounding_boxes.index(i)
        if detection_classes[detected_object_index] not in class_colours:
            class_colours[detection_classes[detected_object_index]] = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        if detection_classes[detected_object_index] == 'person':
            cv2.rectangle(frame, start_point, end_point, class_colours[detection_classes[detected_object_index]], 3)
            cv2.circle(frame, centroid, 5, (255, 0,0), -1)

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
    
    class_colours = {}
    frame_num = 0
    try:
        # read the first frame
        success, frame = video_capture.read()
        print(type(frame))
        frame_num = frame_num + 1 

        # create the window
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)

        # run whilst there are frames and the window is still open
        while success:
            # get detection data for the current frame
            frame_data = detection_data[str(frame_num)]
            
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
    finally:
        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    VIDEO_PATH = "resources/video_2.mp4"
    main(VIDEO_PATH, "My Video")
