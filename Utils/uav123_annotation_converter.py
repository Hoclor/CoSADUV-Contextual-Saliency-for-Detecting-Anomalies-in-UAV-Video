"""A tool to convert annotations for UAV123 datset into Ground-Truth or Bounding-Box images.

Based off the same code as `cvat_annotation_converter.py'.
"""

import os
import time
import cv2
import numpy as np
from tqdm import tqdm

def draw_annotations(dataset_folder, sequence_name, display=False):
    # Get the sequnce folder and annotations folder
    sequence_folder = os.path.join(dataset_folder, 'data_seq', 'UAV123', sequence_name)
    annotation_file = os.path.join(dataset_folder, 'anno', 'UAV123', sequence_name + '.txt')
    target_folder   = os.path.join(dataset_folder, 'bounding_boxes', 'UAV123', sequence_name)
    # Check if the target_folder exists
    if not os.path.exists(target_folder):
        # Create the target_folder
        os.makedirs(target_folder)

    # Get a list of frames in sequence_folder
    if os.name == 'posix':
        # Unix
        frames = os.listdir(sequence_folder)
    else:
        # Windows (os.name == 'nt')
        with os.scandir(sequence_folder) as file_iterator:
            frames = [file_object.name for file_object in list(file_iterator)]


    # Extract the annotations from the annotation_file
    with open(annotation_file, 'r') as f:
        annotations = f.readlines()
    # Convert each line from a string to a list with 4 numbers
    def process_line(line):
        ret = line.strip().split(',')
        ret = list(map(lambda n : int(n) if n != 'NaN' else -1, ret))
        return ret
    annotations = list(map(process_line, annotations))
    

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    framerate = 30
    (width, height) = (1280, 720)
    out = cv2.VideoWriter(os.path.join(target_folder, sequence_name + '.avi'), fourcc, framerate, (width, height), 1)

    last_time = time.time()
    for frame_count, frame_name in enumerate(tqdm(frames)):
        # Read this frame
        frame = cv2.imread(os.path.join(sequence_folder, frame_name))

        # Get the corresponding annotation
        annotation = annotations[frame_count]

        # Only draw an annotiation if it exists - i.e. annotation is not -1
        if all(a != -1 for a in annotation):
            # Draw the rectangle as a red bounding box
            x_tl = annotation[0]
            y_tl = annotation[1]
            x_br = x_tl + annotation[2]
            y_br = y_tl + annotation[3]
            cv2.rectangle(frame, (x_tl, y_tl), (x_br, y_br), [0, 0, 255], 2, -1)
        
        # Write the frame with boxes
        out.write(frame)
        # Also save the frame individually
        cv2.imwrite(os.path.join(target_folder, frame_name[:-4] + '.png'), frame)
        # Display the resulting frame
        if display:
            # Force the function to run at the framerate of the video
            while time.time() - last_time < ((1/framerate)/1.025): # Enforce time passed as slightly less than 1/framerate, as displaying the output takes some time as well
                pass
            last_time = time.time()
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Release everything
    out.release()
    cv2.destroyAllWindows()

def draw_groundtruth(dataset_folder, sequence_name, display=False):
    # Get the sequnce folder and annotations folder
    sequence_folder = os.path.join(dataset_folder, 'data_seq', 'UAV123', sequence_name)
    annotation_file = os.path.join(dataset_folder, 'anno', 'UAV123', sequence_name + '.txt')
    target_folder   = os.path.join(dataset_folder, 'ground_truth', 'UAV123', sequence_name)
    # Check if the target_folder exists
    if not os.path.exists(target_folder):
        # Create the target_folder
        os.makedirs(target_folder)

    # Get a list of frames in sequence_folder
    if os.name == 'posix':
        # Unix
        frames = os.listdir(sequence_folder)
    else:
        # Windows (os.name == 'nt')
        with os.scandir(sequence_folder) as file_iterator:
            frames = [file_object.name for file_object in list(file_iterator)]


    # Extract the annotations from the annotation_file
    with open(annotation_file, 'r') as f:
        annotations = f.readlines()
    # Convert each line from a string to a list with 4 numbers
    def process_line(line):
        ret = line.strip().split(',')
        ret = list(map(lambda n : int(n) if n != 'NaN' else -1, ret))
        return ret
    annotations = list(map(process_line, annotations))
    

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    framerate = 30
    (width, height) = (1280, 720)
    out = cv2.VideoWriter(os.path.join(target_folder, sequence_name + '.avi'), fourcc, framerate, (width, height), 0)

    blank_frame = np.zeros((height, width), dtype=np.uint8)

    last_time = time.time()
    for frame_count, frame_name in enumerate(tqdm(frames)):
        # Copy a new blank frame
        frame = np.copy(blank_frame)

        # Get the corresponding annotation
        annotation = annotations[frame_count]

        # Only draw an annotiation if it exists - i.e. annotation is not -1
        if all(a != -1 for a in annotation):
            # Draw the rectangle as a filled in white box
            x_tl = annotation[0]
            y_tl = annotation[1]
            x_br = x_tl + annotation[2]
            y_br = y_tl + annotation[3]
            cv2.rectangle(frame, (x_tl, y_tl), (x_br, y_br), 255, cv2.FILLED)
        
        # Write the frame with boxes
        out.write(frame)
        # Also save the frame individually
        cv2.imwrite(os.path.join(target_folder, frame_name[:-4] + '.png'), frame)
        # Display the resulting frame
        if display:
            # Force the function to run at the framerate of the video
            while time.time() - last_time < ((1/framerate)/1.025): # Enforce time passed as slightly less than 1/framerate, as displaying the output takes some time as well
                pass
            last_time = time.time()
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Release everything
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Draw annotations, either on original videos or as ground-truth saliency maps')
    parser.add_argument('--dataset', '-d', dest='dataset', help='Folder containing the UAV123 dataset', required = True)
    parser.add_argument('--sequence', '-seq', dest='name', help='Name of sequence to be processed', required=False)
    parser.add_argument('--bounding_boxes', '-bb', dest='drawing_function', action='store_const', const=draw_annotations, default=draw_groundtruth)
    parser.add_argument('--verbose', '-v', dest='verbose', action='store_true')
    args = parser.parse_args()

    if args.name == None:
        # Read all files in the folder and call the appropriate function on each video/annotation pair found
        #TODO: implement
        pass
    else:
        # Draw bounding boxes on the original video, or ground-truth saliency maps, depending on if -bb was specified
        args.drawing_function(args.dataset, args.name, args.verbose)
