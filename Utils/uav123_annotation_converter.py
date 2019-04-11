"""A tool to convert annotations for UAV123 datset into Ground-Truth or Bounding-Box images.

Based off the same code as `cvat_annotation_converter.py' (https://gist.github.com/cheind/9850e35bb08cfe12500942fb8b55531f).
"""
import os
import random
import time

import numpy as np
from tqdm import tqdm

import cv2


def draw_annotations(dataset_folder, sequence_name, target_folder=None, display=False, settings={}):
    tqdm.write(sequence_name)
    # Get the sequnce folder and annotations folder
    sequence_folder = os.path.join(dataset_folder, 'data_seq', 'UAV123', sequence_name)
    annotation_file = os.path.join(dataset_folder, 'anno', 'UAV123', sequence_name + '.txt')
    # Get the default target folder if none is given
    if target_folder == None:
        target_folder = os.path.join(dataset_folder, 'ground_truth', 'UAV123', sequence_name)
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
    try:
        with open(annotation_file, 'r') as f:
            annotations = f.readlines()
    except FileNotFoundError:
        # Skip this sequence as it doesn't have a single annotation file - simplifies processing, but loses out on some data
        return 0
    
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
    for frame_count, annotation in enumerate(tqdm(annotations)):
        # Get the corresponding frame name
        frame_name = frames[frame_count]

        # Read this frame
        frame = cv2.imread(os.path.join(sequence_folder, frame_name))


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


def draw_groundtruth(dataset_folder, sequence_name, target_folder=None, display=False, settings={}):
    tqdm.write(sequence_name)
    # Get the sequnce folder and annotations folder
    sequence_folder = os.path.join(dataset_folder, 'data_seq', 'UAV123', sequence_name)
    annotation_file = os.path.join(dataset_folder, 'anno', 'UAV123', sequence_name + '.txt')
    # Get the default target folder if none is given
    if target_folder == None:
        target_folder = os.path.join(dataset_folder, 'ground_truth', 'UAV123', sequence_name)
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
    try:
        with open(annotation_file, 'r') as f:
            annotations = f.readlines()
    except FileNotFoundError:
        # Skip this sequence as it doesn't have a single annotation file - simplifies processing, but loses out on some data
        return 0
    
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
    for frame_count, annotation in enumerate(tqdm(annotations)):
        # Copy a new blank frame
        frame = np.copy(blank_frame)

        # Get the corresponding frame name
        frame_name = frames[frame_count]

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


def prepare_for_nvvl(dataset_folder, sequence_name, target_folder=None, display=False, settings={}):
    """ Creates a video of the original sequence, and a separate video of the ground truth data,
    stored in target_folder and target_folder/targets respectively.
    """
    # Read in optional settings
    try:
        random_start = settings['random_start']
    except KeyError:
        # random_start not given, use default value
        random_start = False
    try:
        duration = settings['duration']
    except KeyError:
        # random_start not given, use default value
        duration = -1
    
    # Get the sequnce folder and annotations folder
    sequence_folder = os.path.join(dataset_folder, 'data_seq', 'UAV123', sequence_name)
    annotation_file = os.path.join(dataset_folder, 'anno', 'UAV123', sequence_name + '.txt')
    # Get the default target folder if none is given
    if target_folder == None:
        target_folder = os.path.join(dataset_folder, 'ground_truth', 'UAV123', sequence_name)
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
    try:
        with open(annotation_file, 'r') as f:
            annotations = f.readlines()
    except FileNotFoundError:
        # Skip this sequence as it doesn't have a single annotation file - simplifies processing, but loses out on some data
        return 0

    # If random_start is True, randomly generate a starting frame (that is at least 'duration' frames before the end of the video)
    if random_start:
        try:
            start_time = random.randrange(0, len(annotations) - duration)
        except ValueError:
            # The sequence is too short for the requested duration, so skip it
            return
    else:
        start_time = 0
    # Print out the sequence name, start frame, end frame
    tqdm.write(sequence_name + ' ' + str(start_time) + '-' + str(start_time+duration))

    # Convert each line from a string to a list with 4 numbers
    def process_line(line):
        ret = line.strip().split(',')
        ret = list(map(lambda n : int(n) if n != 'NaN' else -1, ret))
        return ret
    annotations = list(map(process_line, annotations))

    # Define the codec and create VideoWriter objects
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    framerate = 30
    (width, height) = (1280, 720)
    out_data = cv2.VideoWriter(os.path.join(target_folder, sequence_name + '.avi'), fourcc, framerate, (width, height), 1)
    out_target = cv2.VideoWriter(os.path.join(target_folder, 'targets', sequence_name + '.avi'), fourcc, framerate, (width, height), 0)

    blank_frame = np.zeros((height, width), dtype=np.uint8)

    last_time = time.time()
    for frame_count, annotation in enumerate(tqdm(annotations[start_time:start_time+duration])):
        # Copy a new blank frame
        frame = np.copy(blank_frame)

        # Get the corresponding frame name
        frame_name = frames[frame_count]

        # Read the input data frame
        input_frame = cv2.imread(os.path.join(sequence_folder, frame_name))

        # Only draw an annotiation if it exists - i.e. annotation is not -1
        if all(a != -1 for a in annotation):
            # Draw the rectangle as a filled in white box
            x_tl = annotation[0]
            y_tl = annotation[1]
            x_br = x_tl + annotation[2]
            y_br = y_tl + annotation[3]
            cv2.rectangle(frame, (x_tl, y_tl), (x_br, y_br), 255, cv2.FILLED)
        
        # Write the frame with boxes
        out_target.write(frame)
        # Write the original input frame
        out_data.write(input_frame)
        # Display the resulting frame
        if display:
            # Force the function to run at the framerate of the video
            while time.time() - last_time < ((1/framerate)/1.025): # Enforce time passed as slightly less than 1/framerate, as displaying the output takes some time as well
                pass
            last_time = time.time()
            cv2.imshow('input frame',input_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Release everything
    out_target.release()
    out_data.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Draw annotations, either on original videos or as ground-truth saliency maps')
    parser.add_argument('--dataset', '-d', dest='dataset', help='Folder containing the UAV123 dataset', required=False, default='C:\\Users\\simon\\Downloads\\Project Datasets\\UAV123\\UAV123')
    parser.add_argument('--sequence', '-s', dest='name', help='Name of sequence to be processed', required=False)
    parser.add_argument('--target', '-t', dest='target_folder', help='Name of folder to write results to', required=False)
    parser.add_argument('--function', '-f', dest='drawing_function', help='Function to use: \'bounding_boxes\', \'groundtruth\', or \'nvvl\'', required = True)
    parser.add_argument('--verbose', '-v', dest='verbose', action='store_true')
    args = parser.parse_args()

    if args.drawing_function == 'bounding_boxes':
        drawing_function = draw_annotations
    elif args.drawing_function == 'groundtruth':
        drawing_function = draw_groundtruth
    elif args.drawing_function == 'nvvl':
        drawing_function = prepare_for_nvvl
        try:
            duration = int(input("Duration (in frames at 30fps, -1 for full video): "))
        except ValueError:
            duration = -1 # int not given, use -1 (default value)
        rand_start = input("Choose a random starting frame? (y/n): ").lower()
        rand_start = True if rand_start in 'yes' and rand_start != '' else False
    
    settings = {
        'random_start': rand_start,
        'duration': duration
    }

    if args.name == None:
        # Read all files in the folder and call the appropriate function on each video/annotation pair found
        # Get a list of frames in sequence_folder
        if os.name == 'posix':
            # Unix
            sequences = os.listdir(os.path.join(args.dataset, 'data_seq', 'UAV123'))
        else:
            # Windows (os.name == 'nt')
            with os.scandir(os.path.join(args.dataset, 'data_seq', 'UAV123')) as folder_iterator:
                sequences = [folder_object.name for folder_object in list(folder_iterator)]
        # Call the drawing function with each sequence name in sequences
        for seq_name in tqdm(sequences):
            drawing_function(args.dataset, seq_name, target_folder=args.target_folder, display=args.verbose, settings=settings)
    elif len(args.name.strip().split(',')) > 1:
        for seq_name in tqdm(args.name.strip().split(',')):
            drawing_function(args.dataset, seq_name.strip(), target_folder=args.target_folder, display=args.verbose, settings=settings)
    else:
        # Draw bounding boxes on the original video, or ground-truth saliency maps, depending on if -bb was specified
        drawing_function(args.dataset, args.name, target_folder=args.target_folder, display=args.verbose, settings=settings)
