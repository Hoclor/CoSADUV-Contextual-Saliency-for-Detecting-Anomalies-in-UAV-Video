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
    out = cv2.VideoWriter(os.path.join(target_folder, sequence_name + '.avi'), fourcc, framerate, (width, height), 0)
    print(os.path.join(target_folder, sequence_name + '.avi'))

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
        
        frame_count += 1

    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def draw_groundtruth(sequence_name, sequence_folder, annotation_file, display=False):
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
        ret = line.split(','),
        ret = list(map(int, ret))
        return ret
    annotations = list(map(process_line, annotations))
    

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    framerate = 30
    (width, height) = (1280, 720)
    out = cv2.VideoWriter(sequence_name + '_groundtruth.avi', fourcc, framerate, (width, height), 0)

    blank_frame = np.zeros((height, width), dtype=np.uint8)

    for frame_count, frame_name in tqdm(enumerate(frames)):
        # Copy a new blank frame
        frame = np.copy(blank_frame)

        # Get the next line of the annotations file
        for track in tracks:
            # Check that this track has any box nodes left
            if(len(track) > 0):
                # Since the nodes are sorted by frame number, we only have to check the first one
                box = track[0]
                if(int(box.attrib['frame']) == frame_count):
                    # Draw the rectangle described by this 'box' node on this frame
                    # Cast the coordinates to floats, then to ints, as the cv2.rectangle function cannot handle float pixel values
                    # And int(str) cannot handle float strings
                    x_tl = int(float(box.attrib['xtl']))
                    y_tl = int(float(box.attrib['ytl']))
                    x_br = int(float(box.attrib['xbr']))
                    y_br = int(float(box.attrib['ybr']))
                    cv2.rectangle(frame, (x_tl, y_tl), (x_br, y_br), 255, cv2.FILLED)
                    # delete this box from the track, so we can keep only checking the first box in the future
                    track.remove(box)
        
        # Write the frame with boxes
        # Convert to BGR so video can be properly saved
        # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame)
        # Display the resulting frame
        if display:
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_count += 1
    
    # Keep going, as the frame count is not necessarily accurate so we might not be done
    while(True):
        # Copy a new blank frame
        frame = np.copy(blank_frame)
        # Read the next frame but discard it, to check if the video is done yet
        ret, og_frame = cap.read()
        if not ret:
            # Video is done, so break out of the loop
            break

        # Loop over the track objects. For all that have an annotation for this frame, draw a corresponding rectangle with a colour from the colours list
        for track in tracks:
            # Check that this track has any box nodes left
            if(len(track) > 0):
                # Since the nodes are sorted by frame number, we only have to check the first one
                box = track[0]
                if(int(box.attrib['frame']) == frame_count):
                    # Draw the rectangle described by this 'box' node on this frame
                    # Cast the coordinates to floats, then to ints, as the cv2.rectangle function cannot handle float pixel values
                    # And int(str) cannot handle float strings
                    x_tl = int(float(box.attrib['xtl']))
                    y_tl = int(float(box.attrib['ytl']))
                    x_br = int(float(box.attrib['xbr']))
                    y_br = int(float(box.attrib['ybr']))
                    cv2.rectangle(frame, (x_tl, y_tl), (x_br, y_br), 255, cv2.FILLED)
                    # delete this box from the track, so we can keep only checking the first box in the future
                    track.remove(box)
        
        # Write the frame with boxes 
        out.write(frame)
        # Display the resulting frame
        if display:
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_count += 1


    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Draw annotations, either on original videos or as ground-truth saliency maps')
    parser.add_argument('--folder', '-f', dest='folder', help='Folder containing input files. Video and annotation file names must match exactly, with videos as .mp4 or .avi, and annotations as .xml', required = False)
    parser.add_argument('--video', '-vid', dest='video', help='Input video file', required=False)
    parser.add_argument('--annotation', '-ann', dest='ann', help='Dense annotation file', required=False)
    parser.add_argument('--bounding_boxes', '-bb', dest='drawing_function', action='store_const', const=draw_annotations, default=draw_groundtruth)
    parser.add_argument('--verbose', '-v', dest='verbose', action='store_true')
    args = parser.parse_args()


    # Check that either folder was given, or if not then both video and ann was given
    if args.folder == None and (args.video == None or args.ann == None):
        print("Error: invalid inputs given. Either -folder, or both -video and -ann must be specified.")

    if args.folder != None:
        # Read all files in the folder and call the appropriate function on each video/annotation pair found
        #TODO: implement
        pass
    else:
        # Draw bounding boxes on the original video, or ground-truth saliency maps, depending on if -bb was specified
        args.drawing_function(args.video, args.ann, args.verbose)
