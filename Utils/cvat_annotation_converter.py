"""A tool to convert annotation files created with CVAT into ground-truth style images for machine learning.
The initial code was copied from https://gist.github.com/cheind/9850e35bb08cfe12500942fb8b55531f, originally written for
a similar purpose for the tool BeaverDam (which produces json), and was then adapted for use with CVAT (which produces xml).
"""

import cv2
import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm

# Create a list of BGR colours stored as 3-tuples of uint_8s
colours = [
    [255, 0, 0], # Blue
    [0, 255, 0], # Green
    [0, 0, 255], # Red
    [0, 255, 255], # Yellow
    [255, 255, 0], # Cyan
    [255, 0, 255], # Magenta
    [192, 192, 192], # Silver
    [0, 0, 128], # Maroon
    [0, 128, 128], # Olive
    [0, 165, 255] # Orange
]

def draw_annotations(video, annotations, display=False):
    tree = ET.parse(args.ann)
    root = tree.getroot()
    # Create a list of 'track' nodes that are children of the root
    tracks = [child for child in root if child.tag == 'track']

    # Read the video in as a video object
    cap = cv2.VideoCapture(args.video)
    # Get a rough count of the number of frames in the video
    rough_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Find the name/path of the video, without file type
    name_index = -1
    while(args.video[name_index] != '.'):
        name_index -= 1

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    framerate = cap.get(cv2.CAP_PROP_FPS)
    (width, height) = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(args.video[:name_index] + '_annotated.avi', fourcc, framerate, (width, height))


    for frame_count in tqdm(range(rough_frame_count)):
        # Read the next frame of the video
        ret, frame = cap.read()
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
                    cv2.rectangle(frame, (x_tl, y_tl), (x_br, y_br), colours[int(track.attrib['id']) % len(colours)], 2, -1)
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
    
    # Keep going, as the frame count is not necessarily accurate so we might not be done
    while(True):
        # Read the next frame of the video
        ret, frame = cap.read()
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
                    cv2.rectangle(frame, (x_tl, y_tl), (x_br, y_br), colours[int(track.attrib['id']) % len(colours)], 2, -1)
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

def draw_groundtruth(video, annotations, display=False):
    tree = ET.parse(args.ann)
    root = tree.getroot()
    # Create a list of 'track' nodes that are children of the root
    tracks = [child for child in root if child.tag == 'track']

    # Read the video in as a video object
    cap = cv2.VideoCapture(args.video)
    # Get a rough count of the number of frames in the video
    rough_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Find the name/path of the video, without file type
    name_index = -1
    while(args.video[name_index] != '.'):
        name_index -= 1

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    framerate = cap.get(cv2.CAP_PROP_FPS)
    (width, height) = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(args.video[:name_index] + '_groundtruth.avi', fourcc, framerate, (width, height), 0)


    blank_frame = np.zeros((height, width), dtype=np.uint8)

    for frame_count in tqdm(range(rough_frame_count)):
        # Copy a new blank frame
        frame = np.copy(blank_frame)
        # Read the next frame but discard it, to check if the video is done yet
        ret, _ = cap.read()
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
