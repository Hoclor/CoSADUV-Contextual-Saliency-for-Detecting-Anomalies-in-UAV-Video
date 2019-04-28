"""A tool to convert annotations for UAV123 datset into Ground-Truth or Bounding-Box images.

Based off the same code as `cvat_annotation_converter.py'
(https://gist.github.com/cheind/9850e35bb08cfe12500942fb8b55531f).
"""
import os
import random
import time

import numpy as np
from tqdm import tqdm

import cv2


def draw_annotations(
    dataset_folder,
    sequence_name,
    target_folder=None,
    img_size=(640, 480),
    display=False,
):
    tqdm.write(sequence_name)
    # Get the sequnce folder and annotations folder
    sequence_folder = os.path.join(dataset_folder, "data_seq", "UAV123", sequence_name)
    annotation_file = os.path.join(
        dataset_folder, "anno", "UAV123", sequence_name + ".txt"
    )
    # Get the default target folder if none is given
    if target_folder == None:
        target_folder = os.path.join(
            dataset_folder, "ground_truth", "UAV123", sequence_name
        )
    # Check if the target_folder exists
    if not os.path.exists(target_folder):
        # Create the target_folder
        os.makedirs(target_folder)

    # Get a list of frames in sequence_folder
    if os.name == "posix":
        # Unix
        frames = os.listdir(sequence_folder)
    else:
        # Windows (os.name == 'nt')
        with os.scandir(sequence_folder) as file_iterator:
            frames = [file_object.name for file_object in list(file_iterator)]

    # Extract the annotations from the annotation_file
    try:
        with open(annotation_file, "r") as f:
            annotations = f.readlines()
    except FileNotFoundError:
        # Skip this sequence as it doesn't have a single annotation file
        # (simplifies processing, but loses out on some data)
        return 0

    # Convert each line from a string to a list with 4 numbers
    def process_line(line):
        ret = line.strip().split(",")
        ret = list(map(lambda n: int(n) if n != "NaN" else -1, ret))
        return ret

    annotations = list(map(process_line, annotations))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    framerate = 30
    (width, height) = (1280, 720)
    out = cv2.VideoWriter(
        os.path.join(target_folder, sequence_name + ".avi"),
        fourcc,
        framerate,
        (width, height),
        1,
    )

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
        cv2.imwrite(os.path.join(target_folder, frame_name[:-4] + ".png"), frame)
        # Display the resulting frame
        if display:
            # Force the function to run at the framerate of the video
            while time.time() - last_time < ((1 / framerate) / 1.025):
                # Enforce time passed as slightly less than 1/framerate,
                # as displaying the output takes some time as well
                pass
            last_time = time.time()
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # Release everything
    out.release()
    cv2.destroyAllWindows()


def draw_groundtruth(
    dataset_folder,
    sequence_name,
    target_folder=None,
    img_size=(640, 480),
    display=False,
):
    tqdm.write(sequence_name)
    # Get the sequnce folder and annotations folder
    sequence_folder = os.path.join(dataset_folder, "data_seq", "UAV123", sequence_name)
    annotation_file = os.path.join(
        dataset_folder, "anno", "UAV123", sequence_name + ".txt"
    )
    # Get the default target folder if none is given
    if target_folder == None:
        target_folder = os.path.join(
            dataset_folder, "ground_truth", "UAV123", sequence_name
        )
    # Check if the target_folder exists
    if not os.path.exists(target_folder):
        # Create the target_folder
        os.makedirs(target_folder)

    # Get a list of frames in sequence_folder
    if os.name == "posix":
        # Unix
        frames = os.listdir(sequence_folder)
    else:
        # Windows (os.name == 'nt')
        with os.scandir(sequence_folder) as file_iterator:
            frames = [file_object.name for file_object in list(file_iterator)]

    # Extract the annotations from the annotation_file
    try:
        with open(annotation_file, "r") as f:
            annotations = f.readlines()
    except FileNotFoundError:
        # Skip this sequence as it doesn't have a single annotation file
        # (simplifies processing, but loses out on some data)
        return 0

    # Convert each line from a string to a list with 4 numbers
    def process_line(line):
        ret = line.strip().split(",")
        ret = list(map(lambda n: int(n) if n != "NaN" else -1, ret))
        return ret

    annotations = list(map(process_line, annotations))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    framerate = 30
    (width, height) = (1280, 720)
    out = cv2.VideoWriter(
        os.path.join(target_folder, sequence_name + ".avi"),
        fourcc,
        framerate,
        (width, height),
        0,
    )

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
        cv2.imwrite(os.path.join(target_folder, frame_name[:-4] + ".png"), frame)
        # Display the resulting frame
        if display:
            # Force the function to run at the framerate of the video
            while time.time() - last_time < ((1 / framerate) / 1.025):
                # Enforce time passed as slightly less than 1/framerate,
                # as displaying the output takes some time as well
                pass
            last_time = time.time()
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # Release everything
    out.release()
    cv2.destroyAllWindows()


def prepare_for_nvvl(
    dataset_folder,
    sequence_name,
    target_folder=None,
    img_size=(640, 480),
    display=False,
):
    """ Writes frames of the original sequence into target_folder/sequence_name/frames,
    and grayscale ground truth frames into target_folder/sequence_name/targets.
    """
    # Get the sequnce folder and annotations folder
    sequence_folder = os.path.join(dataset_folder, "data_seq", "UAV123", sequence_name)
    annotation_file = os.path.join(
        dataset_folder, "anno", "UAV123", sequence_name + ".txt"
    )

    # Check if the required folders exists
    if not os.path.exists(os.path.join(target_folder, sequence_name, "frames")):
        # Create the target_folder
        os.makedirs(os.path.join(target_folder, sequence_name, "frames"))
    if not os.path.exists(os.path.join(target_folder, sequence_name, "targets")):
        # Create the target_folder
        os.makedirs(os.path.join(target_folder, sequence_name, "targets"))

    # Get a list of frames in sequence_folder
    if os.name == "posix":
        # Unix
        frames = os.listdir(sequence_folder)
    else:
        # Windows (os.name == 'nt')
        with os.scandir(sequence_folder) as file_iterator:
            frames = [file_object.name for file_object in list(file_iterator)]

    # Extract the annotations from the annotation_file
    try:
        with open(annotation_file, "r") as f:
            annotations = f.readlines()
    except FileNotFoundError:
        # Skip this sequence as it doesn't have a single annotation file
        # (simplifies processing, but loses out on some data)
        return 0

    # Print out the sequence name
    tqdm.write(sequence_name)

    # Convert each line from a string to a list with 4 numbers
    def process_line(line):
        ret = line.strip().split(",")
        ret = list(map(lambda n: int(n) if n != "NaN" else -1, ret))
        return ret

    annotations = list(map(process_line, annotations))

    width, height = img_size

    # Original UAV123 dataset size: 1280x720
    blank_frame = np.zeros((720, 1280), dtype=np.uint8)

    for frame_count, annotation in enumerate(tqdm(annotations)):
        # Copy a new blank frame
        target_frame = np.copy(blank_frame)

        # Check if we have processed the entire video sequence - if so, stop
        if frame_count >= len(frames):
            break

        # Get the corresponding frame name
        frame_name = frames[frame_count]

        # Read the input data frame
        input_frame = cv2.imread(os.path.join(sequence_folder, frame_name))
        input_frame = cv2.resize(input_frame, (width, height))

        # Only draw an annotiation if it exists - i.e. annotation is not -1
        if all(a != -1 for a in annotation):
            # Draw the rectangle as a filled in white box
            x_tl = annotation[0]
            y_tl = annotation[1]
            x_br = x_tl + annotation[2]
            y_br = y_tl + annotation[3]
            cv2.rectangle(target_frame, (x_tl, y_tl), (x_br, y_br), 255, cv2.FILLED)

        # Resize the target frame
        target_frame = cv2.resize(target_frame, (width, height))

        # Write the input frame
        cv2.imwrite(
            os.path.join(target_folder, sequence_name, "frames", frame_name),
            input_frame,
        )
        # Write the target
        cv2.imwrite(
            os.path.join(target_folder, sequence_name, "targets", frame_name),
            target_frame,
        )
        # Display the resulting frame
        if display:
            cv2.imshow("input frame", input_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    # Release everything
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Draw annotations, either on original videos or as ground-truth saliency maps"
    )
    parser.add_argument(
        "--default",
        dest="default",
        help="Ignores all other settings and uses default setup",
        action="store_true",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--function",
        "-f",
        dest="drawing_function",
        help="Function to use: 'bounding_boxes', 'groundtruth', or 'nvvl'",
        required=False,
    )
    parser.add_argument(
        "--width",
        "-w",
        dest="width",
        help="Width of the output frames",
        required=False,
        default="640",
    )
    parser.add_argument(
        "--height",
        "-hh",
        dest="height",
        help="Height of the output frames",
        required=False,
        default="480",
    )
    parser.add_argument(
        "--sequence",
        "-s",
        dest="name",
        help="Name of sequence to be processed",
        required=False,
    )
    parser.add_argument(
        "--dataset",
        "-d",
        dest="dataset",
        help="Folder containing the UAV123 dataset",
        required=False,
        default="C:\\Users\\simon\\Downloads\\Project Datasets\\UAV123\\UAV123",
    )
    parser.add_argument(
        "--target",
        "-t",
        dest="target_folder",
        help="Name of folder to write results to",
        required=False,
        default="C:\\Users\\simon\\GitRepositories\\MastersProject\\DSCLRCN-PyTorch\\Dataset\\UAV123\\train",
    )
    parser.add_argument("--verbose", "-v", dest="verbose", action="store_true")
    args = parser.parse_args()

    if args.drawing_function == "bounding_boxes":
        drawing_function = draw_annotations
    elif args.drawing_function == "groundtruth":
        drawing_function = draw_groundtruth
    elif args.drawing_function == "nvvl":
        drawing_function = prepare_for_nvvl
    elif args.default == False:
        print(
            "Error: either --function/-f must be given, or --default must be specified. See -h for more info."
        )
        quit()

    # Convert the width and height into ints
    img_size = (int(args.width), int(args.height))

    if args.default == True:
        dataset = "C:\\Users\\simon\\Downloads\\Project Datasets\\UAV123\\UAV123"
        img_size = (640, 480)  # Original size (1280x720)
        # Omit Building sequences, UAV sequences, and simulated sequences
        # List of sequences split into train, val, test sets
        sequences = {
            "train": [
                "bike1",
                "boat1",
                "boat4",
                "boat7",
                "boat8",
                "boat9",
                "car1",
                "car4",
                "car7",
                "car10",
                "car12",
                "car13",
                "car16",
                "car17",
                "car18",
                "group1",
                "person1",
                "person2",
                "person3",
                "person4",
                "person5",
                "person6",
                "person12",
                "person13",
                "person14",
                "person15",
                "person16",
                "person17",
                "truck1",
                "truck2",
                "wakeboard1",
                "wakeboard2",
                "wakeboard3",
                "wakeboard4",
                "wakeboard5",
            ],
            "val": [
                "bike2",
                "boat2",
                "boat5",
                "car2",
                "car5",
                "car8",
                "car11",
                "car14",
                "group2",
                "person7",
                "person8",
                "person9",
                "person10",
                "person11",
                "truck3",
                "wakeboard6",
                "wakeboard7",
            ],
            "test": [
                "bike3",
                "boat3",
                "boat6",
                "car3",
                "car6",
                "car9",
                "car15",
                "group3",
                "person18",
                "person19",
                "person20",
                "person21",
                "person22",
                "person23",
                "truck4",
                "wakeboard8",
                "wakeboard9",
                "wakeboard10",
            ],
        }
        targets = {
            "train": "C:\\Users\\simon\\GitRepositories\\MastersProject\\model\\Dataset\\UAV123\\train",
            "val": "C:\\Users\\simon\\GitRepositories\\MastersProject\\model\\Dataset\\UAV123\\val",
            "test": "C:\\Users\\simon\\GitRepositories\\MastersProject\\model\\Dataset\\UAV123\\test",
        }
        # Loop over all sequences
        for section in tqdm(["train", "val", "test"]):
            for sequence in tqdm(sequences[section]):
                prepare_for_nvvl(dataset, sequence, targets[section], img_size, False)
    elif args.name == None:
        # Read all files in the folder and call the appropriate function on each
        # video/annotation pair found.

        # Get a list of frames in sequence_folder
        if os.name == "posix":
            # Unix
            sequences = os.listdir(os.path.join(args.dataset, "data_seq", "UAV123"))
        else:
            # Windows (os.name == 'nt')
            with os.scandir(
                os.path.join(args.dataset, "data_seq", "UAV123")
            ) as folder_iterator:
                sequences = [
                    folder_object.name for folder_object in list(folder_iterator)
                ]
        # Call the drawing function with each sequence name in sequences
        for seq_name in tqdm(sequences):
            drawing_function(
                args.dataset,
                seq_name,
                target_folder=args.target_folder,
                img_size=img_size,
                display=args.verbose,
            )
    elif len(args.name.strip().split(",")) > 1:
        for seq_name in tqdm(args.name.strip().split(",")):
            drawing_function(
                args.dataset,
                seq_name.strip(),
                target_folder=args.target_folder,
                img_size=img_size,
                display=args.verbose,
            )
    else:
        # Draw bounding boxes on the original video, or ground-truth saliency maps,
        # depending on if -bb was specified
        drawing_function(
            args.dataset,
            args.name,
            target_folder=args.target_folder,
            img_size=img_size,
            display=args.verbose,
        )

