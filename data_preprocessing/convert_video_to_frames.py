"""A tool to convert annotation files created with CVAT into ground-truth style images
for machine learning. The initial code was copied from:
    https://gist.github.com/cheind/9850e35bb08cfe12500942fb8b55531f
originally written for a similar purpose for the tool BeaverDam (which produces json),
and was then adapted for use with CVAT (which produces xml).
"""

import cv2
import numpy as np
import os
from tqdm import tqdm


def extract_frames(video, height, width, start, end, display=False):
    # Read the video in as a video object
    cap = cv2.VideoCapture(args.video)
    # Get a rough count of the number of frames in the video
    rough_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_len = len(str(rough_frame_count)) # Max length of frame number

    # Find the name/path of the video, without file type
    name_index = -1
    while args.video[name_index] != ".":
        name_index -= 1

    if end == -1:
        tqdm_max_range = rough_frame_count
    else:
        tqdm_max_range = end

    blank_frame = np.zeros((height, width), dtype=np.uint8)
    os.mkdir("frames")
    os.mkdir("targets")

    for frame_count in tqdm(range(tqdm_max_range)):
        # Read the next frame of the video
        ret, frame = cap.read()
        if not ret:
            # Video is done, so break out of the loop
            break

        if frame_count < start:
            continue

        # Resize the frame
        if height != 0 and width != 0:
            frame = cv2.resize(frame, (width, height))

        # Write the frame
        cv2.imwrite("frames\\{}.jpg".format(str(frame_count).zfill(num_len)), frame)
        # Write the target
        cv2.imwrite("targets\\{}.jpg".format(str(frame_count).zfill(num_len)), blank_frame)

        # Display the resulting frame
        if display:
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


        frame_count += 1

    # Keep going, as the frame count is not necessarily accurate so we might not be done
    while True:
        # Read the next frame of the video
        ret, frame = cap.read()
        if not ret:
            # Video is done, so break out of the loop
            break

        if frame_count < start:
            continue
        if frame_count == end:
            break

        # Resize the frame
        if height != 0 and width != 0:
            frame = cv2.resize(frame, (width, height))

        # Write the frame
        cv2.imwrite(".\\{}.jpg".format(str(frame_count).zfill(num_len)), frame)
        # Display the resulting frame
        if display:
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


        frame_count += 1

    # Release everything
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract frames from a video as individual jpg files."
    )
    # parser.add_argument(
    #     "--folder",
    #     "-f",
    #     dest="folder",
    #     help="Folder containing input files. Also where output will be written.",
    #     required=True,
    # )
    parser.add_argument(
        "--video", "-vid", dest="video", help="Input video file", required=True
    )
    parser.add_argument(
        "--height", "-hh", dest="height", help="Height of output frame", required=False
    )
    parser.add_argument(
        "--width", "-w", dest="width", help="Width of output frame", required=False
    )
    parser.add_argument(
        "--start", "-s", dest="start", help="Starting frame", required=False
    )
    parser.add_argument(
        "--end", "-e", dest="end", help="Ending frame", required=False
    )
    parser.add_argument("--verbose", "-v", dest="verbose", action="store_true")
    args = parser.parse_args()

    height = 0 if args.height == None else int(args.height)
    width = 0 if args.width == None else int(args.width)

    start = 0 if args.start == None else int(args.start)
    end = -1 if args.end == None else int(args.end)

    # Draw bounding boxes on the original video, or ground-truth saliency maps,
    # depending on if -bb was specified
    extract_frames(args.video, height, width, start, end, args.verbose)
