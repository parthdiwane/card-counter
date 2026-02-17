import cv2
import numpy as np
import mss
import time
import os
import glob
import sys
from datetime import datetime


# CONFIGS -- DO NOT TOUCH
RECORDINGS_DIR = "/Users/parth/coding/projs/card-counter/model/recordings"

def get_recordings(duration=15) -> str: 
    """
        returns a string of the file path for the recordings
    """
    # configs
    FPS = 30
    # mss monitor indices: 0=all screens, 1=laptop display, 2=external monitor
    MONITOR_NUM = 2
    with mss.mss() as sct:

        monitor = sct.monitors[1]
        w = monitor["width"] # get width 
        h = monitor["height"] # get jeight


        fourcc = cv2.VideoWriter_fourcc(*"mp4v") # compression
        filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".mp4"
        filepath = os.path.join(RECORDINGS_DIR, filename)
        out = cv2.VideoWriter(filepath, fourcc, FPS, (w, h))

        frame_time = 1 / FPS
        print('Recording...', file=sys.stderr)
        start_time = time.time()
        while time.time() - start_time < duration:
            start = time.time()  # frame timer
            frame = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            out.write(frame)

            elapsed = time.time() - start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
        out.release()
        return filepath
if __name__ == "__main__":
    filepath = get_recordings()
    print(filepath)