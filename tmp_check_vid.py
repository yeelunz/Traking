import cv2
import os

video_path = r'dataset\CTS_US_正常組-20260223T045150Z-3-001\CTS_US_正常組\n002\G-R.avi'
if os.path.exists(video_path):
    cap = cv2.VideoCapture(video_path)
    print(f'Frames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}')
    cap.release()
else:
    print(f'File not found: {video_path}')
