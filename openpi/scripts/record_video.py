import cv2
import sys
import signal

if len(sys.argv) < 3:
    print("Usage: python record_video.py <camera_index> <output_file>")
    sys.exit(1)

cam_idx = int(sys.argv[1])
out_file = sys.argv[2]

cap = cv2.VideoCapture(cam_idx)
if not cap.isOpened():
    print(f"Failed to open camera {cam_idx}")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30.0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(out_file, fourcc, fps, (width, height))

running = True

def signal_handler(sig, frame):
    global running
    running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

print(f"Started recording from camera {cam_idx} to {out_file} at {width}x{height} @ {fps}fps. Send SIGINT/SIGTERM to stop.")

while running:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break
    out.write(frame)

cap.release()
out.release()
print("Recording saved successfully.")
