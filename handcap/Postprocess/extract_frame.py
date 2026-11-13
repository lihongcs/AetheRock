import cv2
import argparse
import os

def extract_frame(video_path, frame_idx, output_name=None):
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_idx < 0 or frame_idx >= total_frames:
        print(f"Error: frame_idx {frame_idx} is out of bounds. Total frames: {total_frames}")
        cap.release()
        return

    # Set the video position to the specified frame index
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    
    # Read the frame
    ret, frame = cap.read()

    if not ret:
        print(f"Error: Could not read frame {frame_idx}")
        cap.release()
        return

    # Create refer_frames directory in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "refer_frames")
    os.makedirs(output_dir, exist_ok=True)

    # Determine output filename if not provided
    if output_name is None:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_name = f"{base_name}_frame_{frame_idx}.jpg"
    
    # Add .jpg extension if no extension is provided
    if not os.path.splitext(output_name)[1]:
        output_name += ".jpg"

    output_path = os.path.join(output_dir, output_name)

    # Save the frame as an image
    cv2.imwrite(output_path, frame)
    print(f"Successfully saved frame {frame_idx} to: {output_path}")

    # Release the video capture object
    cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract a specific frame from a video and save it as an image.")
    parser.add_argument("video_path", type=str, help="Path to the input video file")
    parser.add_argument("frame_idx", type=int, help="Index of the frame to extract (0-indexed)")
    parser.add_argument("--name", "-n", type=str, default=None, help="Output image name (e.g. 'my_frame' or 'my_frame.jpg'). Saved in 'refer_frames' folder.")

    args = parser.parse_args()
    extract_frame(args.video_path, args.frame_idx, args.name)
