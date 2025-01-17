import cv2
import os
import glob

def images_to_video(image_folder, output_path, fps=30):
    # Get all png files in folder, sorted by name
    images = sorted(glob.glob(os.path.join(image_folder, "*.png")))
    
    if not images:
        print("No images found")
        return
    
    # Read first image to get dimensions
    frame = cv2.imread(images[0])
    height, width, _ = frame.shape
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'avc1' for .mp4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write each frame to video
    for image_path in images:
        frame = cv2.imread(image_path)
        out.write(frame)
    
    out.release()
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    # Usage example:
    image_folder = "/oscar/home/wfu16/data/users/wfu16/GRAB/processed_data/segvis"  # Folder containing the PNG sequences
    output_path = "output_video.mp4"      # Where to save the video
    images_to_video(image_folder, output_path, fps=60)