#!/usr/bin/env python3
"""
HDF5 image viewer
Views the per-time-step camera frames in a dataset collected with Isaac Lab

Usage:
    python View_hdf5_images.py <hdf5_file> [--demo DEMO_ID] [--cam CAM_NAME] [--fps FPS]

Examples:
    python View_hdf5_images.py test_franka_stack_image_dataset_20260205_1630.hdf5
    python View_hdf5_images.py test_franka_stack_image_dataset_20260205_1630.hdf5 --cam wrist_cam
    python View_hdf5_images.py test_franka_stack_image_dataset_20260205_1630.hdf5 --demo 0 --cam table_cam --fps 30

Keys:
    Space   : pause / resume playback
    Left    : previous frame
    Right   : next frame
    Up      : speed up (FPS +5)
    Down    : slow down (FPS -5)
    D       : next demo
    A       : previous demo
    C       : switch camera (table_cam / wrist_cam)
    S       : save the current frame as an image
    Q/Esc   : quit
"""

import argparse
import os
import sys

import cv2
import h5py
import numpy as np


def list_demos(hdf5_file):
    """List every available demo."""
    demos = []
    with h5py.File(hdf5_file, "r") as f:
        if "data" in f:
            for key in f["data"].keys():
                if key.startswith("demo_"):
                    demos.append(key)
    return sorted(demos, key=lambda x: int(x.split("_")[1]))


def list_cameras(hdf5_file, demo_name):
    """List the cameras available in the given demo."""
    cameras = []
    with h5py.File(hdf5_file, "r") as f:
        obs_group = f[f"data/{demo_name}/obs"]
        for key in obs_group.keys():
            if "cam" in key.lower() or obs_group[key].ndim == 4:  # 4D = (T, H, W, C)
                cameras.append(key)
    return cameras


def get_images(hdf5_file, demo_name, cam_name):
    """Return every image for the given demo and camera."""
    with h5py.File(hdf5_file, "r") as f:
        images = f[f"data/{demo_name}/obs/{cam_name}"][:]
    return images


def main():
    parser = argparse.ArgumentParser(description="HDF5 image viewer")
    parser.add_argument("hdf5_file", type=str, help="Path to the HDF5 file")
    parser.add_argument("--demo", type=int, default=0, help="Demo ID to view (default: 0)")
    parser.add_argument("--cam", type=str, default=None, help="Camera name (default: auto-detect)")
    parser.add_argument("--fps", type=int, default=15, help="Playback frame rate (default: 15)")
    parser.add_argument("--start", type=int, default=0, help="Starting frame (default: 0)")
    args = parser.parse_args()

    # Check the file exists
    if not os.path.exists(args.hdf5_file):
        print(f"Error: file '{args.hdf5_file}' does not exist")
        sys.exit(1)

    # List every demo
    demos = list_demos(args.hdf5_file)
    if not demos:
        print("Error: no demo found in the HDF5 file")
        sys.exit(1)

    print(f"Found {len(demos)} demos: {demos}")

    # Select the demo
    demo_idx = args.demo
    if demo_idx >= len(demos):
        print(f"Warning: demo {demo_idx} does not exist, falling back to demo 0")
        demo_idx = 0
    demo_name = demos[demo_idx]

    # List the available cameras
    cameras = list_cameras(args.hdf5_file, demo_name)
    if not cameras:
        print(f"Error: no camera data found in demo '{demo_name}'")
        sys.exit(1)

    print(f"Available cameras: {cameras}")

    # Select the camera
    if args.cam and args.cam in cameras:
        cam_name = args.cam
    else:
        cam_name = cameras[0]
        if args.cam:
            print(f"Warning: camera '{args.cam}' does not exist, falling back to '{cam_name}'")

    cam_idx = cameras.index(cam_name)

    # Load the images
    print(f"Loading images from {demo_name}/{cam_name}...")
    images = get_images(args.hdf5_file, demo_name, cam_name)
    print(f"Image shape: {images.shape}")

    # Playback settings
    frame_idx = min(args.start, len(images) - 1)
    fps = args.fps
    paused = False
    window_name = "HDF5 Image Viewer"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 600, 600)

    print("\n--- Keys ---")
    print("Space : pause / resume")
    print("< >   : previous / next frame")
    print("^ v   : speed up / slow down")
    print("A D   : previous / next demo")
    print("C     : switch camera")
    print("S     : save current frame")
    print("Q/Esc : quit")
    print("--------------\n")

    while True:
        # Grab the current frame
        frame = images[frame_idx]

        # Convert RGB to BGR (OpenCV expects BGR)
        if frame.ndim == 3 and frame.shape[2] == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame

        # Overlay the info text
        info_frame = frame_bgr.copy()
        info_text = f"{demo_name} | {cam_name} | Frame: {frame_idx}/{len(images)-1} | FPS: {fps}"
        status_text = "PAUSED" if paused else "PLAYING"
        
        # Upscale the image so the text stays readable
        scale = 3
        info_frame = cv2.resize(info_frame, (frame_bgr.shape[1] * scale, frame_bgr.shape[0] * scale), 
                                interpolation=cv2.INTER_NEAREST)
        
        cv2.putText(info_frame, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(info_frame, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                    (0, 0, 255) if paused else (0, 255, 0), 2)

        cv2.imshow(window_name, info_frame)

        # Compute the wait time
        wait_time = max(1, int(1000 / fps)) if not paused else 0

        # Handle key presses
        key = cv2.waitKey(wait_time if not paused else 100) & 0xFF

        if key == ord('q') or key == 27:  # Q or Esc
            break
        elif key == ord(' '):  # Space - pause / resume
            paused = not paused
        elif key == 81 or key == ord('a'):  # Left arrow or A - previous frame / previous demo
            if key == ord('a'):  # A - previous demo
                demo_idx = (demo_idx - 1) % len(demos)
                demo_name = demos[demo_idx]
                cameras = list_cameras(args.hdf5_file, demo_name)
                if cam_name not in cameras:
                    cam_name = cameras[0]
                    cam_idx = 0
                images = get_images(args.hdf5_file, demo_name, cam_name)
                frame_idx = 0
                print(f"Switched to {demo_name}, {len(images)} frames")
            else:  # Left - previous frame
                frame_idx = max(0, frame_idx - 1)
        elif key == 83 or key == ord('d'):  # Right arrow or D - next frame / next demo
            if key == ord('d'):  # D - next demo
                demo_idx = (demo_idx + 1) % len(demos)
                demo_name = demos[demo_idx]
                cameras = list_cameras(args.hdf5_file, demo_name)
                if cam_name not in cameras:
                    cam_name = cameras[0]
                    cam_idx = 0
                images = get_images(args.hdf5_file, demo_name, cam_name)
                frame_idx = 0
                print(f"Switched to {demo_name}, {len(images)} frames")
            else:  # Right - next frame
                frame_idx = min(len(images) - 1, frame_idx + 1)
        elif key == 82:  # Up arrow - speed up
            fps = min(60, fps + 5)
            print(f"FPS: {fps}")
        elif key == 84:  # Down arrow - slow down
            fps = max(1, fps - 5)
            print(f"FPS: {fps}")
        elif key == ord('c'):  # C - switch camera
            cam_idx = (cam_idx + 1) % len(cameras)
            cam_name = cameras[cam_idx]
            images = get_images(args.hdf5_file, demo_name, cam_name)
            frame_idx = min(frame_idx, len(images) - 1)
            print(f"Switched to camera: {cam_name}")
        elif key == ord('s'):  # S - save the current frame
            save_path = f"{demo_name}_{cam_name}_frame{frame_idx:04d}.png"
            cv2.imwrite(save_path, frame_bgr)
            print(f"Saved: {save_path}")
        elif key == 2:  # Left arrow (alternative key code)
            frame_idx = max(0, frame_idx - 1)
        elif key == 3:  # Right arrow (alternative key code)
            frame_idx = min(len(images) - 1, frame_idx + 1)

        # Auto-advance to the next frame
        if not paused:
            frame_idx = (frame_idx + 1) % len(images)

    cv2.destroyAllWindows()
    print("Quit")


if __name__ == "__main__":
    main()
