import argparse
import cv2
import numpy as np
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Fire Detection with YOLOv8")
    parser.add_argument('--weights', type=str, default='Models/1.pt', help='model path')
    parser.add_argument('--source', type=str, default='0', help='video source, 0 for webcam or path to video file')
    parser.add_argument('--frame-skip', type=int, default=3, help='number of frames to skip for processing to save CPU')
    parser.add_argument('--persistence', type=int, default=5, help='number of consecutive processed frames with fire to trigger alert')
    parser.add_argument('--conf', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--target-class', type=int, default=1, help='class index representing fire (0 for best.pt, 1 for best (1).pt)')
    
    return parser.parse_args()

def main():
    args = parse_args()

    # Load YOLOv8 model
    print(f"Loading model: {args.weights}")
    model = YOLO(args.weights)

    # Use the specified target class ID from arguments
    target_class_id = args.target_class
    
    # Verify the class ID exists in the model
    if target_class_id not in model.names:
        print(f"Error: Target class ID {target_class_id} not found in model.")
        print(f"Available classes: {model.names}")
        return
        
    target_class_name = model.names[target_class_id]
    print(f"Targeting class: {target_class_name} (ID: {target_class_id})")
    
    # Initialize video capture
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {args.source}")
        return

    frame_count = 0
    consecutive_fire_frames = 0
    
    # For FPS calculation
    fps_start_time = cv2.getTickCount()
    fps_frame_count = 0
    current_fps = 0

    print("Starting detection. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or error reading frame.")
            break
            
        frame_count += 1
        fps_frame_count += 1
        
        # Calculate FPS every 30 frames
        if fps_frame_count >= 30:
            fps_end_time = cv2.getTickCount()
            time_elapsed = (fps_end_time - fps_start_time) / cv2.getTickFrequency()
            current_fps = fps_frame_count / time_elapsed
            fps_frame_count = 0
            fps_start_time = cv2.getTickCount()

        # Frame skipping logic
        if frame_count % args.frame_skip != 0:
            # Still display the skipped frame with FPS info
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Fire Detection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Run YOLOv8 inference
        results = model(frame, conf=args.conf, verbose=False)
        
        fire_detected_in_current_frame = False
        
        # Parse results
        for item in results:
            boxes = item.boxes
            for box in boxes:
                # Get class ID
                cls = int(box.cls[0].item())
                conf = box.conf[0].item()
                
                # Check if the detected class is our target class (Fire)
                if cls == target_class_id:
                    fire_detected_in_current_frame = True
                    
                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2) # Orange box
                    label = f"{target_class_name}: {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                
        # Persistence counter logic
        if fire_detected_in_current_frame:
            consecutive_fire_frames += 1
        else:
            consecutive_fire_frames = 0
            
        # Trigger alert if persistence threshold is met
        if consecutive_fire_frames >= args.persistence:
            # Visual alert on the frame
            cv2.putText(frame, "FIRE ALERT!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            # Create a red border around the whole frame
            frame[:10, :] = [0, 0, 255] # Top
            frame[-10:, :] = [0, 0, 255] # Bottom
            frame[:, :10] = [0, 0, 255] # Left
            frame[:, -10:] = [0, 0, 255] # Right
            
            print(f"ALERT: Fire detected in {args.persistence} consecutive frames!")
            
        # Display debug info
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Persistence: {consecutive_fire_frames}/{args.persistence}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Show the frame
        cv2.imshow("Fire Detection", frame)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
