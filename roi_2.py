import cv2
from ultralytics import YOLO
import os

# Load the YOLOv8 model
model = YOLO(r'G:\shoes_extracted_frame\shoes_detection.pt')  # Path to your trained model

# Define the classes
classes = ['person', 'shoes']

# Start video capture (0 for webcam or provide video file path)
cap = cv2.VideoCapture(r"G:\shoes_extracted_frame\NVR_ch2_main_20240604090000_20240604090053.mp4")

# Get the width, height, and frames per second (FPS) of the input video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Calculate the number of frames to write for 1 minute
frames_to_write = fps * 60  # 60 seconds * FPS

# Ensure output folder exists
output_folder = r"G:\shoes_extracted_frame\output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define the codec and create VideoWriter object to save the output video
output_path = os.path.join(output_folder, "output_video.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Define the fixed ROI coordinates (x1, y1, x2, y2)
roi_rm = (945, 353, 1924, 1085)  # Adjust these as needed

# Frame counter
frame_counter = 0

# Phase 2: Start Detection
while frame_counter < frames_to_write:  # Limit to 1 minute of video
    ret, frame = cap.read()
    if not ret:
        break

    # Draw rectangle around the ROI for visibility
    cv2.rectangle(frame, (roi_rm[0], roi_rm[1]), (roi_rm[2], roi_rm[3]), (255, 0, 0), 2)  # Blue rectangle for ROI

    # Crop the frame to the ROI
    roi_x1, roi_y1, roi_x2, roi_y2 = roi_rm            
    roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]

    # Perform object detection on the cropped ROI frame
    results = model.predict(roi_frame, conf=0.5899)

    # Initialize flags for detections
    detected_person_in_roi = False
    shoes_detected = False

    for result in results[0].boxes:
        cls = int(result.cls[0])  # Class index
        conf = result.conf[0]  # Confidence
        bbox = result.xyxy[0].cpu().numpy()  # Bounding box in the cropped image

        # Map bounding box back to original frame coordinates
        bbox[0] += roi_x1  # x1
        bbox[1] += roi_y1  # y1
        bbox[2] += roi_x1  # x2
        bbox[3] += roi_y1  # y2

        # Draw bounding box for detected objects
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        cv2.putText(frame, f"{classes[cls]}:{conf}", (int(bbox[0]), int(bbox[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Check if a person is detected within the ROI
        if classes[cls] == 'person':
            detected_person_in_roi = True

            # Check if shoes are also detected within the same ROI
            if not shoes_detected:
                for inner_result in results[0].boxes:
                    inner_cls = int(inner_result.cls[0])
                    if classes[inner_cls] == 'shoes':
                        shoes_detected = True
                        break

            if not shoes_detected:
                cv2.putText(frame, "Warning: Person without shoes!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                print("Error: Person detected in ROI without shoes")

    # Show the frame with detections
    cv2.imshow('Detection', frame)

    # Write the annotated frame to the output video
    out.write(frame)

    # Increment frame counter
    frame_counter += 1

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture, video writer, and close any open windows
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video saved to {output_path}")
