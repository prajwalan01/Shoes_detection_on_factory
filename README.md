
Here’s a concise README for your GitHub project:

Shoes Detection in Factory
Description
This project uses YOLOv8-based object detection to monitor factory workers' safety by detecting whether individuals are wearing shoes in a specified Region of Interest (ROI) within video frames. It displays warnings for non-compliance and saves the annotated output video.

Features
Detects person and shoes within a defined ROI.
Shows bounding boxes around detected objects.
Displays warning ("Person without shoes!") when non-compliance is detected.
Saves the annotated video (output_video.mp4) for further analysis.
Logs non-compliance events in the console.
Requirements
Python 3.x
OpenCV
YOLOv8 (Ultralytics)
Installation
Clone the repository:
bash
Copy code
git clone https://github.com/your-username/shoes-detection-factory.git
Install required dependencies:
bash
Copy code
pip install -r requirements.txt
Usage
Place your input video in the project folder.
Update the cap = cv2.VideoCapture('video_path') line in the code to point to your video.
Run the script:
bash
Copy code
python roi_2.py
The output video will be saved as output_video.mp4 in the output folder.

