# Computer Vision Projects

This repository contains a collection of Python scripts using OpenCV to demonstrate various computer vision tasks. These include image processing, face detection, video processing, object tracking, and face recognition. The scripts utilize libraries like `cv2`, `glob`, `numpy`, and `pandas` to implement different use cases of computer vision.

## Features

1. **Image Loading and Display**
   - Load and display images in grayscale and color.
   - Print image properties (e.g., type, dimensions).
   - Resize images and save them to disk.

2. **Face Detection**
   - Use Haar Cascade Classifiers to detect faces in images.
   - Draw bounding boxes around detected faces.
   - Show the original, grayed, and resized versions of images.

3. **Video Capture and Processing**
   - Capture and display live video from webcam or video files.
   - Process video frames to detect moving objects.
   - Track entry and exit times of objects and save them to a CSV file.

4. **Face Recognition**
   - Train a face recognition model using LBPH (Local Binary Pattern Histogram).
   - Detect faces and identify individuals (e.g., Obama, Mandela, Trump).

5. **Motion Detection**
   - Implement motion detection using frame difference.
   - Track movement and log timestamps of entry and exit of objects.

## Requirements

- Python 3.x
- OpenCV (`cv2`), Numpy, Pandas
- Required Haar Cascade files for face detection

To install the necessary Python libraries, you can use the following command:

```bash
pip install opencv-python numpy pandas
```

## Usage

### Image Processing

1. **Load and display an image**:
   - The script reads an image (`galaxy.jpg`), prints its type and shape, and displays it in grayscale.
   - It also demonstrates resizing an image and saving it.

2. **Loop through multiple images**:
   - The script uses `glob` to find all `.jpg` files in the `second` folder, resize them to 100x100, display them, and save the resized images.

### Face Detection

- Load an image and use a pre-trained Haar Cascade model (`haarcascade_frontalface_default.xml`) to detect faces.
- Draw bounding boxes around detected faces and display the original and grayed images.

### Face Recognition

- Train a face recognizer using the LBPH method and recognize faces in a given image.
- The script identifies faces (e.g., Obama, Mandela, Trump) and displays the result on the image.

### Motion Detection

- Use frame differencing to detect motion in a video.
- Log the entry and exit times of objects and save them to a CSV file (`Times.csv`).

### Video Capture

- Capture video from a webcam or a file and process each frame.
- Detect motion, display the frame, and log the activity.

## Example Script: Image Loading and Face Detection

```python
import cv2

# Load image in grayscale
img = cv2.imread("./first/galaxy.jpg", 0)

# Print image type and shape
print(type(img))
print(img)
print(img.shape)

# Show image
cv2.imshow("Galaxy", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Detect faces in the image
face_cascade = cv2.CascadeClassifier("./face_detector/haarcascade_frontalface_default.xml")
faces = face_cascade.detectMultiScale(img, scaleFactor=1.05, minNeighbors=5)

# Draw bounding boxes around detected faces
for x, y, w, h in faces:
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

# Show the result
cv2.imshow("Face Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Motion Detection Script Example

```python
import cv2
import pandas
from datetime import datetime

# Initialize video capture
video = cv2.VideoCapture("video.mp4")
first_frame = None
status_list = [None, None]
times = []
df = pandas.DataFrame(columns=["Start", "End"])

while True:
    check, frame = video.read()
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (21, 21), 0)

    if first_frame is None:
        first_frame = gray_img
        continue

    delta_frame = cv2.absdiff(first_frame, gray_img)
    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    contours, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    status = 0
    for contour in contours:
        if cv2.contourArea(contour) < 10000:
            continue

        status = 1
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    status_list.append(status)

    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.now())
    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.now())

    # Show frames
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        if status == 1:
            times.append(datetime.now())
        break

video.release()
cv2.destroyAllWindows()

# Save times to CSV
for time in range(0, len(times), 2):
    df = df.append({
        "Start": times[time],
        "End": times[time + 1],
    }, ignore_index=True)

df.to_csv("Times.csv")
```

## Contributing

Feel free to fork the repository, create a branch, make your changes, and submit a pull request. Contributions are always welcome!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- OpenCV for computer vision tasks.
- Haar Cascade for face detection.
- LBPH Face Recognizer for face recognition.

