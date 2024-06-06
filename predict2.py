from ultralytics import YOLO
import cv2
from cvzone.SelfiSegmentationModule import SelfiSegmentation

# Load models
model = YOLO('ModelTrain/best22.pt')
live = YOLO('ModelTrain/liveness.pt')

# Get model class names
names = model.names
video_capture = cv2.VideoCapture(0)

# Initialize background segmentor
segmentor = SelfiSegmentation()
bgimage = cv2.imread('Images/background.jpg')

# Load input image
input_image_path = 'Images/thanh2.jpg'
frame = cv2.imread(input_image_path)
height, width, _ = frame.shape
bg_image_resized = cv2.resize(bgimage, (width, height))

# Function to draw bounding boxes
def draw_predictions(frame, bounding_box, label):
    x1, y1, x2, y2 = bounding_box
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 2)
    cv2.putText(frame,
                label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 225, 0),
                2,
                cv2.LINE_4)

# Remove background from the input image
frame = segmentor.removeBG(frame, bg_image_resized, cutThreshold=0.8)

# Predict liveliness
liveliness = live.predict(source=frame)

# Process liveliness detection results
for r in liveliness:
    detection_count = r.boxes.shape[0]
    for i in range(detection_count):
        cls = int(r.boxes.cls[i])
        name = r.names[cls]
        bounding_box = r.boxes.xyxy[i].cpu().numpy()
        if name == 'live':
            x1, y1, x2, y2 = map(int, bounding_box)
            face_crop = frame[y1:y2, x1:x2]
            cv2.imwrite('cropped_face.jpg', face_crop)

            # Predict on cropped face using the face detection model
            face_result = model.predict(source=face_crop, conf=0.8)
            for fr in face_result:
                face_detection_count = fr.boxes.shape[0]
                if face_detection_count > 0:
                    for index in range(face_detection_count):
                        face_cls = int(fr.boxes.cls[index])
                        face_name = fr.names[face_cls]
                        confidence = fr.boxes.conf[index]
                        if confidence > 0.7:
                            label = f'{face_name}: {confidence:.2f}'
                        else:
                            label = 'Chua nhan dien duoc'
                        draw_predictions(frame, bounding_box, label)
                else:
                    label = 'Chua nhan dien duoc'
                    draw_predictions(frame, bounding_box, label)

# Display the output image
cv2.imshow('Face Detection', frame)
cv2.waitKey(0)  # Wait for a key press to close the window

# Save the output image
cv2.imwrite('output_image.jpg', frame)
print('Image saved as output_image.jpg')

# Release resources
cv2.destroyAllWindows()
