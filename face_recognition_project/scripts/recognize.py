import cv2
import numpy as np
import os

def load_trained_data(model_path, labels_path):
    """Load the trained face recognizer and label mappings."""
    if not os.path.exists(model_path):
        print(f"Trained model '{model_path}' not found. Please train the model first.")
        exit(1)
    if not os.path.exists(labels_path):
        print(f"Label mappings '{labels_path}' not found. Please train the model first.")
        exit(1)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)

    labels = np.load(labels_path, allow_pickle=True).item()
    return recognizer, labels

def recognize_faces():
    """Perform real-time face recognition."""
    # Load pre-trained face detector model (Haar cascade and DNN)
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Load the trained recognizer and labels
    model_path = '../trainer/trainer.yml'
    labels_path = '../trainer/labels.npy'
    recognizer, labels = load_trained_data(model_path, labels_path)

    # Initialize video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # Set width
    cam.set(4, 480)  # Set height

    font = cv2.FONT_HERSHEY_SIMPLEX
    min_confidence = 50  # Minimum confidence for recognition

    print("Starting real-time face recognition. Press 'q' to exit.")

    while True:
        ret, img = cam.read()
        if not ret:
            print("Failed to grab frame from webcam. Exiting...")
            break

        # Convert the image to RGB (OpenCV uses BGR by default)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize image for faster processing
        rgb_img = cv2.resize(rgb_img, (640, 480))

        # Detect faces in the image
        faces = face_detector.detectMultiScale(rgb_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract the face region of interest
            roi_gray = rgb_img[y:y+h, x:x+w]
            roi_gray = cv2.cvtColor(roi_gray, cv2.COLOR_RGB2GRAY)  # Convert to grayscale for recognition

            # Recognize the face
            id_, conf = recognizer.predict(roi_gray)

            if conf < min_confidence:
                name = labels.get(id_, "Unknown")
                confidence_text = f"{round(100 - conf)}%"
            else:
                name = "User Not Found"
                confidence_text = f"{round(100 - conf)}%"

            # Draw rectangle around face and put the name and confidence text
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, str(name), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence_text), (x + 5, y + h + 25), font, 1, (255, 255, 0), 1)

        # Display the resulting image
        cv2.imshow('Face Recognition', img)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("Exiting real-time face recognition.")
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces()




