import cv2
import os
import numpy as np

def assure_path_exists(path):
    """Ensure that a directory exists."""
    if not os.path.exists(path):
        os.makedirs(path)

def get_images_and_labels(dataset_path, face_detector):
    """Retrieve images and corresponding labels from the dataset."""
    image_paths = []
    labels = []
    label_dict = {}
    current_id = 0

    assure_path_exists("../trainer/")

    # Iterate through each person in the dataset
    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_dir):
            continue

        label_dict[current_id] = person_name  # Map ID to person name

        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            if not (image_path.endswith('.jpg') or image_path.endswith('.png')):
                continue
            image_paths.append(image_path)
            labels.append(current_id)

        current_id += 1

    face_samples = []
    face_ids = []

    # Process each image
    for image_path, label in zip(image_paths, labels):
        # Convert image to grayscale
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_samples.append(gray[y:y+h, x:x+w])
            face_ids.append(label)

    return face_samples, face_ids, label_dict

def train_recognizer():
    """Train the LBPH face recognizer and save the model."""
    # Initialize Haar Cascade face detector
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Specify the dataset directory
    dataset_path = 'dataset'  # Adjust the path if necessary

    # Get face samples and labels
    faces, ids, label_dict = get_images_and_labels(dataset_path, face_detector)

    if len(faces) == 0:
        print("No faces found in the dataset. Please add images to the dataset directory.")
        return

    # Initialize the LBPH face recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Train the recognizer
    recognizer.train(faces, np.array(ids))

    # Save the trained model
    recognizer.save("../trainer/trainer.yml")

    # Save the label dictionary
    with open("../trainer/labels.npy", 'wb') as f:
        np.save(f, label_dict)

    print(f"Training completed. {len(label_dict)} classes trained.")
    print("Trained model and label mappings are saved in the 'trainer/' directory.")

if __name__ == "__main__":
    train_recognizer()

