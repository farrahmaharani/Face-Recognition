import cv2
import os
import numpy as np
import random


def split_dataset(data, test_ratio=0.2):
    test_size = int(len(data) * test_ratio)
    random.shuffle(data)
    return data[test_size:], data[:test_size]

def train_and_test_model():
    faces = []
    labels = []
    # Face detection using Haarcascade
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    for idx, actress_dir in enumerate(os.listdir(data_dir)):
        samples_per_label = 0

        for filename in os.listdir(os.path.join(data_dir, actress_dir)):
            if not filename.endswith(".png"):
                continue
            image = cv2.imread(os.path.join(data_dir, actress_dir, filename))
            if image is None:
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            faces_detected = face_cascade.detectMultiScale(gray)

            # Ensure only one face is detected
            if len(faces_detected) == 1:
                (x, y, w, h) = faces_detected[0]
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (32, 32))
                faces.append(face)
                labels.append(idx)
                samples_per_label += 1

        if samples_per_label < 2:
            print(f"Warning: Insufficient samples for label {idx} ({actress_dir}). Minimum of 2 samples required for LBPH training.")
 
    faces = np.array(faces)
    labels = np.array(labels)

    if len(set(labels)) < 2:
        print("Error: Insufficient unique labels for training. At least two different labels are required.")
        return

    data_train, data_test = split_dataset(list(zip(faces, labels)))

    faces_train, labels_train = zip(*data_train)
    faces_test, labels_test = zip(*data_test)


    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(np.array(faces_train), np.array(labels_train))

    correct_predictions = 0
    for i in range(len(faces_test)):
        face_roi = faces_test[i]
        label, confidence = face_recognizer.predict(face_roi)

        if label == labels_test[i]:
            correct_predictions += 1

        normalized_confidence = confidence / max_confidence if max_confidence > 0 else 0
        print(f"Actual label: {labels_test[i]}, Predicted label: {label}, Normalized confidence: {normalized_confidence:.4f}")

    accuracy = correct_predictions / len(faces_test) if len(faces_test) > 0 else 0
    print(f"\nAverage Accuracy: {accuracy:.4f}")

    # Save the trained result into a model file
    face_recognizer.save("trained_model.yml")

def predict_using_model():
    if not os.path.exists("trained_model.yml"):
        print("Error: Model not found. Train the model first.")
        return
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read("trained_model.yml")

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    label_to_name = {
        0: "Hiroshi Kamiya",
        1: "Inori Minase",
        2: "Kana Hanazawa",
        3: "Kenjirou Tsuda",
        4: "Makoto Furukawa",
        5: "Nobuhiko Okamoto",
        6: "Rie Takahashi",
        7: "Saori Hayami",
        8: "Yui Ishikawa",
        9: "Yuki Kaji"
    }

    test_image_path = input("Enter the path to the test image: ")

    test_image = cv2.imread(test_image_path)

    if test_image is None:
        print("Error: Could not load the test image.")
        return

    gray_test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

    faces_detected = face_cascade.detectMultiScale(gray_test_image)

    max_confidence = float('-inf')

    for (x, y, w, h) in faces_detected:
        face_roi = gray_test_image[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (32, 32))

    label, confidence = face_recognizer.predict(face_roi)

    max_confidence = max(max_confidence, confidence)

    actress_name = label_to_name.get(label, f"Unknown Actress {label}")

    confidence_percentage = (confidence / max_confidence) * 100 if max_confidence > 0 else 0

    print(f"Predicted actress: {actress_name}, Confidence: {confidence_percentage:.2f}%")

    # Draw bounding box and label on the image
    cv2.rectangle(test_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(test_image, f"Predicted: {actress_name} ({confidence_percentage:.2f}%)", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the image with predictions
    cv2.imshow("Test image", test_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main
data_dir = "Dataset"
max_confidence = float('-inf')

while True:
    print("\nMenu:")
    print("1. Train and test model")
    print("2. Predict using model")
    print("3. Exit")

    choice = input("Enter your choice (1/2/3): ")

    if choice == '1':
        train_and_test_model()
    elif choice == '2':
        predict_using_model()
    elif choice == '3':
        print("Exiting the program.")
        break
    else:
        print("Invalid choice. Please enter a valid option (1/2/3).")
