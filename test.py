from flask import Flask, render_template, request, redirect, url_for
import cv2
import face_recognition as fr
import numpy as np
import os
import datetime
import csv

app = Flask(__name__)

# Initialize variables for face recognition and attendance tracking
known_names = []
known_name_encodings = []
attendance_log = []

# Folder to save captured faces
faces_folder = "faces"

# Load known faces from the "faces" folder
def load_known_faces():
    known_names.clear()
    known_name_encodings.clear()

    for filename in os.listdir(faces_folder):
        if filename.endswith(".jpg"):
            image_path = os.path.join(faces_folder, filename)
            name, _ = os.path.splitext(filename)
            known_names.append(name)
            
            image = fr.load_image_file(image_path)
            encoding = fr.face_encodings(image)[0]
            known_name_encodings.append(encoding)

# Function to capture a frame from the webcam and perform face detection
def detect_faces():
    video_capture = cv2.VideoCapture(0)

    # Load known faces from the "faces" folder
    load_known_faces()

    while True:
        ret, frame = video_capture.read()

        # Convert BGR to RGB (face_recognition uses RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all face locations in the current frame
        face_locations = fr.face_locations(rgb_frame, model="hog")
        face_encodings = fr.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = fr.compare_faces(known_name_encodings, face_encoding)
            name = "Unknown"

            face_distances = fr.face_distance(known_name_encodings, face_encoding)
            best_match = np.argmin(face_distances)

            if matches[best_match]:
                name = known_names[best_match]

                # Log attendance with timestamp only if not logged before in this session
                entry = {"Name": name, "ID": name.split('_')[1], "Timestamp": datetime.datetime.now().strftime("%H:%M:%S")}
                if entry not in attendance_log:
                    attendance_log.append(entry)
                    print(f"Attendance Logged: {entry}")

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 15), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    video_capture.release()
    cv2.destroyAllWindows()



# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to add faces for training
@app.route('/add_face', methods=['POST'])
def add_face():
    global known_names, known_name_encodings

    # Get name and ID from the form
    name = request.form['name']
    face_id = request.form['id']

    # Capture a frame from the webcam
    video_capture = cv2.VideoCapture(0)
    ret, frame = video_capture.read()

    # Save the captured face image
    image_name = f"{name}_{face_id}.jpg"
    image_path = os.path.join(faces_folder, image_name)
    cv2.imwrite(image_path, frame)

    # Load known faces from the "faces" folder
    load_known_faces()

    # Release the webcam
    video_capture.release()

    return redirect(url_for('home'))

# Route to start face detection
@app.route('/detect_faces')
def detect_faces_route():
    detect_faces()

    # Save attendance log to a CSV file with the current date
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    csv_file_path = f"attendance_{date_str}.csv"
    with open(csv_file_path, 'w', newline='') as csvfile:
        fieldnames = ["Name", "ID", "Timestamp"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(attendance_log)

    return redirect(url_for('home'))

if __name__ == '__main__':
    # Create the "faces" folder if it doesn't exist
    if not os.path.exists(faces_folder):
        os.makedirs(faces_folder)

    app.run(debug=True)