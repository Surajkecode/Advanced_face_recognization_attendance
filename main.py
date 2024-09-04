import face_recognition
# Library for facial recognition, providing functions to detect and recognize faces in images.
import numpy as np
# NumPy for numerical operations, used for handling arrays and matrices in image processing.
import tkinter as tk
# Tkinter for creating the graphical user interface (GUI) for the attendance system.
from tkinter import messagebox, ttk, filedialog
# Additional Tkinter modules for dialogs, themed widgets, and file dialogs in the GUI.
from PIL import Image, ImageTk
# PIL (Pillow) for image processing and converting between formats to display images in the Tkinter GUI.
import csv
# CSV module for reading and writing CSV files, used to store and manage student data and attendance records.
from datetime import datetime
# Module for working with dates and times, used to record the date and time of attendance.
import os
# OS module for interacting with the file system, used for managing file paths and directories.

import cv2
# File paths for storing registered students data, attendance records, and student images

registered_students_file = 'registered_students.csv'
attendance_file = 'attendance.csv'
student_images_dir = 'student_images'

# Ensure the directory for storing student images exists; if not, create it.

# The 'exist_ok=True' parameter prevents an error if the directory already exists.

os.makedirs(student_images_dir, exist_ok=True)

# Function to register a new student by capturing their name, roll number, and photo.
def register_student():
    # Get the student's name and roll number from the respective entry widgets.

    name = entry_name.get()
    roll_no = entry_roll_no.get()

    # Check if either name or roll number is empty, and show an error message if true.

    if not name or not roll_no:
        messagebox.showerror("Error", "Please enter name and roll number.")
        return
        # Exit the function if input validation fails.

    # Capture the student's photo and save it to the student_images_dir.

    image_filename = capture_photo(name, roll_no)

    # Open the CSV file in append mode to add the new student's details.
    # 'newline=""' ensures that there are no extra blank lines added between rows.

    with open(registered_students_file, 'a', newline='') as file:
        writer = csv.writer(file)
        # Create a CSV writer object to write data to the file.
        # Write the student's name, roll number, and the photo's filename to the CSV file.

        writer.writerow([name, roll_no, image_filename])

    # Show a success message once the student is successfully registered.

    messagebox.showinfo("Success", f"Student {name} with Roll No {roll_no} successfully registered.")
    # Update the student list in the application, reflecting the new addition.

    update_student_list()

# Function to capture a photo of the student using the webcam.
def capture_photo(student_name, roll_number):
    # Initialize the webcam for video capture. Index '0' usually refers to the default webcam.

    video_capture = cv2.VideoCapture(0)
    # Check if the webcam is accessible, and raise an error if not.

    if not video_capture.isOpened():
        raise RuntimeError("Cannot access the webcam.")

    while True:
        # Capture a single frame from the webcam. 'ret' indicates success, 'frame' contains the image.

        ret, frame = video_capture.read()
        # If capturing the frame fails, raise an error.

        if not ret:
            raise RuntimeError("Failed to capture image.")

        # Display the live video feed in a window. The user should adjust their face in the frame.

        cv2.imshow('Adjust Your Face and Press Q to Capture', frame)

        # Wait for the user to press the 'Q' key to capture the image.
        # '0xFF' masks the key press to the last byte, ensuring cross-platform compatibility.

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            # Exit the loop once the user presses 'Q'.

    # Construct the filename for the captured image using the student's name and roll number.

    image_filename = os.path.join(student_images_dir, f'{student_name}_{roll_number}.jpg')

    # Save the captured frame as an image file in the student_images_dir.

    cv2.imwrite(image_filename, frame)

    # Release the webcam resource, closing it for further use.
    video_capture.release()

    # Close all OpenCV windows that might be open.
    cv2.destroyAllWindows()

    # Return the filename of the captured image for further use in the registration process.
    return image_filename


def recognize_and_mark_attendance():
    # Initialize the webcam for video capture. Index '0' usually refers to the default webcam.
    video_capture = cv2.VideoCapture(0)
    # Check if the webcam is accessible, and show an error message if not.
    if not video_capture.isOpened():
        messagebox.showinfo("Error", "Cannot access the webcam.")
        return
        # Exit the function if the webcam cannot be accessed.

    # Initialize lists to store known face encodings, names, and roll numbers.
    known_face_encodings = []
    known_face_names = []
    known_face_roll_nos = []

    # Open the CSV file containing the registered students' details.
    with open(registered_students_file, 'r') as file:
        reader = csv.reader(file)
        # Create a CSV reader object to read the file.
        # Iterate over each row in the CSV file.

        for row in reader:
            # Extract the student's name, roll number, and the filename of their registered image.

            registered_name, registered_roll_no, registered_image_filename = row
            # Load the registered image file for face recognition.

            registered_image = face_recognition.load_image_file(registered_image_filename)
            # Get the face encodings from the loaded image.

            registered_face_encodings = face_recognition.face_encodings(registered_image)
            # If face encodings are found, add the first encoding to the known encodings list.

            if registered_face_encodings:
                known_face_encodings.append(registered_face_encodings[0])
                known_face_names.append(registered_name)
                # Add the student's name to the known names list.
                known_face_roll_nos.append(
                    registered_roll_no)
                # Add the student's roll number to the known roll numbers list.

    # Dictionary to keep track of recognized students and the time they were recognized.
    recognized_students = {}

    while True:

        # Capture a single frame from the webcam. 'ret' indicates success, 'frame' contains the image.
        ret, frame = video_capture.read()

        # If capturing the frame fails, show an error message and exit the loop.
        if not ret:
            messagebox.showinfo("Error", "Failed to capture image.")
            break

        # Convert the captured frame from BGR (OpenCV format) to RGB (face_recognition format).

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect all face locations in the current frame using the 'hog' model for speed.

        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        # Get the face encodings for all detected faces in the current frame.

        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Iterate over the detected face locations and their corresponding encodings.

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Compare the detected face encoding with the known face encodings.

            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            # Calculate the face distances between the detected face and known faces.

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            # Get the index of the best match (smallest distance).

            best_match_index = np.argmin(face_distances)
            # If a match is found and the distance is below a threshold (0.6), consider it a valid match.

            if matches[best_match_index] and face_distances[best_match_index] < 0.6:
                # Retrieve the name and roll number of the matched student.

                name = known_face_names[best_match_index]
                roll_no = known_face_roll_nos[best_match_index]

                # Get the current date and time of recognition.

                now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                # If the student has not been recognized yet in this session, add them to the recognized list.

                if name not in recognized_students:
                    recognized_students[name] = (roll_no, now)
                    # Store the roll number and time of recognition.

                # Prepare the text to be displayed on the frame, including the student's name and roll number.
                display_text = f"{name} ({roll_no})"
                # Draw a rectangle around the recognized face in the frame.

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                # Display the student's name and roll number above the rectangle.

                cv2.putText(frame, display_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                # If no match is found, draw a red rectangle around the unrecognized face.

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                # Display 'Unknown' above the rectangle for unrecognized faces.

                cv2.putText(frame, "Unknown", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Display the current frame with the recognized faces in a window.

        cv2.imshow('Verification - Press Q to Capture and Enter to Exit', frame)

        # Wait for the user to press 'Q' to capture and exit the loop, or 'Enter' to continue.

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break  # Exit the loop when 'Q' is pressed.

    # Release the webcam resource, closing it for further use.
    video_capture.release()
    # Close all OpenCV windows that might be open.
    cv2.destroyAllWindows()

    # Mark attendance and show results for recognized students.
    mark_attendance(recognized_students)
    # Call the function to mark attendance for the students who were recognized.
    show_verification_result(
        recognized_students)
    # Call the function to display the results of the verification process to the user.


def mark_attendance(recognized_students):
    today_date = datetime.now().strftime('%Y-%m-%d')
    # Get the current date in 'YYYY-MM-DD' format.
    updated_students = {}
    # Initialize an empty dictionary to keep track of students whose attendance was successfully updated.

    existing_students = []  # Initialize a list to store existing attendance records.
    try:
        with open(attendance_file, 'r', newline='') as f:  # Attempt to open the attendance file in read mode.
            reader = csv.reader(f)
            # Create a CSV reader object to read the attendance records.
            for row in reader:
                # Loop through each row in the attendance file.
                if len(row) >= 3:
                    # Check if the row has at least three elements (name, roll number, and date).
                    name, roll_no, date, time = row[0], row[1], row[2], row[3] if len(row) > 3 else ''
                    # Extract the name, roll number, date, and time from the row.
                    if name in recognized_students and date == today_date:
                        # Check if the student's name is in the recognized students list and if the date matches today's date.
                        recognized_students.pop(name)
                        # Remove the student from the recognized students list if their attendance has already been marked today.
                        messagebox.showwarning("Duplicate Entry", f"Attendance already marked for {name} today.")
                        # Show a warning message indicating that the student's attendance is already marked.
                else:
                    existing_students.append(row)
                    # If the row doesn't have enough elements, add it to the existing_students list.
    except FileNotFoundError:
        # Handle the case where the attendance file does not exist.
        existing_students = []
        # If the file is not found, initialize an empty list for existing students.

    # Write new attendance records to the attendance file.
    with open(attendance_file, 'a', newline='') as f:
        # Open the attendance file in append mode.
        writer = csv.writer(f)
        # Create a CSV writer object to write new attendance records.
        for name, (roll_no, time) in recognized_students.items():
            # Loop through each recognized student.
            writer.writerow([name, roll_no, today_date, time])
            # Write the student's name, roll number, date, and time to the attendance file.
            updated_students[name] = (roll_no, time)  # Add the student to the updated_students dictionary.

    # Update the attendance list in the UI.
    update_attendance_list()  # Call the function to refresh the attendance list displayed in the application.

    if updated_students:  # Check if there were any new students whose attendance was marked.
        messagebox.showinfo("Success",
                            # If there were, show a success message listing all students whose attendance was marked.
                            f"Attendance marked for {', '.join([f'{name} at {time}' for name, (roll_no, time) in updated_students.items()])}.")
    else:
        messagebox.showinfo("Info", "No new attendance was marked.")
        # If no new attendance was marked, show an info message.


def show_verification_result(recognized_students):
    if not recognized_students:  # Check if there are no recognized students.
        messagebox.showinfo("Verification Results", "No students recognized.")
        # If no students were recognized, show an info message.
    else:
        result = '\n'.join(  # Create a string with the name, roll number, and time for each recognized student.
            [f'{name} (Roll No: {roll_no}) at {time}' for name, (roll_no, time) in recognized_students.items()])
        messagebox.showinfo("Verification Results", f"Recognized Students:\n{result}")
        # Show a message box displaying the list of recognized students.




def delete_entry():
    selected_item = tree.selection()
    if not selected_item:
        messagebox.showerror("Error", "Please select a student to delete.")
        return

    # Retrieve student name and roll number from the selected tree item
    student_name = tree.item(selected_item)['values'][0]
    roll_no = tree.item(selected_item)['values'][1]

    # Read all registered students data from the CSV file
    with open(registered_students_file, 'r') as file:
        lines = file.readlines()

    # Filter out the line that contains the selected student
    with open(registered_students_file, 'w', newline='') as file:
        writer = csv.writer(file)
        for line in lines:
            if not line.startswith(student_name):
                file.write(line)

    # Find and remove the image file associated with the student
    try:
        image_file = os.path.join(student_images_dir, f"{student_name}_{roll_no}.jpg")
        if os.path.exists(image_file):
            os.remove(image_file)
        else:
            messagebox.showwarning("Warning", f"Image for {student_name} not found.")
    except Exception as e:
        messagebox.showerror("Error", f"Could not delete image file: {e}")

    # Update the student list in the UI
    update_student_list()

    # Show success message
    messagebox.showinfo("Success", f"Student {student_name} deleted successfully.")

# Function to export attendance list
def export_attendance():
    with open(attendance_file, 'r') as file:
        lines = file.readlines()

    export_file = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if not export_file:
        return

    with open(export_file, 'w', newline='') as file:
        file.writelines(lines)

    messagebox.showinfo("Success", "Attendance list exported successfully.")

def export_all_students():
    with open(registered_students_file, 'r') as file:
        lines = file.readlines()

    export_file = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if not export_file:
        return

    with open(export_file, 'w', newline='') as file:
        file.writelines(lines)

    messagebox.showinfo("Success", "Registered students exported successfully.")


def update_student_list():
    tree.delete(*tree.get_children())
    with open(registered_students_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            tree.insert('', 'end', values=row[:2])


def update_attendance_list():
    attendance_tree.delete(*attendance_tree.get_children())
    with open(attendance_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            attendance_tree.insert('', 'end', values=row)


def create_main_menu():
    frame_main_menu = ttk.Frame(root)
    # Create a new frame widget using ttk (Themed Tkinter) and assign it to the main window (root).
    notebook.add(frame_main_menu, text="Home")
    # Add the frame to the notebook widget as a new tab with the label "Home".

    # Load and set the background image for the Home tab
    home_bg_photo = Image.open(r"E:\face Advanced\FaceAdvanced\Attendanceface (1)\Attendanceface\register.jpg")
    # Open the image file for the Home tab background.
    home_bg_photo = home_bg_photo.resize((800, 600), Image.LANCZOS).convert("RGBA")
    # Resize the image to fit the window and convert it to RGBA format for transparency.
    home_bg_photo_tk = ImageTk.PhotoImage(home_bg_photo)
    # Convert the image to a format that Tkinter can use (PhotoImage).

    home_bg_label = tk.Label(frame_main_menu, image=home_bg_photo_tk)
    # Create a label widget to display the background image in the frame.
    home_bg_label.image = home_bg_photo_tk
    # Keep a reference to the image to prevent it from being garbage collected by Python.
    home_bg_label.place(x=0, y=0, relwidth=1, relheight=1)
    # Position the label to cover the entire frame using relative width and height.

    tk.Label(frame_main_menu, text="Student Registration and Verification System", font=('Arial', 18),
             background="#228B22", foreground="white").pack(pady=20)
    # Add a label with the title text, set its font, background color, and text color, and add padding around it.
    ttk.Button(frame_main_menu, text="Register Student", command=lambda: notebook.select(frame_register)).pack(pady=10)
    # Add a button to navigate to the "Register Student" tab, with padding around it.
    ttk.Button(frame_main_menu, text="Verify Students", command=lambda: notebook.select(frame_verification)).pack(
        pady=10)
    # Add a button to navigate to the "Verify Students" tab, with padding around it.
    ttk.Button(frame_main_menu, text="Registered Students", command=lambda: notebook.select(frame_list)).pack(pady=10)
    # Add a button to navigate to the "Registered Students" tab, with padding around it.
    ttk.Button(frame_main_menu, text="Attendance", command=lambda: notebook.select(frame_attendance)).pack(pady=10)
    # Add a button to navigate to the "Attendance" tab, with padding around it.



# Setup UI
root = tk.Tk()
root.title("Student Registration and Verification")
root.geometry("800x460")  # Adjusted to match background image size

# Apply styles
style = ttk.Style()
style.configure('TButton', background='#8B0000', foreground='red', padding=10, font=('Arial', 12))
style.configure('Treeview', font=('Arial', 12))
style.configure('Treeview.Heading', font=('Arial', 14))

notebook = ttk.Notebook(root)
notebook.pack(pady=10, expand=True)

# Main Menu Frame
create_main_menu()

# Registration Frame
frame_register = ttk.Frame(notebook)
notebook.add(frame_register, text="Register Student")

# Load and set the background image for the Registration tab
register_bg_photo = Image.open(r"E:\face Advanced\FaceAdvanced\Attendanceface (1)\Attendanceface\b.jpg")
register_bg_photo = register_bg_photo.resize((800, 600), Image.LANCZOS).convert("RGBA")
register_bg_photo_tk = ImageTk.PhotoImage(register_bg_photo)

register_bg_label = tk.Label(frame_register, image=register_bg_photo_tk)
register_bg_label.image = register_bg_photo_tk  # Keep a reference to the image to prevent garbage collection
register_bg_label.place(x=0, y=0, relwidth=1, relheight=1)

tk.Label(frame_register, text="Name:", font=('Arial', 14), background="Green").place(x=5, y=100)
entry_name = ttk.Entry(frame_register, font=('Arial', 14))
entry_name.place(x=70, y=100)

tk.Label(frame_register, text="Roll No:", font=('Arial', 14), background="Green").place(x=1, y=150)
entry_roll_no = ttk.Entry(frame_register, font=('Arial', 14))
entry_roll_no.place(x=75, y=150)

register_button = ttk.Button(frame_register, text="Register Student", command=register_student)
register_button.place(x=80, y=200)

# Verification Frame
frame_verification = ttk.Frame(notebook)
notebook.add(frame_verification, text="Verification")

# Load and set the background image for the Verification tab
verify_bg_photo = Image.open(r"E:\face Advanced\FaceAdvanced\Attendanceface (1)\Attendanceface\home_image.jpg")
verify_bg_photo = verify_bg_photo.resize((800, 600), Image.LANCZOS).convert("RGBA")
verify_bg_photo_tk = ImageTk.PhotoImage(verify_bg_photo)

verify_bg_label = tk.Label(frame_verification, image=verify_bg_photo_tk)
verify_bg_label.image = verify_bg_photo_tk  # Keep a reference to the image to prevent garbage collection
verify_bg_label.place(x=0, y=0, relwidth=1, relheight=1)

verify_button = ttk.Button(frame_verification, text="Verify Students", command=recognize_and_mark_attendance)
verify_button.pack(pady=10)

# Registered Students Frame
frame_list = ttk.Frame(notebook)
notebook.add(frame_list, text="Registered Students")

# Load and set the background image for the Registered Students tab
regstu_bg_photo = Image.open(r"E:\face Advanced\FaceAdvanced\Attendanceface (1)\Attendanceface\regstu.jpg")
regstu_bg_photo = regstu_bg_photo.resize((800, 600), Image.LANCZOS).convert("RGBA")
regstu_bg_photo_tk = ImageTk.PhotoImage(regstu_bg_photo)

regstu_bg_label = tk.Label(frame_list, image=regstu_bg_photo_tk)
regstu_bg_label.image = regstu_bg_photo_tk  # Keep a reference to the image to prevent garbage collection
regstu_bg_label.place(x=0, y=0, relwidth=1, relheight=1)

tree = ttk.Treeview(frame_list, columns=('Name', 'Roll No'), show='headings')
tree.heading('Name', text='Name')
tree.heading('Roll No', text='Roll No')
tree.pack(pady=20, expand=True)

delete_button = ttk.Button(frame_list, text="Delete Entry", command=delete_entry)
delete_button.pack(pady=10)

# Adding the Export List button
export_button = ttk.Button(frame_list, text="Export List", command=export_all_students)
export_button.pack(pady=10)

## Attendance Frame
frame_attendance = ttk.Frame(notebook)
notebook.add(frame_attendance, text="Attendance")

# Load and set the background image for the Attendance tab
attend_bg_photo = Image.open(r"E:\face Advanced\FaceAdvanced\Attendanceface (1)\Attendanceface\attend.jpg")
attend_bg_photo = attend_bg_photo.resize((800, 600), Image.LANCZOS).convert("RGBA")
attend_bg_photo_tk = ImageTk.PhotoImage(attend_bg_photo)

attend_bg_label = tk.Label(frame_attendance, image=attend_bg_photo_tk)
attend_bg_label.image = attend_bg_photo_tk  # Keep a reference to the image to prevent garbage collection
attend_bg_label.place(x=0, y=0, relwidth=1, relheight=1)

attendance_tree = ttk.Treeview(frame_attendance, columns=('Name', 'Roll No', 'Date', 'Time'), show='headings')
attendance_tree.heading('Name', text='Name')
attendance_tree.heading('Roll No', text='Roll No')
attendance_tree.heading('Date', text='Date')
attendance_tree.heading('Time', text='Time')
attendance_tree.pack(pady=5, expand=True)

# Adding the Export List button to the Attendance tab (with correct functionality)
export_attendance_button = ttk.Button(frame_attendance, text="Export List", command=export_attendance)
export_attendance_button.pack(pady=10)

# Initialize the student and attendance lists
update_student_list()
update_attendance_list()


root.mainloop()
'''Here’s a brief explanation of how and why the algorithms are used in face recognition:

### HOG (Histogram of Oriented Gradients)

How:
1. Feature Extraction: HOG is used to detect features in images. It works by dividing the image into small cells and calculating the histogram of gradient directions within each cell.
2. Descriptor Calculation: These histograms are then combined into a descriptor which represents the shape and appearance of objects in the image.
3. Detection: The descriptor is used in conjunction with a machine learning model (like a linear SVM) to detect faces.

Why:
- Robustness: HOG features are robust to changes in illumination and can handle variations in face pose and expression.
- Efficiency: It provides a good balance between computational efficiency and accuracy, making it suitable for real-time face detection.

### Face Encoding (from `face_recognition` Library)

How:
1. Face Detection: The library first detects faces in an image using methods like HOG or a deep learning-based detector.
2. Feature Extraction: For each detected face, it computes a set of 128-dimensional face encodings (features) that represent the unique characteristics of the face.
3. Comparison: These encodings are used to compare and identify faces by measuring the Euclidean distance between feature vectors.

Why:
- Accuracy: The `face_recognition` library uses deep learning-based face encodings, which are highly accurate for distinguishing between different faces.
- Ease of Use: The library simplifies the process of face recognition by providing pre-trained models and easy-to-use functions for face encoding and matching.

In summary say that guys , HOG is often used for initial face detection due to its efficiency and robustness, while face encodings provide a more detailed and accurate representation of facial features for recognition and comparison.'''

"""1. Single-Face Detection
How It Works: The system is designed to detect and recognize only one face at a time. It usually works by focusing on the most prominent face in the camera's field of view.
Process:
The system captures an image or video frame.
It applies face detection algorithms (like Haar cascades, HOG + SVM, or CNN-based detectors) to identify the largest or most prominent face.
After detecting the face, the system extracts facial features (using methods like FaceNet, DeepFace, etc.) and compares them with stored data to recognize the individual.
Use Case: This method is simpler and faster, ideal for applications where only one person is expected to be in front of the camera at a time, such as at an entry gate or in a one-on-one interaction."""

""" # Multiple-Face Detection:

1. Detection Process
A. Image/Video Frame Capture
The system first captures an image or a sequence of video frames from a camera feed.
This image/frame is then passed to the face detection algorithm.
B. Face Detection Algorithm
The algorithm scans the entire frame to identify all potential face regions.
Common algorithms for face detection include:
Haar Cascades: Uses machine learning to detect faces based on edge or line features. However, it is relatively outdated for detecting multiple faces.
Histogram of Oriented Gradients (HOG) + Support Vector Machine (SVM): A method that detects faces by analyzing the gradients in the image. It's faster than Haar Cascades but can struggle with complex scenes.
Convolutional Neural Networks (CNNs): Modern face detection models, such as MTCNN (Multi-task Cascaded Convolutional Networks) or the YOLO (You Only Look Once) family, use deep learning to accurately detect faces, even in challenging conditions.
C. Face Detection Steps
Sliding Window: The algorithm typically uses a sliding window technique to scan through the image at different scales and locations to identify face-like patterns.
Bounding Box Generation: Once faces are detected, each face is marked with a bounding box, indicating the location and size of the detected faces.
Non-Maximum Suppression (NMS): To avoid overlapping boxes for the same face, NMS is applied to keep the most relevant bounding box.
2. Face Tracking
Object Tracking Algorithms: Once faces are detected, the system can track them across multiple frames using algorithms like:
Correlation Filters: Tracks the face based on appearance.
Kalman Filters: Predicts the next position of the face, adjusting based on actual detected locations.
Optical Flow: Tracks the motion of pixels between frames to follow face movement.
3. Feature Extraction
For each detected face, the system extracts key facial features. This can involve:
Landmark Detection: Identifying specific facial points (e.g., eyes, nose, mouth).
Face Embeddings: Using models like FaceNet, DeepFace, or ArcFace, the face is encoded into a high-dimensional vector, which represents the unique features of the face.
4. Recognition and Matching
Database Comparison: The extracted features are compared with stored face embeddings in a database to find the best match.
Softmax or Nearest Neighbor Classifier: A classifier is used to determine the identity of each face based on the closest match in the database."""


"""Disclaimer: This is Final Project,if you are modify this project in your side ,I can't responsible"""