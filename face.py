"""
  DISCLAIMER:
    This script is purely for educational purposes only. Any images included in this project 
    are used solely for recreational and educational facial recognition demonstrations involving 
    prominent political figures. The images are not used for any commercial purposes

  Description:
    This program is designed for face recognition and annotation in images. 
    It loads images from a specified directory, detects faces, and matches 
    them against a known database of faces. The program allows for adding new 
    faces to the database and processes images in parallel to improve performance.
    Users can also quit the program at any time by typing 'q', 'quit', or 'stop' 
    in the terminal
"""

import sys
import threading            # Used for creating and managing separate threads for concurrent execution
import cv2                  # Used for image manipulation and displaying images
import face_recognition     # Used to locate and recognize faces in images
import pickle               # Used to save and load face encodings and names
from   pathlib import Path  # Provides an easy way to work with file paths
from   multiprocessing import Pool, cpu_count # Allows processing multiple images simultaneously using multiple CPU cores

stop_flag = threading.Event()

def load_image(file_path: str) -> 'ndarray' or None:
    """
    Load an image file for face recognition processing

    Parameters:
        file_path (str): Path to the image file
    Return:
        Loaded image (ndarray) or None if an error occurs
    """
    try:
        return face_recognition.load_image_file(file_path)
    except Exception as error:
        print(f"Error loading image {file_path}: {error}")
        return None

def resize_image(image: 'ndarray', target_width: int = None, target_height: int = None) -> 'ndarray':
    """
    Resize an image to a specified width or height while maintaining aspect ratio

    Parameters:
        image --- (ndarray): Original image to be resized
        target_width  (int): Desired width of the resized image
        target_height (int): Desired height of the resized image
    Return:
        resized_image (ndarray)
    """
    (original_height, original_width) = image.shape[:2]
    if target_width is None and target_height is None:
        return image
    if target_width is None:
        resize_ratio = target_height / float(original_height)
        dimensions   = (int(original_width * resize_ratio), target_height)
    else:
        resize_ratio = target_width / float(original_width)
        dimensions   = (target_width, int(original_height * resize_ratio))
    return cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)

def save_face_data(encodings: list, names: list, encodings_file_path: str, names_file_path: str):
    """
    Save known face encodings and names to files

    Parameters:
        encodings -------- (list): List of face encodings
        names ------------ (list): List of names corresponding to the face encodings
        encodings_file_path (str): Path to the file for storing encodings
        names_file_path --- (str): Path to the file for storing names
    """
    with open(encodings_file_path, "wb") as encodings_file, open(names_file_path, "wb") as names_file:
        pickle.dump(encodings, encodings_file)
        pickle.dump(names, names_file)

def add_face_to_database(name: str, image_path: str, known_face_encodings: list, known_face_names: list, encodings_file_path: str, names_file_path: str):
    """
    Add a new face to the known faces database

    Parameters:
        name ---------------- (str): Name of the person associated with the face
        image_path ---------- (str): Path to the image file
        known_face_encodings (list): List of known face encodings
        known_face_names --- (list): List of known face names
        encodings_file_path - (str): Path to the file for storing encodings
        names_file_path ----- (str): Path to the file for storing names
    """
    new_face_image = load_image(image_path)
    if new_face_image is not None:
        known_face_encodings.append(face_recognition.face_encodings(new_face_image)[0])
        known_face_names.append(name)
        save_face_data(known_face_encodings, known_face_names, encodings_file_path, names_file_path)
        print(f"Added new face for {name}")

def process_image(image_file_path: Path, known_face_encodings: list, known_face_names: list, target_display_width: int):
    """
    Process a single image file for face recognition

    Parameters:
        image_file_path ---- (Path): Path to the image file
        known_face_encodings (list): List of known face encodings
        known_face_names --- (list): List of known face names
        target_display_width  (int): Target width for displaying the image
    """
    face_image = load_image(image_file_path)
    if face_image is None:
        return None, None

    # Detect face locations and encodings in the image
    face_locations = face_recognition.face_locations(face_image)
    face_encodings = face_recognition.face_encodings(face_image, face_locations)
    face_image_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR) # Convert the image to BGR color (OpenCV uses BGR by default)

    # The variables top, right, bottom, and left represent the coordinates of the bounding box around a detected face in an image. These
    #  coordinates are used to draw a rectangle around the face
    #    Here’s what each of these variables specifically represents:
    #      top --: The y-coordinate of the top    edge of the bounding box
    #      bottom: The y-coordinate of the bottom edge of the bounding box
    #      right-: The x-coordinate of the right  edge of the bounding box
    #      left--: The x-coordinate of the left   edge of the bounding box
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        face_matches  = face_recognition.compare_faces(known_face_encodings, face_encoding)
        detected_name = known_face_names[face_matches.index(True)] if True in face_matches else "Unknown"

        cv2.rectangle(face_image_bgr, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(face_image_bgr, detected_name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    resized_image = resize_image(face_image_bgr, target_width=target_display_width)
    return resized_image, image_file_path

def monitor_input():
    while True:
        if input().strip().lower() in ['q', 'quit', 'stop']:
            stop_flag.set()
            print("Exiting the program.")
            break

def main():
    face_data_directory = Path.cwd() / 'face_data'
    face_data_directory.mkdir(parents=True, exist_ok=True)

    encodings_file_path = face_data_directory / "face_encodings.pkl"
    names_file_path     = face_data_directory / "face_names.pkl"

    try:
        with open(encodings_file_path, "rb") as encodings_file, open(names_file_path, "rb") as names_file:
            known_face_encodings = pickle.load(encodings_file)
            known_face_names     = pickle.load(names_file)
    except FileNotFoundError:
        known_face_encodings = []
        known_face_names     = []

    while True:
        add_new_face_prompt = input("Do you want to add a new face? (yes/no): ").strip().lower()
        if add_new_face_prompt in ['yes', 'y']:
            person_name     = input("Enter the name of the person: ").strip()
            face_image_path = input("Enter the path to the image file: ").strip()
            add_face_to_database(person_name, face_image_path, known_face_encodings, known_face_names, encodings_file_path, names_file_path)
        elif add_new_face_prompt in ['quit', 'stop', 'q']:
            print("Exiting the program.")
            sys.exit(0)
        else:
            break

    current_directory = Path.cwd()
    image_files       = [file for extension in ('*.png', '*.jpg', '*.jpeg') for file in current_directory.rglob(extension)]
    target_display_width = 800

    print('Enter q, quit, or stop to end the program (close all open face recognition windows after you entered one of these commands)')

    # Use an input thread that can stop the program while the user is viewing images
    input_thread = threading.Thread(target=monitor_input)
    input_thread.start()

    # Use a multiprocessing Pool to process images in parallel. Provides significant speed increase depending on
    # system (compared to sequential processing of images) and is easier to implement compared to threading
    with Pool(processes=cpu_count()) as pool:
        # Apply the process_image function to each image file asynchronously and store the results in the results list
        results = [pool.apply_async(process_image, (image_file, known_face_encodings, known_face_names, target_display_width)) for image_file in image_files]

        for result in results:
            if stop_flag.is_set(): break
            resized_image, image_file_path = result.get()
            if resized_image is not None:
                cv2.imshow("Image", resized_image)
                key = cv2.waitKey(0)
                if key in [ord('q'), ord('Q')]:
                    stop_flag.set()
                    print("Exiting the program. Please close any current windows.")
                    break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
