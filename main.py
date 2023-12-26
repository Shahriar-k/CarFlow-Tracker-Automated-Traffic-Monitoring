import cv2
import os

def count_cars(video_path):
    cap = cv2.VideoCapture(video_path)

    # Load the classifier file using the full path provided by OpenCV
    car_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_cars.xml')

    if car_cascade.empty():
        print("Error: Unable to load the Haarcascade file.")
        return

    car_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale for car detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect cars in the frame
        cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))

        # Draw rectangles around the detected cars
        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            car_count += 1

        # Display the frame with detected cars
        cv2.imshow('Car Counting', frame)

        if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
            break

    # Release the video capture object
    cap.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()

    print(f'Total number of cars: {car_count}')

if __name__ == "__main__":
    video_path = r"C:\Users\shahriar\Desktop\NEW\video\your_video.mp4"
    count_cars(video_path)
