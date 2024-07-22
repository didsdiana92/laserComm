import numpy as np
import cv2
import sys

def process_image(image_path):
    # Read the image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not read the image from {image_path}. Please check the file path and try again.")
        return

    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply a Gaussian blur
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    # Apply edge detection
    edges = cv2.Canny(blurred_image, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours)} contours.")

    # Draw contours and calculate spatial statistics
    for contour in contours:
        # Draw the contour on the original image
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

        # Calculate the area and centroid of the contour
        area = cv2.contourArea(contour)
        if area > 0:
            M = cv2.moments(contour)
            centroid_x = int(M['m10'] / M['m00']) if M['m00'] != 0 else 0
            centroid_y = int(M['m01'] / M['m00']) if M['m00'] != 0 else 0
            cv2.circle(image, (centroid_x, centroid_y), 5, (0, 0, 255), -1)
            print(f"Contour Area: {area:.2f}, Centroid: ({centroid_x}, {centroid_y})")

    # Show the images
    cv2.imshow('Original Image with Contours', image)
    cv2.imshow('Edges', edges)
    print("Displaying images. Press any key to close the windows.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 advanced_image_processing.py <path_to_image>")
        sys.exit(1)
    # To introduce the image as an argument when executing the program
    image_path = sys.argv[1]
    process_image(image_path)

if __name__ == "__main__":
    main()
