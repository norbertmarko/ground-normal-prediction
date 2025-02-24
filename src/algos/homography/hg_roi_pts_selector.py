from os import environ
import argparse
import cv2
import numpy as np


# initialize global variables
points = []
selected_point = -1  # index of the selected point (-1 if none)


def draw_points(image, points):
    """Draw points and lines on the image."""
    for idx, point in enumerate(points):
        color = (255, 0, 0) if idx != selected_point else (0, 0, 255)
        cv2.circle(image, tuple(point), 8, color, -1)
    if len(points) == 4:
        for i in range(4):
            cv2.line(image, tuple(points[i]), tuple(points[(i + 1) % 4]), (255, 0, 0), 2)


def mouse_callback(event, x, y, flags, param):
    """Mouse callback function for selecting points."""
    global points, selected_point
    if event == cv2.EVENT_LBUTTONDOWN:
        if flags & cv2.EVENT_FLAG_SHIFTKEY:
            # add a new point if Shift is pressed and fewer than 4 points exist
            if len(points) < 4:
                points.append([x, y])
        else:
            # check if clicking near an existing point to select it
            distances = [np.linalg.norm(np.array([x, y]) - np.array(pt)) for pt in points]
            if distances and min(distances) < 10:  # threshold for selecting a point
                selected_point = np.argmin(distances)
            else:
                selected_point = -1

    elif event == cv2.EVENT_MOUSEMOVE:
        # move the selected point if one is selected
        if selected_point != -1:
            points[selected_point] = [x, y]

    elif event == cv2.EVENT_LBUTTONUP:
        # deselect the point on mouse release
        selected_point = -1


def center_window(window_name, image_shape):
    """Center the OpenCV window on the screen."""
    # Get screen resolution from environment variables (works in WSL and Linux)
    screen_width = int(environ.get("DISPLAY_WIDTH", 3840))  # default to 3840 if not set (1920)
    screen_height = int(environ.get("DISPLAY_HEIGHT", 2160))  # default to 2160 if not set (1080)

    # Calculate window position
    window_width, window_height = image_shape[1], image_shape[0]
    x = max((screen_width - window_width) // 2, 0)
    y = max((screen_height - window_height) // 2, 0)

    cv2.moveWindow(window_name, x, y)


def main():
    global points
    
    parser = argparse.ArgumentParser(description="Hand-pick 4 points to define a ROI.")
    parser.add_argument(
        "image_path", 
        type=str,
        help="Path to the input image.",
        nargs="?",  # make the argument optional
        # default="/mnt/e/PandaSet/039/camera/front_camera/00.jpg" 
        default="/mnt/e/kitti/dataset/sequences/00/image_2/000138.png"
    )
    args = parser.parse_args()

    image = cv2.imread(args.image_path)
    if image is None:
        print(f"Error: Could not read the image from {args.image_path}.")
        return
    
    print(
        "[INFO] Hold Shift key and put down 4 points to define a region of interest " +
        "in the following order: top-left, top-right, bottom-right, bottom-left.\n\n" +
        "Click and drag the points to move them. The points are updated in the console.\n\n" +
        "Press 'q' to quit the program."
          )
    
    height, width = image.shape[:2]
    print(f"img_src_size: [{width}, {height}]")
    
    clone = image.copy()  
    cv2.namedWindow("Image")
    center_window("Image", image.shape)
    cv2.setMouseCallback("Image", mouse_callback)

    while True:
        display_image = clone.copy()
        draw_points(display_image, points)
        cv2.imshow("Image", display_image)

        if len(points) == 4:
            print(f"Current points: {points}", end="\r")  # console output

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # q key quits the program
            break

    # print the final state of points before quitting
    if points:
        print(f"\nFinal points: {points}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
