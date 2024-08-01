import cv2
import numpy as np
from rembg import remove
from PIL import Image
import napari
from qtpy.QtWidgets import QPushButton
import scipy.spatial
import math
import sys

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[3] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[2] = pts[np.argmax(diff)]

    return rect

def main(input_image_path, output_image_path):
    # Load image
    image = Image.open(input_image_path)
    image_np = np.array(image)

    # Remove background and get mask
    mask = remove(image, only_mask=True)

    # Threshold the grayscale mask
    ret, thresh = cv2.threshold(np.array(mask), 127, 255, 0)

    # Find contours in the thresholded mask
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Create an empty image for drawing the contours
    tmp = np.zeros_like(image_np)

    # Draw contours on the empty image
    boundary = cv2.drawContours(tmp, contours, -1, (255, 255, 255), 1)
    boundary[boundary > 0] = 255

    # Find the largest contour which we assume to be the frame
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
    screenCnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    # Initial points from detected contour
    points = [tuple(pt[0]) for pt in screenCnt]

    # Convert points to numpy array and correct coordinate system (cv2 to napari)
    points_array = np.array(points)[:, [1, 0]]  # Convert (x, y) to (y, x)

    # Function to store points and perform perspective transform
    def store_and_transform():
        global points_array
        points_array = points_layer.data
        if len(points_array) != 4:
            print("You need to select exactly 4 points.")
            return

        # Order the points
        p = order_points(np.array(points_array)[:, [1, 0]])  # Convert back to (x, y)

        # Widths and heights of the projected image
        w1 = scipy.spatial.distance.euclidean(p[0], p[1])
        w2 = scipy.spatial.distance.euclidean(p[2], p[3])
        h1 = scipy.spatial.distance.euclidean(p[0], p[2])
        h2 = scipy.spatial.distance.euclidean(p[1], p[3])

        w = max(w1, w2)
        h = max(h1, h2)

        # Make numpy arrays and append 1 for linear algebra
        m1 = np.array((p[0][0], p[0][1], 1)).astype('float32')
        m2 = np.array((p[1][0], p[1][1], 1)).astype('float32')
        m3 = np.array((p[2][0], p[2][1], 1)).astype('float32')
        m4 = np.array((p[3][0], p[3][1], 1)).astype('float32')

        # Calculate the focal distance
        k2 = np.dot(np.cross(m1, m4), m3) / np.dot(np.cross(m2, m4), m3)
        k3 = np.dot(np.cross(m1, m4), m2) / np.dot(np.cross(m3, m4), m2)

        n2 = k2 * m2 - m1
        n3 = k3 * m3 - m1

        n21 = n2[0]
        n22 = n2[1]
        n23 = n2[2]

        n31 = n3[0]
        n32 = n3[1]
        n33 = n3[2]

        u0 = (image_np.shape[1]) / 2.0
        v0 = (image_np.shape[0]) / 2.0

        f = math.sqrt(np.abs((1.0 / (n23 * n33)) * (
                    (n21 * n31 - (n21 * n33 + n23 * n31) * u0 + n23 * n33 * u0 * u0) + (
                        n22 * n32 - (n22 * n33 + n23 * n32) * v0 + n23 * n33 * v0 * v0))))

        A = np.array([[f, 0, u0], [0, f, v0], [0, 0, 1]]).astype('float32')

        At = np.transpose(A)
        Ati = np.linalg.inv(At)
        Ai = np.linalg.inv(A)

        # Calculate the real aspect ratio
        ar_real = math.sqrt(np.dot(np.dot(np.dot(n2, Ati), Ai), n2) / np.dot(np.dot(np.dot(n3, Ati), Ai), n3))

        if ar_real < (w / h):
            W = int(w)
            H = int(W / ar_real)
        else:
            H = int(h)
            W = int(ar_real * H)

        pts1 = np.array(p).astype('float32')
        pts2 = np.float32([[0, 0], [W, 0], [0, H], [W, H]])

        # Project the image with the new w/h
        M = cv2.getPerspectiveTransform(pts1, pts2)

        # Warp the perspective to the new dimensions
        warped_img = cv2.warpPerspective(image_np, M, (W, H))

        # Convert the warped image to RGB if necessary
        if len(warped_img.shape) == 3 and warped_img.shape[2] == 3:
            warped_img = cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)

        # Save the final image with its new dimensions
        cv2.imwrite(output_image_path, warped_img)
        print("Transformed image saved at:", output_image_path)
        viewer.close()

    # Display the image using napari
    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(image_np, name='Original Image')
        points_layer = viewer.add_points(points_array, size=50, face_color='red', name='Edge Points', symbol='disc')

        # Add a button to store points and close viewer
        store_button = QPushButton('Store Points and Transform')
        store_button.clicked.connect(store_and_transform)
        viewer.window.add_dock_widget(store_button, area='right')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_image_path> <output_image_path>")
        sys.exit(1)
    
    input_image_path = sys.argv[1]
    output_image_path = sys.argv[2]

    main(input_image_path, output_image_path)
