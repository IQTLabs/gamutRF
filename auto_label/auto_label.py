import os
from argparse import ArgumentParser
import numpy as np
import cv2 as cv


def label(filename):
    img = cv.imread(filename)
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # opening and closing
    # https://docs.opencv.org/3.4/d9/d61/tutorial_py_morphological_ops.html
    # https://docs.opencv.org/3.4/d3/dbe/tutorial_opening_closing_hats.html
    kernel = np.ones((5, 5), np.uint8)
    imgray = cv.morphologyEx(imgray, cv.MORPH_OPEN, kernel)
    imgray = cv.morphologyEx(imgray, cv.MORPH_CLOSE, kernel)

    # bilateral smoothing
    # https://docs.opencv.org/3.4/dc/dd3/tutorial_gausian_median_blur_bilateral_filter.html
    bilateral_kernel = 8
    imgray = cv.bilateralFilter(
        imgray, bilateral_kernel, bilateral_kernel * 2, bilateral_kernel / 2
    )
    ret, thresh = cv.threshold(imgray, 140, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # remove internal contours
    # https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html
    good_contours = []
    for i in range(len(contours)):
        if hierarchy[0, i, 3] == -1:
            good_contours.append(contours[i])
    contours = good_contours

    # draw rectangels over all contours
    for cnt in contours:
        # Contour approximation https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
        # epsilon = 0.01*cv.arcLength(cnt,True)
        # cnt = cv.approxPolyDP(cnt,epsilon,True)

        # Convex hull
        # cnt = cv.convexHull(cnt)

        x, y, w, h = cv.boundingRect(cnt)
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv.imshow("Image", img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "filepath",
        type=str,
        default="",
        help="Filename or directory.",
    )
    args = parser.parse_args()

    filepath = args.filepath

    # get files
    files = []
    if os.path.isfile(filepath):
        files.append(filepath)
    elif os.path.isdir(filepath):
        files.extend(
            [os.path.join(filepath, filename) for filename in os.listdir(filepath)]
        )
    else:
        raise ValueError("filepath must be existing directory or filename")

    # loop through files
    for img_filename in files:
        label(img_filename)


if __name__ == "__main__":
    main()
