import os
import argparse
import numpy as np
import cv2 as cv

import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu

# Other useful resources
# Contour approximation https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html


def rect_filter_msk(rects, shape):
    max_width = 95
    max_height = 25

    good_rects = []
    if len(rects) > 10:
        rects = []

    # print(f"{rects=}")
    for rect in rects:
        x, y, w, h = rect

        if (w < 0.5 * max_width) and (h < 0.5 * max_height):
            continue
        if (
            (w > 1.1 * max_width)
            or (w < 0.1 * max_width)
            or (h < 0.1 * max_height)
            or (h > 1.2 * max_height)
        ):
            continue

        if w < (0.7 * max_width) and w > (0.19 * max_width):
            center = x + (0.5 * w)
            x = int(max(center - (0.5 * max_width), 0))
            w = int(min(center + (0.5 * max_width) - x, shape[1] - 1 - x))

        if w * h < (0.5 * max_width * max_height):
            continue
        good_rects.append([x, y, w, h])
    # print(f"{good_rects=}")
    return good_rects


msk_args = {
    "invert": False,
    "pre_threshold": 1,
    "histogram_equalization": False,
    "pre_thresh_morphology": [
        # {
        #     "morph_func": cv.MORPH_CLOSE,
        #     "kernel": cv.getStructuringElement(cv.MORPH_RECT, (50,30)),
        # },
        # {
        #     "morph_func": cv.MORPH_OPEN,
        #     "kernel": cv.getStructuringElement(cv.MORPH_RECT, (3,3)),
        # },
        # {
        #     "morph_func": cv.MORPH_CLOSE,
        #     "kernel": cv.getStructuringElement(cv.MORPH_RECT, (24,12)),
        # },
    ],
    "kernel_open": cv.getStructuringElement(
        cv.MORPH_RECT,
        (3, 3),  # (10,10)
    ),  # np.ones((3, 3), np.uint8),
    "kernel_close": cv.getStructuringElement(
        cv.MORPH_RECT, (24, 12)
    ),  # np.ones((15, 15), np.uint8),
    "threshold_op": 0,
    "post_thresh_morphology": [],
    "bilateral_kernel_size": 8,
    "group_horizontal": False,
    "area_threshold": 0.7,
    "yolo_label": 0,
    "vertical": False,
    "dc_block": None,
    "horizontal_adjust": 0,
    "custom_rect_filter": rect_filter_msk,
}


def rect_filter_fhss_css(rects, shape):
    max_width = 100
    max_height = 65

    good_rects = []
    if len(rects) > 10:
        rects = []

    for rect in rects:
        x, y, w, h = rect

        if (w < 0.5 * max_width) and (h < 0.5 * max_height):
            continue
        if (w > 1.1 * max_width) or (h > 1.1 * max_height):
            continue

        if w < (0.7 * max_width) and w > (0.22 * max_width):
            center = x + (0.5 * w)
            x = int(max(center - (max_width * 0.5), 0))
            w = int(min(center + (max_width * 0.5) - x, shape[1] - 1 - x))
        good_rects.append([x, y, w, h])
    return good_rects


fhss_css_args = {
    "invert": True,
    "pre_threshold": 0,
    "histogram_equalization": False,
    "pre_thresh_morphology": [
        # {
        #     "morph_func": cv.MORPH_CLOSE,
        #     "kernel": cv.getStructuringElement(cv.MORPH_RECT, (50,30)),
        # },
        {
            "morph_func": cv.MORPH_OPEN,
            "kernel": cv.getStructuringElement(cv.MORPH_RECT, (3, 3)),
        },
        {
            "morph_func": cv.MORPH_CLOSE,
            "kernel": cv.getStructuringElement(cv.MORPH_RECT, (12, 12)),
        },
        # {
        #     "morph_func": cv.MORPH_OPEN,
        #     "kernel": cv.getStructuringElement(cv.MORPH_RECT, (10,10)),
        # },
        # {
        #     "morph_func": cv.MORPH_CLOSE,
        #     "kernel": cv.getStructuringElement(cv.MORPH_RECT, (10,10)),
        # },
        # {
        #     "morph_func": cv.MORPH_OPEN,
        #     "kernel": cv.getStructuringElement(cv.MORPH_RECT, (10,10)),
        # },
        # {
        #     "morph_func": cv.MORPH_CLOSE,
        #     "kernel": cv.getStructuringElement(cv.MORPH_RECT, (10,10)),
        # },
    ],
    "kernel_open": cv.getStructuringElement(
        cv.MORPH_RECT,
        (12, 12),  # (10,10)
    ),  # np.ones((3, 3), np.uint8),
    "kernel_close": cv.getStructuringElement(
        cv.MORPH_RECT, (15, 15)
    ),  # np.ones((15, 15), np.uint8),
    "threshold_op": 0,
    "post_thresh_morphology": [],
    "bilateral_kernel_size": 8,
    "group_horizontal": False,
    "area_threshold": 0.7,
    "yolo_label": 0,
    "vertical": False,
    "dc_block": None,
    "horizontal_adjust": 0,
    "custom_rect_filter": rect_filter_fhss_css,
}


tbs_crs_args = {
    "invert": True,
    "pre_threshold": 1,
    "histogram_equalization": True,
    "pre_thresh_morphology": [
        {
            "morph_func": cv.erode,
            "kernel": np.ones((2, 1), np.uint8),
        },
        {
            "morph_func": cv.dilate,
            "kernel": np.ones((2, 1), np.uint8),
        },
        {
            "morph_func": cv.erode,
            "kernel": np.ones((4, 1), np.uint8),
        },
        {
            "morph_func": cv.dilate,
            "kernel": np.ones((10, 2), np.uint8),
        },
        {
            "morph_func": cv.erode,
            "kernel": np.ones((10, 1), np.uint8),
            "kwargs": {
                "iterations": 2,
            },
        },
        {
            "morph_func": cv.dilate,
            "kernel": np.ones((15, 2), np.uint8),
        },
    ],
    "kernel_open": cv.getStructuringElement(
        cv.MORPH_RECT, (4, 50)
    ),  # np.ones((50, 4), np.uint8),
    "kernel_close": cv.getStructuringElement(
        cv.MORPH_RECT, (6, 25)
    ),  # np.ones((25, 6), np.uint8)
    "threshold_op": "otsu",
    "post_thresh_morphology": [
        {
            "morph_func": cv.dilate,
            "kernel": np.ones((1, 4), np.uint8),
        }
    ],
    "bilateral_kernel_size": 2,
    "group_horizontal": False,
    "area_threshold": 0.7,
    "yolo_label": 0,
    "vertical": True,
    "dc_block": 5,
    "horizontal_adjust": -4,
}

wifi_args = {
    "invert": True,
    "pre_threshold": -1,
    "histogram_equalization": False,
    "pre_thresh_morphology": [
        {
            "morph_func": cv.MORPH_CLOSE,
            "kernel": cv.getStructuringElement(cv.MORPH_RECT, (65, 1)),
        }
    ],
    "kernel_open": cv.getStructuringElement(
        cv.MORPH_RECT, (65, 1)
    ),  # np.ones((3, 3), np.uint8),
    "kernel_close": cv.getStructuringElement(
        cv.MORPH_RECT, (65, 1)
    ),  # np.ones((15, 15), np.uint8),
    "threshold_op": "otsu",
    "post_thresh_morphology": [],
    "bilateral_kernel_size": 8,
    "group_horizontal": True,
    "area_threshold": 0.7,
    "yolo_label": 0,
    "vertical": False,
    "dc_block": None,
    "horizontal_adjust": 0,
}

default_args = {
    "invert": None,
    "pre_threshold": None,
    "histogram_equalization": None,
    "pre_thresh_morphology": None,
    "kernel_open": None,
    "kernel_close": None,
    "threshold_op": None,
    "post_thresh_morphology": None,
    "bilateral_kernel_size": None,
    "group_horizontal": None,
    "area_threshold": None,
    "yolo_label": None,
    "vertical": None,
    "dc_block": None,
    "horizontal_adjust": None,
    "custom_rect_filter": None,
}
debug = True


def multi_otsu(imgray):
    # Applying multi-Otsu threshold for the default value, generating
    # three classes.
    thresholds = threshold_multiotsu(imgray)

    if debug:
        # Using the threshold values, we generate the three regions.
        regions = np.digitize(imgray, bins=thresholds)
        # print(thresholds)
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3.5))

        # Plotting the original image.
        ax[0].imshow(imgray, cmap="gray")
        ax[0].set_title("Original")
        ax[0].axis("off")

        # Plotting the histogram and the two thresholds obtained from
        # multi-Otsu.
        ax[1].hist(imgray.ravel(), bins=255)
        ax[1].set_title("Histogram")
        for thresh in thresholds:
            ax[1].axvline(thresh, color="r")

        # Plotting the Multi Otsu result.
        ax[2].imshow(regions, cmap="jet")
        ax[2].set_title("Multi-Otsu result")
        ax[2].axis("off")

        plt.subplots_adjust()

        plt.draw()
        plt.waitforbuttonpress()  # this will wait for indefinite time
        plt.close(fig)
    return thresholds


def union(a, b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0] + a[2], b[0] + b[2]) - x
    h = max(a[1] + a[3], b[1] + b[3]) - y
    return (x, y, w, h)


def check_horizontal_intersect(a, b):
    return (a[1] <= (b[1] + b[3])) and (b[1] <= (a[1] + a[3]))


def group_horizontal_rects(rects):
    i = 0
    while i < len(rects):
        x, y, w, h = rects[i]
        j = i + 1
        while j < len(rects):
            x2, y2, w2, h2 = rects[j]
            if check_horizontal_intersect(rects[i], rects[j]):
                rects[i] = union(rects[i], rects[j])
                rects.pop(j)
                j = i
            j += 1
        i += 1
    return rects


def cv_plot(img, title):
    if debug:
        cv.imshow(title, img)
        k = cv.waitKey(0)

        if k in [3, 27]:
            exit()
        cv.destroyAllWindows()


def label(
    filename,
    invert,
    kernel_open,
    kernel_close,
    bilateral_kernel_size,
    group_horizontal,
    area_threshold,
    yolo_label,
    vertical,
    dc_block,
    horizontal_adjust,
    pre_threshold,
    threshold_op,
    post_thresh_morphology,
    histogram_equalization,
    pre_thresh_morphology,
    custom_rect_filter,
):
    print(f"Processing {filename}")
    if debug:
        print(
            f"Currently in debugging mode. While using GUI window, press any key to continue and ESC or Ctrl+c to exit."
        )
    img = cv.imread(filename)
    original_img = img.copy()
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv_plot(np.hstack((img, cv.cvtColor(imgray, cv.COLOR_GRAY2RGB))), "imgray")

    # invert image (depends on colormap)
    if invert:
        imgray_new = cv.bitwise_not(imgray)
        cv_plot(np.hstack((imgray, imgray_new)), "imgray invert")
        imgray = imgray_new
    imgray_original = imgray.copy()

    # display multi otsu thresholding (informational)
    if debug:
        multi_thresh = multi_otsu(imgray)

    if pre_threshold >= 0:
        multi_thresh = multi_otsu(imgray)
        ret, imgray_new = cv.threshold(
            imgray, multi_thresh[pre_threshold], 255, cv.THRESH_TOZERO
        )
        cv_plot(np.hstack((imgray, imgray_new)), "thresh trunc")
        imgray = imgray_new

    if histogram_equalization:
        # imgray = cv.equalizeHist(imgray)
        # cv_plot(np.hstack((imgray_original, imgray)), "histogram")

        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        imgray_new = clahe.apply(imgray)
        cv_plot(np.hstack((imgray, imgray_new)), "histogram clahe")
        imgray = imgray_new

    for morphology in pre_thresh_morphology:
        morph_func = morphology["morph_func"]
        kernel = morphology["kernel"]
        kwargs = morphology["kwargs"] if "kwargs" in morphology else {}
        if morph_func in [cv.dilate, cv.erode]:
            imgray_new = morph_func(imgray, kernel, **kwargs)
            cv_plot(
                np.hstack((imgray, imgray_new)),
                f"pre_thresh_morphology: {morph_func.__name__}",
            )
            imgray = imgray_new
        elif morph_func in [cv.MORPH_OPEN, cv.MORPH_CLOSE]:
            imgray_new = cv.morphologyEx(imgray, morph_func, kernel, **kwargs)
            cv_plot(
                np.hstack((imgray, imgray_new)), f"pre_thresh_morphology: {morph_func}"
            )
            imgray = imgray_new
        else:
            raise ValueError(f"pre_thresh_morphology function not valid: {morph_func}")

    # imgray_new = cv.medianBlur(imgray,3)
    # cv_plot(np.hstack((imgray, imgray_new)), "median blur")
    # imgray=imgray_new

    # opening and closing
    # https://docs.opencv.org/3.4/d9/d61/tutorial_py_morphological_ops.html
    # https://docs.opencv.org/3.4/d3/dbe/tutorial_opening_closing_hats.html
    # kernel_open = np.ones((3, 3), np.uint8)
    imgray_new = cv.morphologyEx(imgray, cv.MORPH_OPEN, kernel_open)
    cv_plot(np.hstack((imgray, imgray_new)), "open morph")
    imgray = imgray_new

    # kernel_close = np.ones((15, 15), np.uint8)
    imgray_new = cv.morphologyEx(imgray, cv.MORPH_CLOSE, kernel_close)
    cv_plot(np.hstack((imgray, imgray_new)), "close morph")
    imgray = imgray_new

    # bilateral smoothing
    # https://docs.opencv.org/3.4/dc/dd3/tutorial_gausian_median_blur_bilateral_filter.html
    # bilateral_kernel_size = 8#32
    imgray_new = cv.bilateralFilter(
        imgray,
        bilateral_kernel_size,
        bilateral_kernel_size * 2,
        bilateral_kernel_size / 2,
    )
    cv_plot(np.hstack((imgray, imgray_new)), "bilateral smoothing")
    imgray = imgray_new

    # thresholding (adaptive gaussian)
    # thresh = cv.adaptiveThreshold(imgray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,21,-5)
    # cv.imshow("adaptive thresh", thresh)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    if debug:
        multi_thresh = multi_otsu(imgray)

    if threshold_op == "otsu":
        # thresholding (otsu)
        ret, thresh = cv.threshold(imgray, 0, 255, cv.THRESH_OTSU)
        cv_plot(np.hstack((imgray, thresh)), "otsu threshold")
    elif isinstance(threshold_op, int):
        multi_thresh = multi_otsu(imgray)
        ret, thresh = cv.threshold(
            imgray, multi_thresh[threshold_op], 255, cv.THRESH_TOZERO
        )
        cv_plot(np.hstack((imgray, thresh)), f"thresh trunc {threshold_op}")
    else:
        raise ValueError("threshold_op must be str or int")

    # thresholding (global, hardcoded)
    # ret, thresh = cv.threshold(imgray, 160, 255, 0)
    # cv_plot(thresh, "global_thresh")

    for morphology in post_thresh_morphology:
        morph_func = morphology["morph_func"]
        kernel = morphology["kernel"]
        kwargs = morphology["kwargs"] if "kwargs" in morphology else {}
        if morph_func in [cv.dilate, cv.erode]:
            thresh_new = morph_func(thresh, kernel, **kwargs)
            cv_plot(
                np.hstack((thresh, thresh_new)),
                f"post_thresh_morphology: {morph_func.__name__}",
            )
            thresh = thresh_new
        elif morph_func in [cv.MORPH_OPEN, cv.MORPH_CLOSE]:
            thresh_new = cv.morphologyEx(thresh, morph_func, kernel, **kwargs)
            cv_plot(
                np.hstack((thresh, thresh_new)), f"pre_thresh_morphology: {morph_func}"
            )
            thresh = thresh_new
        else:
            raise ValueError(f"post_thresh_morphology function not valid: {morph_func}")

    # find contours
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # remove internal contours
    # https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html
    good_contours = []
    for i in range(len(contours)):
        if hierarchy[0, i, 3] == -1:
            good_contours.append(contours[i])
    contours = good_contours

    # convert contours to bounding rectangles
    rects = [list(cv.boundingRect(cnt)) for cnt in contours]

    if custom_rect_filter:
        rects = custom_rect_filter(rects, img.shape)

    # group contours across horizontal axis
    if group_horizontal:
        rects = group_horizontal_rects(rects)

    # don't include DC bias
    if dc_block:
        good_rects = []
        for rect in rects:
            if (
                abs(rect[0] - img.shape[1] / 2) > dc_block
                and abs(rect[0] + rect[2] - img.shape[1] / 2) > dc_block
            ):
                good_rects.append(rect)
        rects = good_rects

    # manually adjust bounding boxes
    if horizontal_adjust:
        for rect in rects:
            rect[0] += horizontal_adjust

    # remove large contours (covering entire image)
    image_area = img.shape[0] * img.shape[1]
    good_rects = []
    for rect in rects:
        if rect[2] * rect[3] < image_area * area_threshold:
            good_rects.append(rect)
    rects = good_rects

    yolo_boxes = []

    # draw rectangles
    for rect in rects:
        x, y, w, h = rect
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        x_center_yolo = (x + (0.5 * w)) / img.shape[1]
        y_center_yolo = (y + (0.5 * h)) / img.shape[0]
        w_yolo = w / img.shape[1]
        h_yolo = h / img.shape[0]

        yolo_boxes.append([yolo_label, x_center_yolo, y_center_yolo, w_yolo, h_yolo])

    cv_plot(np.hstack((original_img, img)), "final image")

    if not debug:
        label_dir = os.path.join(os.path.dirname(filename), "labels")
        no_label_dir = os.path.join(os.path.dirname(filename), "no_labels")
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        if not os.path.exists(no_label_dir):
            os.makedirs(no_label_dir)

        yolo_label_filename = os.path.join(
            os.path.dirname(filename),
            "labels",
            os.path.splitext(os.path.basename(filename))[0] + ".txt",
        )
        print(f"Writing YOLOv8 labels to {yolo_label_filename}")
        with open(yolo_label_filename, "w") as yolo_label_file:
            for box in yolo_boxes:
                row = " ".join(str(b) for b in box) + "\n"
                yolo_label_file.write(row)

        if yolo_boxes:
            label_img_filename = os.path.join(
                os.path.dirname(filename),
                "labels",
                "label_" + os.path.basename(filename),
            )
            print(f"Writing labelled image to {label_img_filename}")
            cv.imwrite(label_img_filename, img)
        else:
            no_label_img_filename = os.path.join(
                os.path.dirname(filename), "no_labels", os.path.basename(filename)
            )
            print(f"Writing non-labelled image to {no_label_img_filename}")
            cv.imwrite(no_label_img_filename, img)


args_dict = {
    "wifi": wifi_args,
    "tbs_crossfire": tbs_crs_args,
    "fhss_css": fhss_css_args,
    "msk": msk_args,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filepath",
        type=str,
        default="",
        help="Filename or directory.",
    )
    parser.add_argument(
        "signal_type",
        type=str,
        choices=list(args_dict.keys()),
        help="Type of signal. Will decide labelling parameters.",
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Display image plots.",
    )
    args = parser.parse_args()

    filepath = args.filepath
    label_args = default_args
    label_args.update(args_dict[args.signal_type])
    global debug
    debug = args.debug

    img_extensions = (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")

    # get files
    files = []
    if os.path.isfile(filepath):
        files.append(filepath)
    elif os.path.isdir(filepath):
        files.extend(
            [
                os.path.join(filepath, filename)
                for filename in os.listdir(filepath)
                if os.path.isfile(os.path.join(filepath, filename))
                and filename.lower().endswith(img_extensions)
            ]
        )
    else:
        raise ValueError("filepath must be existing directory or filename")

    # loop through files
    for i, img_filename in enumerate(files):
        print(f"\n[{i+1}/{len(files)}]")
        label(img_filename, **label_args)


if __name__ == "__main__":
    main()
