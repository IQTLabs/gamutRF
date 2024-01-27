import os
import argparse
import numpy as np
import cv2 as cv

import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu

# Other useful resources
# Contour approximation https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html

# trying to reverse turbo colormap
# https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/colormap.cpp#L685
# https://gist.github.com/mikhailov-work/6a308c20e494d9e0ccc29036b28faa7a
# unsigned char turbo_srgb_bytes[256][3] = {{48,18,59},{50,21,67},{51,24,74},{52,27,81},{53,30,88},{54,33,95},{55,36,102},{56,39,109},{57,42,115},{58,45,121},{59,47,128},{60,50,134},{61,53,139},{62,56,145},{63,59,151},{63,62,156},{64,64,162},{65,67,167},{65,70,172},{66,73,177},{66,75,181},{67,78,186},{68,81,191},{68,84,195},{68,86,199},{69,89,203},{69,92,207},{69,94,211},{70,97,214},{70,100,218},{70,102,221},{70,105,224},{70,107,227},{71,110,230},{71,113,233},{71,115,235},{71,118,238},{71,120,240},{71,123,242},{70,125,244},{70,128,246},{70,130,248},{70,133,250},{70,135,251},{69,138,252},{69,140,253},{68,143,254},{67,145,254},{66,148,255},{65,150,255},{64,153,255},{62,155,254},{61,158,254},{59,160,253},{58,163,252},{56,165,251},{55,168,250},{53,171,248},{51,173,247},{49,175,245},{47,178,244},{46,180,242},{44,183,240},{42,185,238},{40,188,235},{39,190,233},{37,192,231},{35,195,228},{34,197,226},{32,199,223},{31,201,221},{30,203,218},{28,205,216},{27,208,213},{26,210,210},{26,212,208},{25,213,205},{24,215,202},{24,217,200},{24,219,197},{24,221,194},{24,222,192},{24,224,189},{25,226,187},{25,227,185},{26,228,182},{28,230,180},{29,231,178},{31,233,175},{32,234,172},{34,235,170},{37,236,167},{39,238,164},{42,239,161},{44,240,158},{47,241,155},{50,242,152},{53,243,148},{56,244,145},{60,245,142},{63,246,138},{67,247,135},{70,248,132},{74,248,128},{78,249,125},{82,250,122},{85,250,118},{89,251,115},{93,252,111},{97,252,108},{101,253,105},{105,253,102},{109,254,98},{113,254,95},{117,254,92},{121,254,89},{125,255,86},{128,255,83},{132,255,81},{136,255,78},{139,255,75},{143,255,73},{146,255,71},{150,254,68},{153,254,66},{156,254,64},{159,253,63},{161,253,61},{164,252,60},{167,252,58},{169,251,57},{172,251,56},{175,250,55},{177,249,54},{180,248,54},{183,247,53},{185,246,53},{188,245,52},{190,244,52},{193,243,52},{195,241,52},{198,240,52},{200,239,52},{203,237,52},{205,236,52},{208,234,52},{210,233,53},{212,231,53},{215,229,53},{217,228,54},{219,226,54},{221,224,55},{223,223,55},{225,221,55},{227,219,56},{229,217,56},{231,215,57},{233,213,57},{235,211,57},{236,209,58},{238,207,58},{239,205,58},{241,203,58},{242,201,58},{244,199,58},{245,197,58},{246,195,58},{247,193,58},{248,190,57},{249,188,57},{250,186,57},{251,184,56},{251,182,55},{252,179,54},{252,177,54},{253,174,53},{253,172,52},{254,169,51},{254,167,50},{254,164,49},{254,161,48},{254,158,47},{254,155,45},{254,153,44},{254,150,43},{254,147,42},{254,144,41},{253,141,39},{253,138,38},{252,135,37},{252,132,35},{251,129,34},{251,126,33},{250,123,31},{249,120,30},{249,117,29},{248,114,28},{247,111,26},{246,108,25},{245,105,24},{244,102,23},{243,99,21},{242,96,20},{241,93,19},{240,91,18},{239,88,17},{237,85,16},{236,83,15},{235,80,14},{234,78,13},{232,75,12},{231,73,12},{229,71,11},{228,69,10},{226,67,10},{225,65,9},{223,63,8},{221,61,8},{220,59,7},{218,57,7},{216,55,6},{214,53,6},{212,51,5},{210,49,5},{208,47,5},{206,45,4},{204,43,4},{202,42,4},{200,40,3},{197,38,3},{195,37,3},{193,35,2},{190,33,2},{188,32,2},{185,30,2},{183,29,2},{180,27,1},{178,26,1},{175,24,1},{172,23,1},{169,22,1},{167,20,1},{164,19,1},{161,18,1},{158,16,1},{155,15,1},{152,14,1},{149,13,1},{146,11,1},{142,10,1},{139,9,2},{136,8,2},{133,7,2},{129,6,2},{126,5,2},{122,4,3}};
# turbo_colormap = [[48, 18, 59], [50, 21, 67], [51, 24, 74], [52, 27, 81], [53, 30, 88], [54, 33, 95], [55, 36, 102], [56, 39, 109], [57, 42, 115], [58, 45, 121], [59, 47, 128], [60, 50, 134], [61, 53, 139], [62, 56, 145], [63, 59, 151], [63, 62, 156], [64, 64, 162], [65, 67, 167], [65, 70, 172], [66, 73, 177], [66, 75, 181], [67, 78, 186], [68, 81, 191], [68, 84, 195], [68, 86, 199], [69, 89, 203], [69, 92, 207], [69, 94, 211], [70, 97, 214], [70, 100, 218], [70, 102, 221], [70, 105, 224], [70, 107, 227], [71, 110, 230], [71, 113, 233], [71, 115, 235], [71, 118, 238], [71, 120, 240], [71, 123, 242], [70, 125, 244], [70, 128, 246], [70, 130, 248], [70, 133, 250], [70, 135, 251], [69, 138, 252], [69, 140, 253], [68, 143, 254], [67, 145, 254], [66, 148, 255], [65, 150, 255], [64, 153, 255], [62, 155, 254], [61, 158, 254], [59, 160, 253], [58, 163, 252], [56, 165, 251], [55, 168, 250], [53, 171, 248], [51, 173, 247], [49, 175, 245], [47, 178, 244], [46, 180, 242], [44, 183, 240], [42, 185, 238], [40, 188, 235], [39, 190, 233], [37, 192, 231], [35, 195, 228], [34, 197, 226], [32, 199, 223], [31, 201, 221], [30, 203, 218], [28, 205, 216], [27, 208, 213], [26, 210, 210], [26, 212, 208], [25, 213, 205], [24, 215, 202], [24, 217, 200], [24, 219, 197], [24, 221, 194], [24, 222, 192], [24, 224, 189], [25, 226, 187], [25, 227, 185], [26, 228, 182], [28, 230, 180], [29, 231, 178], [31, 233, 175], [32, 234, 172], [34, 235, 170], [37, 236, 167], [39, 238, 164], [42, 239, 161], [44, 240, 158], [47, 241, 155], [50, 242, 152], [53, 243, 148], [56, 244, 145], [60, 245, 142], [63, 246, 138], [67, 247, 135], [70, 248, 132], [74, 248, 128], [78, 249, 125], [82, 250, 122], [85, 250, 118], [89, 251, 115], [93, 252, 111], [97, 252, 108], [101, 253, 105], [105, 253, 102], [109, 254, 98], [113, 254, 95], [117, 254, 92], [121, 254, 89], [125, 255, 86], [128, 255, 83], [132, 255, 81], [136, 255, 78], [139, 255, 75], [143, 255, 73], [146, 255, 71], [150, 254, 68], [153, 254, 66], [156, 254, 64], [159, 253, 63], [161, 253, 61], [164, 252, 60], [167, 252, 58], [169, 251, 57], [172, 251, 56], [175, 250, 55], [177, 249, 54], [180, 248, 54], [183, 247, 53], [185, 246, 53], [188, 245, 52], [190, 244, 52], [193, 243, 52], [195, 241, 52], [198, 240, 52], [200, 239, 52], [203, 237, 52], [205, 236, 52], [208, 234, 52], [210, 233, 53], [212, 231, 53], [215, 229, 53], [217, 228, 54], [219, 226, 54], [221, 224, 55], [223, 223, 55], [225, 221, 55], [227, 219, 56], [229, 217, 56], [231, 215, 57], [233, 213, 57], [235, 211, 57], [236, 209, 58], [238, 207, 58], [239, 205, 58], [241, 203, 58], [242, 201, 58], [244, 199, 58], [245, 197, 58], [246, 195, 58], [247, 193, 58], [248, 190, 57], [249, 188, 57], [250, 186, 57], [251, 184, 56], [251, 182, 55], [252, 179, 54], [252, 177, 54], [253, 174, 53], [253, 172, 52], [254, 169, 51], [254, 167, 50], [254, 164, 49], [254, 161, 48], [254, 158, 47], [254, 155, 45], [254, 153, 44], [254, 150, 43], [254, 147, 42], [254, 144, 41], [253, 141, 39], [253, 138, 38], [252, 135, 37], [252, 132, 35], [251, 129, 34], [251, 126, 33], [250, 123, 31], [249, 120, 30], [249, 117, 29], [248, 114, 28], [247, 111, 26], [246, 108, 25], [245, 105, 24], [244, 102, 23], [243, 99, 21], [242, 96, 20], [241, 93, 19], [240, 91, 18], [239, 88, 17], [237, 85, 16], [236, 83, 15], [235, 80, 14], [234, 78, 13], [232, 75, 12], [231, 73, 12], [229, 71, 11], [228, 69, 10], [226, 67, 10], [225, 65, 9], [223, 63, 8], [221, 61, 8], [220, 59, 7], [218, 57, 7], [216, 55, 6], [214, 53, 6], [212, 51, 5], [210, 49, 5], [208, 47, 5], [206, 45, 4], [204, 43, 4], [202, 42, 4], [200, 40, 3], [197, 38, 3], [195, 37, 3], [193, 35, 2], [190, 33, 2], [188, 32, 2], [185, 30, 2], [183, 29, 2], [180, 27, 1], [178, 26, 1], [175, 24, 1], [172, 23, 1], [169, 22, 1], [167, 20, 1], [164, 19, 1], [161, 18, 1], [158, 16, 1], [155, 15, 1], [152, 14, 1], [149, 13, 1], [146, 11, 1], [142, 10, 1], [139, 9, 2], [136, 8, 2], [133, 7, 2], [129, 6, 2], [126, 5, 2], [122, 4, 3]]


def dji_yolo_labeller(rect):
    x, y, w, h = rect
    if w < 100:
        return 1
    return 0


dji_args = {
    "invert": True,
    "pre_threshold": 1,
    "histogram_equalization": False,
    "pre_thresh_morphology": [],
    "kernel_open": cv.getStructuringElement(
        cv.MORPH_RECT,
        (2, 2),  # (10,10)
    ),  # np.ones((3, 3), np.uint8),
    "kernel_close": cv.getStructuringElement(
        cv.MORPH_RECT, (24, 12)
    ),  # np.ones((15, 15), np.uint8),
    "threshold_op": 0,
    "post_thresh_morphology": [],
    "bilateral_kernel_size": 8,
    "group_horizontal": False,
    "area_threshold": 0.7,
    "yolo_label": dji_yolo_labeller,
    "vertical": False,
    "dc_block": None,
    "horizontal_adjust": 0,
    "custom_rect_filter": None,
}


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
    # Note: msk must use Viridis colormap
    "invert": False,
    "pre_threshold": 1,
    "histogram_equalization": False,
    "pre_thresh_morphology": [],
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

        if callable(yolo_label):
            yolo_label_val = yolo_label(rect)
        else:
            yolo_label_val = yolo_label

        yolo_boxes.append(
            [yolo_label_val, x_center_yolo, y_center_yolo, w_yolo, h_yolo]
        )

    cv_plot(np.hstack((original_img, img)), "final image")
    # if debug:
    #     print(f"{yolo_boxes=}")
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
    "dji": dji_args,
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
