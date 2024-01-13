import os
import argparse
import numpy as np
import cv2 as cv

import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu

# Other useful resources
# Contour approximation https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html

debug = True

def multi_otsu(imgray):
    # Applying multi-Otsu threshold for the default value, generating
    # three classes.
    thresholds = threshold_multiotsu(imgray)

    if debug:
        # Using the threshold values, we generate the three regions.
        regions = np.digitize(imgray, bins=thresholds)
        #print(thresholds)
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3.5))

        # Plotting the original image.
        ax[0].imshow(imgray, cmap='gray')
        ax[0].set_title('Original')
        ax[0].axis('off')

        # Plotting the histogram and the two thresholds obtained from
        # multi-Otsu.
        ax[1].hist(imgray.ravel(), bins=255)
        ax[1].set_title('Histogram')
        for thresh in thresholds:
            ax[1].axvline(thresh, color='r')

        # Plotting the Multi Otsu result.
        ax[2].imshow(regions, cmap='jet')
        ax[2].set_title('Multi-Otsu result')
        ax[2].axis('off')

        plt.subplots_adjust()

        plt.draw()
        plt.waitforbuttonpress() # this will wait for indefinite time
        plt.close(fig)
    return thresholds

def union(a,b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0]+a[2], b[0]+b[2]) - x
    h = max(a[1]+a[3], b[1]+b[3]) - y
    return (x, y, w, h)

def check_horizontal_intersect(a,b):
    return (a[1] <= (b[1]+b[3])) and (b[1] <= (a[1] + a[3]))

def group_horizontal_rects(rects):
    i = 0
    while i < len(rects):
        x, y, w, h = rects[i]
        j = i+1
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
        cv.waitKey(0)
        cv.destroyAllWindows()

def label(filename, invert, custom_morphology, kernel_open, kernel_close, bilateral_kernel_size, group_horizontal, area_threshold, yolo_label, vertical, dc_block, horizontal_adjust ):
    print(filename)
    img = cv.imread(filename)
    original_img = img.copy()
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # invert image (depends on colormap)
    if invert: 
        imgray = cv.bitwise_not(imgray)
    cv_plot(imgray, "imgray")
    imgray_original = imgray.copy()


    # display multi otsu thresholding (informational)
    # if debug:
    #     multi_thresh = multi_otsu(imgray)
    
    if vertical: 
        multi_thresh = multi_otsu(imgray)

        ret, imgray_new = cv.threshold(imgray, multi_thresh[1], 255, cv.THRESH_TOZERO)
        cv_plot(np.hstack((imgray, imgray_new)), "thresh trunc")
        imgray = imgray_new


        # imgray = cv.equalizeHist(imgray)
        # cv_plot(np.hstack((imgray_original, imgray)), "histogram")

        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        imgray_new = clahe.apply(imgray)
        cv_plot(np.hstack((imgray, imgray_new)), "histogram clahe")
        imgray = imgray_new

    

        kernel = np.ones((2,1),np.uint8)
        imgray_new = cv.erode(imgray,kernel,iterations = 1)
        cv_plot(np.hstack((imgray, imgray_new)), "erosions")
        imgray = imgray_new

        kernel = np.ones((2,1), np.uint8)
        imgray_new = cv.dilate(imgray,kernel,iterations = 1)
        cv_plot(np.hstack((imgray, imgray_new)), "dilation")
        imgray = imgray_new

  

        kernel = np.ones((4,1),np.uint8)
        imgray_new = cv.erode(imgray,kernel,iterations = 1)
        cv_plot(np.hstack((imgray, imgray_new)), "erosions")
        imgray = imgray_new

        kernel = np.ones((10,2), np.uint8)
        imgray_new = cv.dilate(imgray,kernel,iterations = 1)
        cv_plot(np.hstack((imgray, imgray_new)), "dilation")
        imgray = imgray_new

        kernel = np.ones((10,1),np.uint8) # (10,2)
        imgray_new = cv.erode(imgray,kernel,iterations = 2)
        cv_plot(np.hstack((imgray, imgray_new)), "erosions")
        imgray = imgray_new

        kernel = np.ones((15, 2), np.uint8)
        imgray_new = cv.dilate(imgray,kernel,iterations = 1)
        cv_plot(np.hstack((imgray, imgray_new)), "dilation")
        imgray = imgray_new

        # imgray_new = cv.medianBlur(imgray,3)
        # cv_plot(np.hstack((imgray, imgray_new)), "median blur")
        # imgray=imgray_new


    for morphology, kernel in custom_morphology:
        imgray_new = cv.morphologyEx(imgray, morphology, kernel)
        
        cv_plot(np.hstack((imgray, imgray_new)), "custom morphology")
        imgray = imgray_new

    # (moved to wifi args dict) custom morphology for use with Wi-Fi (long horizontal filter to connect horizontal components)
    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (65, 1))
    # imgray = cv.morphologyEx(imgray, cv.MORPH_CLOSE, kernel)
    # cv_plot(imgray, "wifi kernel")
 
    # opening and closing
    # https://docs.opencv.org/3.4/d9/d61/tutorial_py_morphological_ops.html
    # https://docs.opencv.org/3.4/d3/dbe/tutorial_opening_closing_hats.html
    #kernel_open = np.ones((3, 3), np.uint8)
    imgray_new = cv.morphologyEx(imgray, cv.MORPH_OPEN, kernel_open)
    cv_plot(np.hstack((imgray, imgray_new)), "open morph")
    imgray = imgray_new
    
    #kernel_close = np.ones((15, 15), np.uint8)
    imgray_new = cv.morphologyEx(imgray, cv.MORPH_CLOSE, kernel_close)
    cv_plot(np.hstack((imgray, imgray_new)), "close morph")
    imgray = imgray_new


    # bilateral smoothing
    # https://docs.opencv.org/3.4/dc/dd3/tutorial_gausian_median_blur_bilateral_filter.html
    #bilateral_kernel_size = 8#32
    imgray_new = cv.bilateralFilter(
        imgray, bilateral_kernel_size, bilateral_kernel_size * 2, bilateral_kernel_size / 2
    )
    cv_plot(np.hstack((imgray, imgray_new)), "bilateral smoothing")
    imgray = imgray_new


    # thresholding (adaptive gaussian)
    # thresh = cv.adaptiveThreshold(imgray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,21,-5)
    # cv.imshow("adaptive thresh", thresh)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # thresholding (otsu)
    ret,thresh = cv.threshold(imgray,0,255,cv.THRESH_OTSU)
    cv_plot(thresh, "otsu threshold")

    # thresholding (global, hardcoded)
    # ret, thresh = cv.threshold(imgray, 160, 255, 0)
    # cv_plot(thresh, "global_thresh")

    
    for morph_func, kernel in post_thresh_morphology:
        if morph_func in [cv.dilate or cv.erode]:
            thresh_new = morph_func(thresh, kernel)
            cv_plot(np.hstack((thresh, thresh_new)), "post_thresh_morphology")
        elif morph_func in [cv.MORPH_OPEN, cv.MORPH_CLOSE]:
            thresh_new = cv.morphologyEx(thresh, morph_func, kernel)
            cv_plot(np.hstack((thresh, thresh_new)), "post_thresh_morphology")
        else: 
            raise ValueError("post_thresh_morphology function not valid")

    # if vertical:
    #     kernel = np.ones((1, 4), np.uint8)
    #     thresh_new = cv.dilate(thresh,kernel,iterations = 1)
    #     cv_plot(np.hstack((thresh, thresh_new)), "dilation")
    #     thresh = thresh_new

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

    # group contours across horizontal axis
    if group_horizontal: 
        rects = group_horizontal_rects(rects)

    # remove large contours (covering entire image)
    image_area = img.shape[0] * img.shape[1]
    good_rects = []
    for rect in rects: 
        if rect[2]*rect[3] < image_area*area_threshold:
            good_rects.append(rect)
    rects = good_rects
    
    # don't include DC bias 
    if dc_block:
        good_rects = []
        for rect in rects:
            if abs(rect[0] - img.shape[1]/2) > 5 and abs(rect[0]+rect[2] - img.shape[1]/2) > 5:
                good_rects.append(rect)
        rects = good_rects

    # manually adjust bounding boxes 
    if horizontal_adjust:
        for rect in rects:
            rect[0] += horizontal_adjust
    
    yolo_boxes = []

    # draw rectangles 
    for rect in rects:
        x, y, w, h = rect
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        x_center_yolo = (x+(0.5*w))/img.shape[1]
        y_center_yolo = (y+(0.5*h))/img.shape[0]
        w_yolo = w/img.shape[1]
        h_yolo = h/img.shape[0]

        yolo_boxes.append([yolo_label, x_center_yolo, y_center_yolo, w_yolo, h_yolo])

    cv_plot(np.hstack((original_img, img)), "final image")
    
    if not debug:
        label_dir = os.path.join(os.path.dirname(filename),"labels")
        no_label_dir = os.path.join(os.path.dirname(filename), "no_labels")
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        if not os.path.exists(no_label_dir):
            os.makedirs(no_label_dir)
        
        yolo_label_filename = os.path.join(os.path.dirname(filename),"labels",os.path.splitext(os.path.basename(filename))[0]+".txt")
        with open(yolo_label_filename, 'w') as yolo_label_file:
            for box in yolo_boxes:
                row = " ".join(str(b) for b in box)+"\n"
                yolo_label_file.write(row)
        
        if yolo_boxes:
            label_img_filename = os.path.join(os.path.dirname(filename),"labels","label_"+os.path.basename(filename))
            cv.imwrite(label_img_filename, img)
        else: 
            no_label_img_filename = os.path.join(os.path.dirname(filename), "no_labels", os.path.basename(filename))
            cv.imwrite(no_label_img_filename, img)




tbs_crs_args = {
    "invert": True,
    "custom_morphology": [],#[(cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 5))),(cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (1,5)))],
    "kernel_open": cv.getStructuringElement(cv.MORPH_RECT, (4, 50)),#np.ones((50, 4), np.uint8),
    "kernel_close": cv.getStructuringElement(cv.MORPH_RECT, (6, 25)),#np.ones((25, 6), np.uint8)
    "post_thresh_morphology": [(cv.dilate, np.ones((1,4), np.uint8))]
    "bilateral_kernel_size": 2,
    "group_horizontal": False, 
    "area_threshold": 0.7,
    "yolo_label": 0,
    "vertical": True,
    "dc_block": True,
    "horizontal_adjust": -4
}

wifi_args = {
    "invert": True,
    "custom_morphology": [(cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (65, 1)))],
    "kernel_open": cv.getStructuringElement(cv.MORPH_RECT, (65, 1)),#np.ones((3, 3), np.uint8), 
    "kernel_close": cv.getStructuringElement(cv.MORPH_RECT, (65, 1)),#np.ones((15, 15), np.uint8),
    "bilateral_kernel_size": 8,
    "group_horizontal": True, 
    "area_threshold": 0.7,
    "yolo_label": 0,
    "vertical":False, 
    "dc_block":False,
    "horizontal_adjust":0,
}

args_dict = ({
    "wifi": wifi_args,
    "tbs_crossfire": tbs_crs_args,
})

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
        choices=["wifi", "tbs_crossfire"],
        help="Type of signal. Will decide labelling parameters."
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()

    filepath = args.filepath
    label_args = args_dict[args.signal_type]
    global debug
    debug = args.debug

    img_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')

    # get files
    files = []
    if os.path.isfile(filepath):
        files.append(filepath)
    elif os.path.isdir(filepath):
        files.extend(
            [os.path.join(filepath, filename) for filename in os.listdir(filepath) if os.path.isfile(os.path.join(filepath, filename)) and filename.lower().endswith(img_extensions)]
        )
    else:
        raise ValueError("filepath must be existing directory or filename")

    # loop through files
    for img_filename in files:
        label(img_filename, **label_args)


if __name__ == "__main__":
    main()
