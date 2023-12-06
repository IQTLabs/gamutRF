import os
import re
import shutil
import supervision as sv
#from tqdm.notebook import tqdm
import numpy as np
import cv2 as cv2
import imageio
import scipy.ndimage as nd
from autodistill.detection import CaptionOntology
#from autodistill_grounded_sam import GroundedSAM
from autodistill_yolov8 import YOLOv8

labelled_target="/home/ubuntu/gamutRF/extract-gamutrf-fpv-Immersion_Ghost/faraday_cage_collect/fpv-Immersion_Ghost-faraday_yolo_dataset"


# Train YOLOV8 model

target_model = YOLOv8("yolov8n.pt")
target_model.train(labelled_target + "/data.yaml", epochs=200)

