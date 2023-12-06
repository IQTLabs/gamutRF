import os
import re
import shutil
import supervision as sv
import numpy as np
import cv2 as cv2
import imageio
import scipy.ndimage as nd
from autodistill.detection import CaptionOntology
from autodistill_grounded_sam import GroundedSAM
from autodistill_yolov8 import YOLOv8
from ultralytics import YOLO
from pathlib import Path

#target = "/home/ubuntu/gamutRF/extract-gamutrf-fpv-Immersion_Ghost/faraday_cage_collect/166Hz/png"
target = "/home/ubuntu/gamutRF/extract-gamutrf-fpv-Immersion_Ghost/test_image"
#labelled_target = "/home/ubuntu/gamutRF/extract-gamutrf-fpv-Immersion_Ghost/faraday_cage_collect/166Hz/sam-bb-png"
labelled_target = "/home/ubuntu/gamutRF/extract-gamutrf-fpv-Immersion_Ghost/test_dataset"
# Run auto labelling using SAM
ontology=CaptionOntology({
    "signal": " fhss-css",
})

base_model = GroundedSAM(ontology=ontology)


for folder in (os.listdir(target)):
        dataset = base_model.label(

            input_folder=target,
            extension=".png"
            
            )
