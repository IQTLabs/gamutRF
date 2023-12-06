# based on pytorch's yolov8n example.

import json
from collections import defaultdict
import os

import torch
from torchvision import transforms
from ultralytics import YOLO

from ts.torch_handler.object_detector import ObjectDetector

IMG_SIZE = 640

try:
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
except ImportError as error:
    XLA_AVAILABLE = False


class Yolov8Handler(ObjectDetector):
    image_processing = transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
        ]
    )

    def initialize(self, context):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Yolov8Handler: using cuda")
        elif XLA_AVAILABLE:
            self.device = xm.xla_device()
        else:
            self.device = torch.device("cpu")
            print("Yolov8Handler: using cpu")

        properties = context.system_properties
        self.manifest = context.manifest
        model_dir = properties.get("model_dir")
        # https://docs.ultralytics.com/modes/predict/#inference-arguments
        with open("model_config.json", "r") as f:
            self.model_config = json.load(f)
        self.model_pt_path = None
        if "serializedFile" in self.manifest["model"]:
            serialized_file = self.manifest["model"]["serializedFile"]
            self.model_pt_path = os.path.join(model_dir, serialized_file)
        self.model = self._load_torchscript_model(self.model_pt_path)
        self.initialized = True

    def _load_torchscript_model(self, model_pt_path):
        """Loads the PyTorch model and returns the NN model object.

        Args:
            model_pt_path (str): denotes the path of the model file.

        Returns:
            (NN Model Object) : Loads the model object.
        """
        # TODO: remove this method if https://github.com/pytorch/text/issues/1793 gets resolved

        model = YOLO(model_pt_path)
        model.to(self.device)
        return model

    def inference(self, data, *args, **kwargs):
        kwargs.update(self.model_config)
        print(f"inference got input tensor {data.shape}")
        return super().inference(data, *args, **kwargs)

    def postprocess(self, res):
        output = []
        for data in res:
            result_dict = defaultdict(list)
            for cls, conf, xywh in zip(
                data.boxes.cls.tolist(), data.boxes.conf, data.boxes.xywh
            ):
                name = data.names[int(cls)]
                result_dict[name].append({"conf": conf.item(), "xywh": xywh.tolist()})
            output.append(result_dict)
        print(f"postprocess returned results for {len(output)} inference(s)")
        return output
