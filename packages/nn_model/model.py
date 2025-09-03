import os
from typing import Tuple

import numpy as np

# Force CPU-only mode to avoid CUDA library dependencies
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TORCH_USE_CUDA_DSA'] = '0'

try:
    import torch
    # Ensure PyTorch uses CPU only
    torch.cuda.is_available = lambda: False
    torch.backends.cudnn.enabled = False
    TORCH_AVAILABLE = True
except Exception as e:
    print(f"WARNING: PyTorch not available: {e}")
    print("Object detection will be disabled")
    TORCH_AVAILABLE = False
    torch = None

try:
    from dt_data_api import DataClient
    from solution.integration_activity import MODEL_NAME, DT_TOKEN
    from dt_device_utils import DeviceHardwareBrand, get_device_hardware_brand
except Exception as e:
    DataClient = None
    def MODEL_NAME():
        return "model"
    def DT_TOKEN():
        return ""
    class DeviceHardwareBrand:
        JETSON_NANO = "JETSON_NANO"
    def get_device_hardware_brand():
        return None

from .constants import IMAGE_SIZE, ASSETS_DIR

JETSON_FP16 = True


def run(input, exception_on_failure=False):
    print(input)
    try:
        import subprocess

        program_output = subprocess.check_output(
            f"{input}", shell=True, universal_newlines=True, stderr=subprocess.STDOUT
        )
    except Exception as e:
        if exception_on_failure:
            print(e.output)
            raise e
        program_output = e.output
    print(program_output)
    return program_output.strip()


class Wrapper:
    def __init__(self, aido_eval=False):
        if not TORCH_AVAILABLE:
            print("WARNING: PyTorch not available, creating dummy wrapper")
            self.model = None
            return
            
        model_name = MODEL_NAME()

        models_path = os.path.join(ASSETS_DIR, "nn_models")
        dcss_models_path = "courses/mooc/objdet/data/nn_models/"

        dcss_weight_file_path = os.path.join(dcss_models_path, f"{model_name}.pt")
        weight_file_path = os.path.join(models_path, f"{model_name}.pt")

        if aido_eval:
            assert os.path.exists(weight_file_path)
        else:
            dt_token = DT_TOKEN()
            
            if get_device_hardware_brand() == DeviceHardwareBrand.JETSON_NANO:
                # when running on the robot, we store models in the persistent `data` directory
                models_path = "/data/nn_models"
                weight_file_path = os.path.join(models_path, f"{model_name}.pt")

            # make models destination dir if it does not exist
            if not os.path.exists(models_path):
                os.makedirs(models_path)

            # open a pointer to the DCSS storage unit (optional dependency)
            if DataClient is None:
                print("WARNING: dt_data_api not available; cannot auto-download model. Skipping downloads.")
                self.model = None
                return
            client = DataClient(dt_token)
            storage = client.storage("user")

            # make sure the model exists
            metadata = None
            try:
                metadata = storage.head(dcss_weight_file_path)
            except FileNotFoundError:
                print(f"FATAL: Model '{model_name}' not found. It was expected at '{dcss_weight_file_path}'.")
                exit(1)

            # extract current ETag
            remote_etag = eval(metadata["ETag"])
            print(f"Remote ETag for model '{model_name}': {remote_etag}")

            # read local etag
            local_etag = None
            etag_file_path = f"{weight_file_path}.etag"
            if os.path.exists(etag_file_path):
                with open(etag_file_path, "rt") as fin:
                    local_etag = fin.read().strip()
                print(f"Found local ETag for model '{model_name}': {local_etag}")
            else:
                print(f"No local model found with name '{model_name}'")

            # do not download if already up-to-date
            print(f"DEBUG: Comparing [{local_etag}] <> [{remote_etag}]")
            if local_etag != remote_etag:
                if local_etag:
                    print(f"Found a different model on DCSS.")
                print(f"Downloading model '{model_name}' from DCSS...")
                # download model
                download = storage.download(dcss_weight_file_path, weight_file_path, force=True)
                download.join()
                assert os.path.exists(weight_file_path)
                # write ETag to file
                with open(etag_file_path, "wt") as fout:
                    fout.write(remote_etag)
                print(f"Model with ETag '{remote_etag}' downloaded!")
            else:
                print(f"Local model is up-to-date!")

        # load pytorch model
        try:
            self.model = Model(weight_file_path)
            print(f"Model loaded successfully from {weight_file_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.model = None

    def predict(self, image: np.ndarray) -> Tuple[list, list, list]:
        if self.model is None:
            # Return empty detections if model is not available
            return [], [], []
        return self.model.infer(image)


class Model:
    def __init__(self, weight_file_path: str):
        super().__init__()

        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available, cannot initialize model")

        # Check if YOLOv5 directory exists
        yolo_path = "/yolov5"
        if not os.path.exists(yolo_path):
            raise RuntimeError(f"YOLOv5 directory not found at {yolo_path}")
            
        # Check if weight file exists
        if not os.path.exists(weight_file_path):
            raise RuntimeError(f"Model weights not found at {weight_file_path}")

        model = torch.hub.load(yolo_path, "custom", path=weight_file_path, source="local")
        model.eval()

        use_fp16: bool = JETSON_FP16 and get_device_hardware_brand() == DeviceHardwareBrand.JETSON_NANO

        if use_fp16:
            model = model.half()

        # Force CPU usage to avoid CUDA library issues in container
        self.model = model.cpu()

        del model

    def infer(self, image: np.ndarray) -> Tuple[list, list, list]:
        det = self.model(image, size=IMAGE_SIZE)

        xyxy = det.xyxy[0]  # grabs det of first image (aka the only image we sent to the net)

        if xyxy.shape[0] > 0:
            conf = xyxy[:, -2]
            clas = xyxy[:, -1]
            xyxy = xyxy[:, :-2]

            return xyxy.tolist(), clas.tolist(), conf.tolist()
        return [], [], []
