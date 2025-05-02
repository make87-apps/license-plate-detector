import json
from pathlib import Path
from importlib.resources import files

import cv2
import numpy as np
import onnxruntime as ort

import make87
from make87_messages.core.empty_pb2 import Empty
from make87_messages.core.header_pb2 import Header
from make87_messages.detection.box.box_2d_pb2 import Box2DAxisAligned
from make87_messages.detection.box.boxes_2d_pb2 import Boxes2DAxisAligned
from make87_messages.detection.ontology.model_ontology_pb2 import ModelOntology
from make87_messages.geometry.box.box_2d_aligned_pb2 import Box2DAxisAligned as Box2DAxisAlignedGeometry
from make87_messages.image.compressed.image_jpeg_pb2 import ImageJPEG


def nms(boxes, scores, iou_threshold):
    idxs = cv2.dnn.NMSBoxes(
        bboxes=boxes, scores=scores, score_threshold=0.0, nms_threshold=iou_threshold
    )
    if len(idxs) == 0:
        return []
    if isinstance(idxs, np.ndarray):
        return idxs.flatten().tolist()
    else:
        return [int(i[0]) if isinstance(i, (list, tuple)) else int(i) for i in idxs]


def xywh2xyxy(x, y, w, h):
    return x - w / 2, y - h / 2, x + w / 2, y + h / 2


def main():
    make87.initialize()
    # Config
    conf_threshold = make87.get_config_value("CONFIDENCE_THRESHOLD", 0.25, float)
    iou_threshold = make87.get_config_value("IOU_THRESHOLD", 0.45, float)
    max_detections = make87.get_config_value("MAX_DETECTIONS", 1000, int)
    image_size = make87.get_config_value("IMAGE_SIZE", 640, int)

    # Providers / Subscribers / Publishers
    ontology_endpoint = make87.get_provider("MODEL_ONTOLOGY", Empty, ModelOntology)
    jpeg_subscriber = make87.get_subscriber(name="IMAGE_DATA", message_type=ImageJPEG)
    detections_publisher = make87.get_publisher(
        name="DETECTIONS", message_type=Boxes2DAxisAligned
    )
    detections_endpoint = make87.get_provider(
        name="DETECTIONS",
        requester_message_type=ImageJPEG,
        provider_message_type=Boxes2DAxisAligned,
    )

    # Load ONNX model with onnxruntime
    model_path = files("app") / "hf" / "model.onnx"
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Load label config
    model_config = Path(files("app") / "hf" / "config.json")
    with open(model_config) as f:
        config = json.load(f)

    # Provide ontology
    def ontology_callback(message: Empty) -> ModelOntology:
        header = Header()
        header.timestamp.GetCurrentTime()
        entries = [
            ModelOntology.ClassEntry(id=int(cid), label=lbl)
            for cid, lbl in config["id2label"].items()
        ]
        return ModelOntology(header=header, classes=entries)

    ontology_endpoint.provide(ontology_callback)

    # Detection callback using onnxruntime
    def detections_callback(message: ImageJPEG) -> Boxes2DAxisAligned:
        # Decode JPEG → BGR image
        buf = np.frombuffer(message.data, dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)

        # Preprocess: resize, BGR→RGB, normalize, NCHW
        resized = cv2.resize(image, (image_size, image_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(rgb, (2, 0, 1))[None, ...]  # shape (1,3,H,W)

        # Inference
        outputs = session.run([output_name], {input_name: blob})
        preds = outputs[0][0]  # shape (num_preds, 85)

        detections = []
        for row in preds:
            obj_conf = row[4]
            class_scores = row[5:]
            class_id = int(np.argmax(class_scores))
            class_conf = class_scores[class_id]
            confidence = obj_conf * class_conf
            if confidence <= conf_threshold:
                continue
            x, y, w, h = row[:4]
            x1, y1, x2, y2 = xywh2xyxy(x, y, w, h)
            detections.append((x1, y1, x2, y2, confidence, class_id))

        if not detections:
            return Boxes2DAxisAligned(header=Header())

        # Apply NMS
        boxes = [
            [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
            for x1, y1, x2, y2, _, _ in detections
        ]
        scores = [float(conf) for *_, conf, _ in detections]
        keep = nms(boxes, scores, iou_threshold)

        # Build output message
        header = make87.header_from_message(
            Header, message=message, append_entity_path="license_plates"
        )
        out = Boxes2DAxisAligned(
            header=header,
            boxes=[
                Box2DAxisAligned(
                    geometry=Box2DAxisAlignedGeometry(
                        header=header,
                        x=float(detections[i][0]),
                        y=float(detections[i][1]),
                        width=float(detections[i][2] - detections[i][0]),
                        height=float(detections[i][3] - detections[i][1]),
                    ),
                    confidence=float(detections[i][4]),
                    class_id=int(detections[i][5]),
                )
                for i in keep[:max_detections]
            ],
        )
        return out

    # Wire up subscribers and providers
    jpeg_subscriber.subscribe(lambda msg: detections_publisher.publish(detections_callback(msg)))
    detections_endpoint.provide(detections_callback)

    make87.loop()


if __name__ == "__main__":
    main()
