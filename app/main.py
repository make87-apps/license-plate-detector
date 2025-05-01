from importlib.resources import files
from pathlib import Path

import cv2
import make87
import numpy as np
import yolov5
from make87_messages.core.empty_pb2 import Empty
from make87_messages.core.header_pb2 import Header
from make87_messages.detection.box.box_2d_pb2 import Box2DAxisAligned
from make87_messages.detection.box.boxes_2d_pb2 import Boxes2DAxisAligned
from make87_messages.detection.ontology.model_ontology_pb2 import ModelOntology
from make87_messages.geometry.box.box_2d_aligned_pb2 import Box2DAxisAligned as Box2DAxisAlignedGeometry
from make87_messages.image.compressed.image_jpeg_pb2 import ImageJPEG


def main():
    make87.initialize()
    conf_threshold = make87.get_config_value("CONFIDENCE_THRESHOLD", 0.25, float)
    iou_threshold = make87.get_config_value("IOU_THRESHOLD", 0.45, float)
    class_agnostic = make87.get_config_value(
        "CLASS_AGNOSTIC_NMS", False, lambda s: {"true": True, "false": False}[s.strip().lower()]
    )
    multi_label = make87.get_config_value("MULTI_LABEL", False, bool)
    max_detections = make87.get_config_value("MAX_DETECTIONS", 1000, int)
    image_size = make87.get_config_value("IMAGE_SIZE", 640, int)
    test_time_augmentation = make87.get_config_value(
        "TEST_TIME_AUGMENTATION", False, lambda s: {"true": True, "false": False}[s.strip().lower()]
    )

    ontology_endpoint = make87.get_provider(
        name="MODEL_ONTOLOGY", requester_message_type=Empty, provider_message_type=ModelOntology
    )
    jpeg_subscriber = make87.get_subscriber(name="IMAGE_DATA", message_type=ImageJPEG)
    detections_publisher = make87.get_publisher(name="DETECTIONS", message_type=Boxes2DAxisAligned)
    detections_endpoint = make87.get_provider(
        name="DETECTIONS", requester_message_type=ImageJPEG, provider_message_type=Boxes2DAxisAligned
    )

    # Access the 'preprocessor_config.json' file within 'app.hf' package
    model_path = files("app") / "hf" / "best.pt"
    model_path = Path(str(model_path))

    model = yolov5.load(model_path=str(model_path), device="cpu")
    model.conf = conf_threshold
    model.iou = iou_threshold
    model.agnostic = class_agnostic
    model.multi_label = multi_label
    model.max_det = max_detections

    # Setup ontology provider
    def ontology_callback(message: Empty) -> ModelOntology:
        header = Header()
        header.timestamp.GetCurrentTime()
        class_entries = [
            ModelOntology.ClassEntry(
                id=int(class_id),
                label=class_label,
            )
            for class_id, class_label in model.names.items()
        ]
        return ModelOntology(header=header, classes=class_entries)

    ontology_endpoint.provide(ontology_callback)

    # Setup pub/sub + provider
    def detections_callback(message: ImageJPEG) -> Boxes2DAxisAligned:
        # Convert message data to an image
        jpeg_array = np.frombuffer(message.data, dtype=np.uint8)
        image = cv2.imdecode(jpeg_array, cv2.IMREAD_UNCHANGED)

        # Run detection on the image
        detections = model(image, size=image_size, augment=test_time_augmentation)

        header = make87.header_from_message(
            Header,
            message=message,
            append_entity_path="license_plates",
        )

        # Create the detection message
        boxes2d = Boxes2DAxisAligned(
            header=header,
            boxes=[
                Box2DAxisAligned(
                    geometry=Box2DAxisAlignedGeometry(
                        header=header,
                        x=x1,
                        y=y1,
                        width=x2 - x1,
                        height=y2 - y1,
                    ),
                    confidence=confidence,
                    class_id=class_id,
                )
                for x1, y1, x2, y2, confidence, class_id in detections.pred[0].cpu().numpy()
            ],
        )
        return boxes2d

    jpeg_subscriber.subscribe(lambda msg: detections_publisher.publish(detections_callback(msg)))
    detections_endpoint.provide(detections_callback)

    make87.loop()


if __name__ == "__main__":
    main()
