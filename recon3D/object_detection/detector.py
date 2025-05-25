"""
Yolo wrapper for the detection of desired objects
"""

from ultralytics import YOLO
import cv2


class Detector:
    """
    Yolo wrapper
    """

    def __init__(
        self, model_path="", segmentation=False, classes=None, conf=0.25, iou=0.45
    ):
        """
        Initialize the Yolo model
        """
        if segmentation:
            self._load_segmentation_model(model_path)
        else:
            self._load_model(model_path)
        self.tracker = "bytetrack.yaml"
        self.model.conf = conf
        self.model.iou = iou
        self.model.classes = classes

    def process_frame(self, image, save=False):
        """
        Process the image and return the detections
        """
        results = self.model.predict(image)
        detections = []
        for res in results:
            for box in res.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                detected_class = int(box.cls[0])
                class_name = res.names[detected_class]

                if self.model.classes is None or class_name in self.model.classes:
                    detections.append(
                        (x1, y1, x2, y2, conf, detected_class, class_name)
                    )
        if save:
            self.model.save()
        return detections

    def process_frame_with_tracking(self, image, tracker="bytetrack.yaml", save=False):
        """
        Process the image with tracking and return the detections
        """
        results = self.model.track(image, persist=True, tracker=tracker)
        tracked_detections = []

        for res in results:
            if res.boxes.id is not None:
                for i, box in enumerate(res.boxes):
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = box.conf[0]
                    detected_class = int(box.cls[0])
                    class_name = res.names[detected_class]
                    track_id = int(box.id[0])

                    if self.model.classes is None or class_name in self.model.classes:
                        tracked_detections.append(
                            (track_id, x1, y1, x2, y2, conf, detected_class, class_name)
                        )

        if save:
            self.save()

        return tracked_detections

    def process_frame_with_tracking_and_segmentation(
        self, image, tracker="bytetrack.yaml", save=False
    ):
        """
        Process the image with tracking and segmentation, returning the detections
        """
        results = self.model.track(image, persist=True, tracker=tracker)
        tracked_detections = []

        for res in results:

            if res.boxes.id is not None:
                for i, box in enumerate(res.boxes):
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = box.conf[0]
                    detected_class = int(box.cls[0])
                    class_name = res.names[detected_class]
                    track_id = int(box.id[0])

                    mask = None
                    if hasattr(res, "masks") and res.masks is not None:
                        mask = res.masks[i].data[0].cpu().numpy()

                    if self.model.classes is None or class_name in self.model.classes:
                        tracked_detections.append(
                            (
                                track_id,
                                x1,
                                y1,
                                x2,
                                y2,
                                conf,
                                detected_class,
                                class_name,
                                mask,
                            )
                        )

        if save:
            self.save()

        return tracked_detections

    def save(self):
        """
        Save the model
        """
        self.model.save()

    def draw_detections(self, image, detections):
        """
        Draw the detections for visualizations
        """
        for x1, y1, x2, y2, conf, detected_class, class_name in detections:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                image,
                f"{class_name}: {conf:.2f}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
        return image

    def _load_model(self, model_path):
        """
        Load a custom model
        """
        if model_path:
            self.model = YOLO(model_path)
        else:
            self.model = YOLO("yolov8n.pt")
        self.model.fuse()

    def _load_segmentation_model(self, model_path):
        """
        Load a segmentation model
        """
        if model_path:
            self.model = YOLO(model_path)
        else:
            self.model = YOLO("yolov8n-seg.pt")
        self.model.fuse()

    def process_video(
        self, video_path, output_path=None, show=False, tracker="bytetrack.yaml"
    ):
        """
        Process a video file with tracking and return detections for each frame
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        all_tracked_detections = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_detections = self.process_frame_with_tracking(frame, tracker=tracker)
            all_tracked_detections.append(frame_detections)

            if output_path or show:
                frame_with_detections = self.draw_tracked_detections(
                    frame.copy(), frame_detections
                )

                if output_path:
                    out.write(frame_with_detections)

                if show:
                    cv2.imshow("Object Tracking", frame_with_detections)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        cap.release()
        if output_path:
            out.release()
        if show:
            cv2.destroyAllWindows()

        return all_tracked_detections

    def draw_tracked_detections(self, image, tracked_detections):
        """
        Draw the tracked detections for visualizations
        """
        for (
            track_id,
            x1,
            y1,
            x2,
            y2,
            conf,
            detected_class,
            class_name,
        ) in tracked_detections:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                image,
                f"{class_name} {track_id}: {conf:.2f}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
        return image

    def save_detections(self, detections, output_path):
        """
        Save the detections to a file
        """
        with open(output_path, "w") as f:
            for frame_id, frame_detections in enumerate(detections):
                for detection in frame_detections:
                    f.write(
                        f"{frame_id},{detection[0]},{detection[1]},{detection[2]},{detection[3]},{detection[4]},{detection[5]},{detection[6]}\n"
                    )

    def get_class_names(self):
        """
        Get the names of classes the model can detect
        """
        return self.model.names if hasattr(self.model, "names") else None

    def set_parameters(self, conf=None, iou=None, classes=None):
        """
        Update detection parameters
        """
        if conf is not None:
            self.model.conf = conf
        if iou is not None:
            self.model.iou = iou
        if classes is not None:
            self.model.classes = classes

    def train(
        self, source_images, source_labels, epochs=10, batch_size=16, img_size=640
    ):
        """
        Train the model
        """
        self.model.train(
            data=source_labels,
            imgsz=img_size,
            epochs=epochs,
            batch_size=batch_size,
            device=0,
            workers=4,
            project="runs/train",
            name="yolo_custom",
            exist_ok=True,
        )

    def export_model(self, format):
        """
        Export the model to a specified format
        """
        self.model.export(format=format)

    def load_model(self, model_path):
        """
        Load a custom model
        """
        self._load_model(model_path)


def process_video_with_tracking(
    self, video_path, output_path=None, show=False, tracker="bytetrack.yaml"
):
    """
    Process a video file with object tracking to maintain consistent IDs across frames
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    all_tracked_detections = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = self.model.track(frame, persist=True, tracker=tracker)

        frame_detections = []
        for res in results:
            if res.boxes.id is not None:
                for i, box in enumerate(res.boxes):
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = box.conf[0]
                    detected_class = int(box.cls[0])
                    class_name = res.names[detected_class]
                    track_id = int(box.id[0])

                    if self.model.classes is None or class_name in self.model.classes:
                        frame_detections.append(
                            (track_id, x1, y1, x2, y2, conf, detected_class, class_name)
                        )

        all_tracked_detections.append(frame_detections)

        if output_path or show:
            frame_with_detections = self.draw_tracked_detections(
                frame.copy(), frame_detections
            )

            if output_path:
                out.write(frame_with_detections)

            if show:
                cv2.imshow("Object Tracking", frame_with_detections)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        frame_idx += 1

    cap.release()
    if output_path:
        out.release()
    if show:
        cv2.destroyAllWindows()

    return all_tracked_detections
