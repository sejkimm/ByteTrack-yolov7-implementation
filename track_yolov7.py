"""
ByteTrack Tracking with YOLOv7 Detection
"""

from skimage import io
import numpy as np
import cv2
from yolov7_wrapper import detector
from byte_tracker_wrapper import ByteTrackWrapper, ByteTrackArgs
from yolov7.utils.plots import plot_one_box


def main():
    args = ByteTrackArgs()
    tracker = ByteTrackWrapper(args=args)

    webcam_stream = cv2.VideoCapture(0)

    while True:
        _, image = webcam_stream.read()
        # image = io.imread("https://farm8.staticflickr.com/7195/6862441991_401883acb2_z.jpg")
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_ncwh = np.expand_dims(image.transpose(2, 0, 1), axis=0)
        output = detector.detect(img_np=image_ncwh)
        output = output.detach().numpy()

        online_targets = tracker.update(
            output_results=output,
            img_info=(image.shape[1], image.shape[2]),
            img_size=(image.shape[1], image.shape[2]),
        )
        print(online_targets)

        for target in online_targets:
            tlwh = target["tlwh"]
            xyxy = [tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]]
            image = plot_one_box(
                xyxy, image, color=(255, 0, 0), label=str(target["id"]), line_thickness=1
            )

        cv2.imshow("test", image)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
