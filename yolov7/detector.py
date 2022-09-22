import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from .models.experimental import attempt_load
from .utils.datasets import LoadStreams, LoadImages
from .utils.general import (
    check_img_size,
    check_requirements,
    check_imshow,
    non_max_suppression,
    apply_classifier,
    scale_coords,
    xyxy2xywh,
    strip_optimizer,
    set_logging,
    increment_path,
)

from .utils.plots import plot_one_box
from .utils.torch_utils import (
    select_device,
    load_classifier,
    time_synchronized,
    TracedModel,
)

class YOLOv7Detector(object):
    def __init__(
        self,
        device,
        weight_dir: str,
        image_size: int,
        confidence_threshold: int,
        iou_threshold: int,
        classes: list,
        agnostic_nms: bool,
    ) -> None:
        self.__image_size = image_size
        self.__conf_thres = confidence_threshold
        self.__iou_thres = iou_threshold
        self.__classes = classes
        self.__agnostic_nms = agnostic_nms

        set_logging()
        device = select_device(device)  # device = 'cpu' or '0' or '0,1,2,3'
        half = device.type != "cpu"  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weight_dir, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        self.__image_size = check_img_size(
            self.__image_size, s=stride
        )  # check img_size

        trace = False
        if trace:
            model = TracedModel(model, device, self.__image_size)

        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name="resnet101", n=2)  # initialize
            modelc.load_state_dict(
                torch.load("weights/resnet101.pt", map_location=device)["model"]
            ).to(device).eval()

        self.__device = device

        # Get names and colors
        names = model.module.names if hasattr(model, "module") else model.names
        self.__model_name_color = [
            [random.randint(0, 255) for _ in range(3)] for _ in names
        ]
        self.__model_names = names

        self.model = model

        if self.__device.type != "cpu":
            self.__warmup(device, image_size, model)  # run ones

    def __warmup(self, device, image_size, model):
        self.model(
            torch.zeros(1, 3, self.__image_size, self.__image_size)
            .to(self.__device)
            .type_as(next(self.model.parameters()))
        )  # run once

    def detect(self, img_np):
        """
        detect objects in an image with YOLOv7 Models.
        """
        # Directories
        # save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize

        # Set Dataloader
        # vid_path, vid_writer = None, None
        # if webcam:
        #     view_img = check_imshow()
        #     cudnn.benchmark = True  # set True to speed up constant image size inference
        #     dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        # else:
        #     dataset = LoadImages(source, img_size=imgsz, stride=stride)

        # Run inference
        old_img_w = old_img_h = self.__image_size
        old_img_b = 1

        device = self.__device
        model = self.model

        t0 = time.time()
        # for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img_np).to(device)
        img = img.half() if device.type != "cpu" else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != "cpu" and (
            old_img_b != img.shape[0]
            or old_img_h != img.shape[2]
            or old_img_w != img.shape[3]
        ):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for _ in range(3):
                model(img)[0]

        # print(model)
        # return
        # Inference
        t1 = time_synchronized()
        pred = model(img)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(
            pred,
            self.__conf_thres,
            self.__iou_thres,
            classes=self.__classes,
            agnostic=self.__agnostic_nms,
        )
        t3 = time_synchronized()

        # Apply Classifier
        # classify = False
        # if classify:
        #     pred = apply_classifier(pred, modelc, img, im0s)

        # print(pred) # [tensor([[267.25546, 165.28973, 424.60416, 299.12402,   0.62394,   1.00000]])]

        # Process detections
        # for i, pred in enumerate(pred):  # detections per image
        #     if webcam:  # batch_size >= 1
        #         p, s, im0, frame = (
        #             path[i],
        #             "%g: " % i,
        #             im0s[i].copy(),
        #             dataset.count,
        #         )
        #     else:
        #         p, s, im0, frame = path, "", im0s, getattr(dataset, "frame", 0)

        #     print(f"p: {p}, s: {s}, frame: {frame}")
        # p = Path(p)  # to Path
        # save_path = str(save_dir / p.name)  # img.jpg
        # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
        gn = torch.tensor(img.shape)[[1, 0, 1, 0]]  # normalization gain whwh'
        
        pred = pred[0]
        if len(pred):
            # Rescale boxes from img_size to im0 size
            # pred[:, :4] = scale_coords(
            #     img.shape[2:], pred[:, :4], im0.shape
            # ).round()

            # Print results
            output_sentence = ""
            for c in pred[:, -1].unique():
                n = (pred[:, -1] == c).sum()  # detections per class
                output_sentence += f"{n} {self.__model_names[int(c)]}{'s' * (n > 1)}, "  # add to string

            print(f"yolov7 detections per class : {output_sentence}")

            view_img = True
            # Write results
            # for *xyxy, conf, cls in reversed(pred):
                # save_txt = False
                # if save_txt:  # Write to file
                #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                #     line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                #     with open(txt_path + '.txt', 'a') as f:
                #         f.write(('%g ' * len(line)).rstrip() % line + '\n')

                # if view_img:  # Add bbox to image
                #     label = f"{self.__model_names[int(cls)]} {conf:.2f}"
                #     plot_one_box(
                #         xyxy,
                #         img_np,
                #         label=label,
                #         color=self.__model_name_color[int(cls)],
                #         line_thickness=1,
                #     )

            # Print time (inference + NMS)
            # print(
            #     f"{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS"
            # )

            # Stream results

            # if view_img:
                # cv2.imshow("yolov7 output", img)
                # cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            # if save_img:
            #     if dataset.mode == 'image':
            #         cv2.imwrite(save_path, im0)
            #         print(f" The image with the result is saved in: {save_path}")
            #     else:  # 'video' or 'stream'
            #         if vid_path != save_path:  # new video
            #             vid_path = save_path
            #             if isinstance(vid_writer, cv2.VideoWriter):
            #                 vid_writer.release()  # release previous video writer
            #             if vid_cap:  # video
            #                 fps = vid_cap.get(cv2.CAP_PROP_FPS)
            #                 w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            #                 h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #             else:  # stream
            #                 fps, w, h = 30, im0.shape[1], im0.shape[0]
            #                 save_path += '.mp4'
            #             vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            #         vid_writer.write(im0)

        # if save_txt or save_img:
        #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #     #print(f"Results saved to {save_dir}{s}")

        # print(f"Done. ({time.time() - t0:.3f}s)")

        return pred