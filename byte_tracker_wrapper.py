import os

from yolov7.tracker.byte_tracker import BYTETracker
from yolov7.utils.plots import plot_one_box


class ByteTrackArgs(object):
    track_thresh: float = 0.5
    track_buffer: int = 20
    match_thresh: float = 0.5
    min_box_area: int = (10,)
    # aspect_ratio_thresh: float = 1.8
    use_bbox_filters: bool = False
    name: str = "yolov7"
    ckpt: str = os.getcwd() + "/app/model/yolov7/yolov7.pth"
    num_classes: int = 80
    det_thresh: float = 0.2
    det_nms_thresholdsh: float = 0.5

    def __init__(self) -> None:
        pass


class ByteTrackWrapper:
    def __init__(self, args: ByteTrackArgs):
        super().__init__()
        self.__args = args
        self.__tracker = BYTETracker(args=args)
        self.frame_id = 0

    def update(self, output_results, img_info, img_size):
        online_targets = self.__tracker.update(
            output_results=output_results, img_info=img_info, img_size=img_size
        )

        outputs = []

        # online_targets attributes :
        # ['_tlwh', 'activate', 'covariance', 'curr_feature',
        # 'end_frame', 'features', 'frame_id', 'history', 'is_activated',
        # 'kalman_filter', 'location', 'mark_lost', 'mark_removed', 'mean',
        # 'multi_predict', 'next_id', 'predict', 're_activate', 'score',
        # 'shared_kalman', 'start_frame', 'state', 'time_since_update',
        # 'tlbr', 'tlbr_to_tlwh', 'tlwh', 'tlwh_to_tlbr', 'tlwh_to_xyah',
        # 'to_xyah', 'track_id', 'tracklet_len', 'update']

        for target in online_targets:
            tlwh = target.tlwh
            tid = target.track_id
            if self.__args.use_bbox_filters:
                vertical = tlwh[2] / tlwh[3] > self.__args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] < self.__args.min_box_area and vertical:
                    continue
            
            outputs.append({'id' : tid, 'tlwh' : tlwh})

        self.frame_id += 1

        return outputs