import os
import json
import yaml
from time import perf_counter
import warnings

warnings.filterwarnings('ignore')

import numpy as np
from tqdm import tqdm

import cv2
from ultralytics import YOLO
# from mmdet.apis import init_detector, inference_detector

from skimage.measure import find_contours
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from shapely.geometry import Polygon, LineString

import configs.paths as path

# Load constants
with open(path.DATA_DIR / 'classes_coco_vehicle.yaml') as _yaml:
    CLASSES_DICT = yaml.load(_yaml, Loader=yaml.FullLoader)
CLASSES = [c for c in CLASSES_DICT]

with open(path.DATA_DIR / 'time_intervals.json') as f:
    TIME_INTERVALS = json.load(f)

with open(path.DATA_DIR / 'polygons.json') as f:
    STAND_POLYGONS = json.load(f)


class Experiment:
    """
    Class for performing experiments with architectures, hyperparameters, etc.
    Handles configs, directories, predicting and results saving.
    """

    def __init__(self, config_name):

        # Get path to YAML config file and load it to dict
        self.config_path = str(path.CONFIG_DIR / config_name)
        with open(self.config_path, 'r') as _yaml:
            self.config = yaml.load(_yaml, Loader=yaml.FullLoader)

        # Load model
        self.model_name = self.config['model_name']
        self.model = self.load_model()

        self.experiment_folder = None

    @staticmethod
    def create_experiment_folder(config_name):
        """Create a folder to store experiment config, result files and annotated frames."""

        config_name = config_name.split('.')[0]
        experiment_folder = path.EXPERIMENTS_DIR / config_name
        experiment_folder.mkdir(parents=True, exist_ok=True)
        return experiment_folder

    def load_model(self):
        """Load model with respect to the source of the pretrained weights."""

        model_path = path.MODELS_DIR / self.model_name
        print('Loading model')
        for key, value in self.config.items():
            print(f'{key}: {value}')
        if 'yolo' in self.model_name:
            return YOLO(model_path)
        # else:
        #     return init_detector(
        #         str(path.MODELS_DIR / self.config['config_file']),
        #         str(path.MODELS_DIR / self.config['checkpoint_file']),
        #         device='cuda:0'
        #     )

    @staticmethod
    def bbox_xywh(xywh):
        x_left, y_top, width, height = xywh.astype(int)
        result = [
            [x_left, y_top],
            [x_left + width, y_top],
            [x_left + width, y_top + height],
            [x_left, y_top + height],
        ]
        return np.array(result)

    @staticmethod
    def bbox_xyxy(xyxy):
        x_left, y_top, x_right, y_bottom = xyxy.astype(int)
        result = [
            [x_left, y_top],
            [x_right, y_top],
            [x_right, y_bottom],
            [x_left, y_bottom],
        ]
        return np.array(result)

    def object_intersects_stand(self, results, stand_polygon):
        """Calculate object markers and their intersections with stand polygon"""

        object_markers = []
        new_object_markers = []

        if self.config['object_marker'] == 'bbox':
            object_markers = results['bboxes']
            new_object_markers = [Polygon(o) for o in object_markers]

        elif self.config['object_marker'] == 'bottom_line':
            object_markers = [bbox[2:] for bbox in results['bboxes']]
            new_object_markers = [LineString(o) for o in object_markers]

        elif self.config['object_marker'] == 'segmentation' and results['contours']:
            object_markers = [self.truncate_contour(mask) for mask in results['contours']]
            for o in object_markers:
                if 1 < len(o) < 4:
                    new_object_markers.append(LineString(o))
                elif len(o) < 2:
                    continue
                else:
                    new_object_markers.append(Polygon(o))

        stand_polygon = Polygon(stand_polygon)
        try:
            intersects = [object_marker.intersects(stand_polygon) for object_marker in new_object_markers]
        except:
            print('Incorrect object marker')
            intersects = []

        return any(intersects), object_markers

    @staticmethod
    def mask2contour(mask):
        """Gets the contour of the binary segmentation mask."""

        return find_contours(mask.T, 0.5)[0].astype(int)

    def truncate_contour(self, mask):
        """
        Takes the boundary of the objects' segmentation mask and cuts off everything except the bottom part.
        The remaining bottom fraction is set by the `mask_trunc_part` parameter of the config file.
        """

        mask = mask.astype(int)
        y_max = np.max(mask[:, 1])
        diff = int((y_max - np.min(mask[:, 1])) * self.config['mask_trunc_part'])
        threshold = y_max - diff
        return mask[mask[:, 1] >= threshold].astype(int)

    def predict_results(self, frame):
        """
        The prediction results from ultralytics and openmmlab models have slightly different format.
        Bring them to a unified form for further processing.
        """

        result = {'contours': []}

        if self.config['platform'] == 'ultralytics':
            # Filter vehicle classes
            predicts = self.model(frame, verbose=False, classes=CLASSES)

            # Filter objects by confidence threshold
            boxes = predicts[0].boxes
            result['confidence'] = np.array(boxes.conf.to('cpu'))
            probs_mask = result['confidence'] > self.config['detection_threshold']
            result['confidence'] = result['confidence'][probs_mask]
            result['classes'] = np.array(boxes.cls.to('cpu'))[probs_mask]
            result['bboxes'] = np.array(boxes.xyxy.to('cpu'))[probs_mask]
            result['bboxes'] = [self.bbox_xyxy(xyxy) for xyxy in result['bboxes']]
            if self.config['object_marker'] == 'segmentation' and predicts[0].masks:
                result['contours'] = [mask for i, mask in enumerate(predicts[0].masks.xy) if probs_mask[i]]

        # elif self.config['platform'] == 'openmmlab':
        #     predicts = inference_detector(self.model, frame).pred_instances
        #
        #     # Filter objects by confidence threshold and vehicle classes
        #     classes_mask = np.isin(predicts.labels.to('cpu'), CLASSES)
        #     result['confidence'] = np.array(predicts.scores.to('cpu'))
        #     probs_mask = result['confidence'] > self.config['detection_threshold']
        #     result['confidence'] = result['confidence'][np.logical_and(classes_mask, probs_mask)]
        #     result['classes'] = np.array(predicts.labels.to('cpu'))[np.logical_and(classes_mask, probs_mask)]
        #     result['bboxes'] = np.array(predicts.bboxes.to('cpu'))[np.logical_and(classes_mask, probs_mask)]
        #     result['bboxes'] = [self.bbox_xyxy(xyxy) for xyxy in result['bboxes']]
        #     if self.config['object_marker'] == 'segmentation' and predicts.masks:
        #         segment_masks = np.array(predicts.masks.to('cpu'))[np.logical_and(classes_mask, probs_mask)]
        #         result['contours'] = [find_contours(mask.T, 0.5)[0].astype(int) for mask in segment_masks]

        return result

    def process_video(self, video_path, stand_polygon):
        """Make predictions for a single video"""

        # Get video specific data
        video_file = video_path.split('/')[-1]
        video_name = video_file.split('.')[0]
        if self.config['save_frames']:
            frames_folder = self.experiment_folder / f'frames/{video_name}'
            frames_folder.mkdir(parents=True, exist_ok=True)

        # Loop through the video frames
        frame_num = 0
        inference_time = 0
        experiment_result = list()
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():

            # Read a frame from the video
            success, frame = cap.read()
            if not success:
                break

            # Make prediction for a single frame
            t1_start = perf_counter()
            results = self.predict_results(frame)

            # Find out if the stand is occupied with a vehicle
            stand_occupied, object_markers = self.object_intersects_stand(results, stand_polygon)
            t1_stop = perf_counter()

            # Measure single frame processing time
            inference_time += t1_stop - t1_start

            experiment_result.append(stand_occupied)

            # Save annotated frames
            if self.config['save_frames']:

                # Draw video name and frame number ease the analysis
                cv2.putText(frame, f'{video_name}, frame {frame_num}', (10, 50), 0, 1, (255, 255, 255), 2,
                            cv2.LINE_AA)

                # Draw stand boundaries
                cv2.polylines(frame, [stand_polygon], isClosed=True, color=(0, 0, 255), thickness=1)
                for i, marker in enumerate(object_markers):
                    # Draw object contours
                    cv2.polylines(frame, [marker], isClosed=True, color=(0, 255, 0),
                                  thickness=1)

                    # Draw object classes and models prediction confidence
                    prob = results['confidence'][i]
                    object_name = CLASSES_DICT[results['classes'][i]]
                    text = f'{prob:.2f} {object_name}'
                    object_coords = results['bboxes'][i][0]
                    cv2.putText(frame, text, object_coords, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

                img_path = str(frames_folder / f'{video_name}_frame_{frame_num}.jpg')
                cv2.imwrite(img_path, frame)

            frame_num += 1

        # Release the video capture object
        cap.release()

        average_frame_time = inference_time / frame_num
        # average_frame_time = None

        return experiment_result, average_frame_time

    @staticmethod
    def intervals2mask(intervals, total_frames):
        """
        Converts the list of the frame intervals to the binary mask,
        where True is the frame, when the stand is occupied with a vehicle and False -- where it is not
        """

        frames_stand_occupied = []
        for interval in intervals:
            frames_stand_occupied += list(range(*(interval + np.array([0, 1]))))

        result = np.zeros(total_frames, dtype=bool)
        result[frames_stand_occupied] = True

        return list(result)

    @staticmethod
    def mask2intervals(mask: np.array) -> list:
        """Converts binary mask of the frames to the list of intervals (reverse to `intervals2mask`)"""

        intervals = []
        total_items = len(mask)

        for i, (frame0, frame1) in enumerate(zip(mask[:-1], mask[1:])):

            # Interval start
            if i == 0 and frame0:
                intervals.append([i])
            if frame1 and not frame0:
                intervals.append([i + 1])

            # Interval stop
            if frame0 and not frame1:
                intervals[-1].append(i)
            if i == total_items - 2 and frame1:
                intervals[-1].append(i + 2)

        return intervals

    def calculate_metrics(self, labels, preds):
        labels, preds = np.array(labels), np.array(preds)
        metrics = ', '.join([
            f'accuracy: {round(accuracy_score(labels, preds), 2)}',
            f'precision: {round(precision_score(labels, preds, zero_division=1), 2)}',
            f'recall: {round(recall_score(labels, preds, zero_division=1), 2)}',
            f'f1: {round(f1_score(labels, preds, zero_division=1), 2)}',
            f'jaccard: {round(self.jaccard(labels, preds), 2)}',
        ])
        return metrics

    @staticmethod
    def jaccard(labels, preds):
        """
        Jaccard index measures the overlap between predicted and true positive frames.
        The size of the intersection divided by the size of the union.
        """
        intersection = np.logical_and(labels, preds)
        union = np.logical_or(labels, preds)

        return intersection.sum() / float(union.sum())

    def save_results(self, result):
        with open(self.experiment_folder / 'result.json', 'w') as f:
            json.dump(result, f)

    def run(self, video_files):
        """Perform a single experiment on the test set and save the results in the corresponding folder"""

        # Create experiment folder and create there a copy of the config file
        self.experiment_folder = self.create_experiment_folder(config_name)
        os.system(f'cp {self.config_path} {self.experiment_folder / config_name}')

        result = {'total': None}
        time_intervals = dict()
        all_predicts, all_labels, avg_frame_times = [], [], []
        for video in tqdm(video_files):
            # Single video predictions
            video_path = str(path.VIDEO_DIR / video)
            video_file = video_path.split('/')[-1]
            stand_polygon = np.array(STAND_POLYGONS[video_file])
            frame_predicts, avg_frame_time = self.process_video(video_path, stand_polygon)
            avg_frame_times.append(avg_frame_time)
            all_predicts += frame_predicts
            time_intervals[video] = self.mask2intervals(frame_predicts)

            # Time intervals to boolean numpy array
            frame_labels = self.intervals2mask(TIME_INTERVALS[video], len(frame_predicts))
            all_labels += frame_labels

            # Calculate metrics
            result[video] = {}
            result[video]['metrics'] = self.calculate_metrics(frame_labels, frame_predicts)

            # Write frame predictions ot results
            labels_preds = list(np.vstack((frame_labels, frame_predicts), dtype=str).T)
            result[video]['frames'] = {frame: ' '.join(label_pred) for frame, label_pred in enumerate(labels_preds)}

        # Save time intervals to json
        with open(self.experiment_folder / 'time_intervals.json', 'w') as f:
            json.dump(time_intervals, f)

        # Write total metrics and average frame processing time
        frame_time = sum(avg_frame_times) / len(avg_frame_times)
        result['total'] = self.calculate_metrics(all_labels, all_predicts) + f' speed: {frame_time:.2f}'

        # Save experiment results
        if self.config['save_results']:
            self.save_results(result)

    def demo(self, video_path, polygon_path, output_path):
        """Script for testing the solution."""

        # Load boundaries for the video
        with open(polygon_path) as f:
            stand_polygon = np.array(json.load(f))

        # Predict object boundaries
        print(f'Processing video {video_path}')
        frame_predicts, _ = self.process_video(video_path, stand_polygon)

        # Save time intervals
        time_intervals = self.mask2intervals(frame_predicts)
        with open(output_path, 'w') as f:
            json.dump(time_intervals, f)
        print(f'Output saved to {output_path}')


if __name__ == '__main__':
    # config_name = 'cascade_mask_rcnn_trunc03_thresh_07.yaml'
    config_name = 'yolov8x-seg-trunc02_test.yaml'
    # config_name = 'yolov8x_bottom_line.yaml'
    # config_name = 'cascade_mask_rcnn_trunc03_thresh_07_bottom_line.yaml'
    # config_name = 'yolov8x_bbox.yaml'
    # config_name = 'cascade_mask_rcnn_trunc03_thresh_07_bbox.yaml'
    experiment = Experiment(config_name)
    # video_files = TIME_INTERVALS.keys()
    video_files = ['video_4.mp4']
    experiment.run(video_files)
