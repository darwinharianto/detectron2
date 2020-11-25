from detectron2.data.datasets import register_coco_instances
import os, random, cv2

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

train_data_folder = '/home/nizar/Desktop/workspace/train_data'
train_model_name = 'nihonbashi'

register_coco_instances(train_model_name, {}, os.path.join(train_data_folder, 'train_det.json'), train_data_folder)

nihonbashi_metadata = MetadataCatalog.get(train_model_name)
dataset_dicts = DatasetCatalog.get(train_model_name)


from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))

cfg.DATASETS.TRAIN = (train_model_name,)
cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "detectron2://COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x/137849621/model_final_a6e10b.pkl"  # initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.02
cfg.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT = 0.01
cfg.SOLVER.CHECKPOINT_PERIOD = 100
cfg.INPUT.RANDOM_FLIP = "none"
cfg.MODEL.ROI_KEYPOINT_HEAD.NAME = "KRCNNConvDeconvUpPVNetHead"
cfg.SOLVER.MAX_ITER = (
    3000
)  # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
    128
)  # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 3 classes (data, fig, hazelnut)
cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 8

MetadataCatalog.get(train_model_name).keypoint_names = ["A", "B", "C", "D", "E", "F", "G", "H"]
MetadataCatalog.get(train_model_name).keypoint_flip_map = []


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=True)
trainer.train()