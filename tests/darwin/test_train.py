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

MetadataCatalog.get(train_model_name).keypoint_names = ["A", "B", "C", "D", "E", "F", "G", "H"]
MetadataCatalog.get(train_model_name).keypoint_flip_map = []

from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))

cfg.DATASETS.TRAIN = ("nihonbashi",)
cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.02
cfg.SOLVER.MAX_ITER = (
    300
)  # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
    128
)  # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 3 classes (data, fig, hazelnut)
cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 8
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0000299.pth")
cfg.MODEL.ROI_KEYPOINT_HEAD.NAME = "KRCNNConvDeconvUpPVNetHead"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
cfg.DATASETS.TEST = ("nihonbashi", )
predictor = DefaultPredictor(cfg)

from detectron2.utils.visualizer import ColorMode

for d in random.sample(dataset_dicts, 3):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=nihonbashi_metadata, 
                   scale=0.8, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))


    visualizer = Visualizer(im[:, :, ::-1], 
                             metadata=nihonbashi_metadata, 
                             scale=0.8,               
                             instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    vis = visualizer.draw_dataset_dict(d)

    im_pred = v.get_image()[:, :, ::-1]
    im_gt = vis.get_image()[:, :, ::-1]


    cv2.imshow('res',cv2.hconcat([im_gt, im_pred]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()