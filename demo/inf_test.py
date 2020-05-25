from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
import numpy as np
import cv2
import os, glob, time, sys
import argparse
from tqdm import tqdm, trange

config_file = "../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"

# parse arguments
parser = argparse.ArgumentParser("Perform inference with MaskRCNN")
parser.add_argument('--imgpath', type=str, default=None, help="Path of the image file")
parser.add_argument('--img_dir', type=str, default=None, help='Image folder to perform inference on.')
parser.add_argument('--nimg', type=int, default=-1, help="Number of images to process")
parser.add_argument('--out_dir', type=str, help='Directory to save results into.')

args = parser.parse_args()

# IO options
# python inf_test.py --imgpath /dockerShared/hdd/Dataset_orig/validation/images/2850.jpg --out_dir /dockerShared/hdd/results/demo/
# python inf_test.py --img_dir /dockerShared/hdd/Dataset_orig/validation/images/ --out_dir /dockerShared/hdd/results/demo/ --nimg 50

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)

if args.imgpath is not None:
    img_l = [args.imgpath]
    out_dir = args.out_dir
elif args.img_dir is not None:
    img_l = glob.glob(os.path.join(args.img_dir, '*.jpg'))
    out_dir = os.path.join(args.out_dir, os.path.basename(args.img_dir))
    os.makedirs(args.out_dir, exist_ok=True)
else:
    print("ERROR: No input images to perform inference on!")
    sys.exit(1)

if args.nimg > 0:
    img_l = img_l[:min(args.nimg, len(img_l))]
os.makedirs(out_dir, exist_ok=True)

for imgpath in tqdm(img_l):
    # load image and then run prediction
    img = cv2.imread(imgpath)
    predictions = coco_demo.run_on_opencv_image(img)
    # save result as image file
    respath = os.path.join(out_dir, os.path.basename(imgpath))
    cv2.imwrite(respath, predictions)
print("DONE!")