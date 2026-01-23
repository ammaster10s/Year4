import os
import numpy as np
import json
from PIL import Image, ImageDraw
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from pycocotools import mask as maskUtils

def segmentation_to_mask(segmentation, width, height):
    """
    Convert COCO segmentation format (RLE or polygon) to a binary mask.
    """
    if isinstance(segmentation, dict):  # RLE format (compressed RLE)
        rle = maskUtils.frPyObjects(segmentation, height, width)
        mask = maskUtils.decode(rle)
        return mask
    elif isinstance(segmentation, list):  # Polygon format
        mask = np.zeros((height, width), dtype=np.uint8)
        for polygon in segmentation:
            poly = np.array(polygon).reshape((-1, 2))
            img = Image.new('L', (width, height), 0)
            ImageDraw.Draw(img).polygon([tuple(p) for p in poly], outline=1, fill=1)
            mask = np.array(img, dtype=np.uint8)
        return mask
    else:
        return None  # Invalid or missing segmentation


def parse_coco_json(json_file, image_dir):
    """
    Parse COCO JSON file and map images to their segmentation masks.
    """
    with open(json_file, 'r') as f:
        coco_data = json.load(f)

    annotations = []
    image_data = {}

    # Map image ID to file name and dimensions
    for img_info in coco_data['images']:
        image_data[img_info['id']] = {
            'file_name': img_info['file_name'],
            'width': img_info['width'],
            'height': img_info['height']
        }

    # Extract segmentation and mask information
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        image_info = image_data[image_id]
        width, height = image_info['width'], image_info['height']
        mask = segmentation_to_mask(ann.get('segmentation'), width, height)
        annotations.append((image_info['file_name'], mask))

    # Add images with no annotations as empty masks
    for image_id, image_info in image_data.items():
        if not any(ann[0] == image_info['file_name'] for ann in annotations):
            print(f"No annotation for {image_info['file_name']}, adding empty mask.")
            empty_mask = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)
            annotations.append((image_info['file_name'], empty_mask))

    return annotations


def load_data_with_masks(image_dir, annotation_file, target_size=(512, 512), mode=("all")):
    """
    Load image data and their corresponding masks based on the selected mode.
    Modes:
        - "all": Include all images, assigning empty masks to images without annotations.
        - "cnv": Include only images with masks (CNV images).
    """
    annotations = parse_coco_json(annotation_file, image_dir)
    images, masks, image_names = [], [], []

    # Process annotations
    for image_filename, mask in annotations:
        image_path = os.path.join(image_dir, image_filename)
        if not os.path.exists(image_path):
            print(f"File not found: {image_path}, skipping.")
            continue

        # Load and preprocess image
        img = img_to_array(load_img(image_path, color_mode='grayscale', target_size=target_size))
        img = preprocess_noise(img)

        # Resize mask or create an empty mask
        if mask is not None:
            if mode == "normal":
                continue  # Skip CNV images in 'normal' mode
            mask_resized = np.array(Image.fromarray(mask).resize(target_size, Image.NEAREST))
            print(f"{image_filename} ({len(images) + 1}) has mask.")
        else:
            if mode == "cnv":
                continue  # Skip NORMAL images in 'cnv' mode
            mask_resized = np.zeros(target_size, dtype=np.uint8)
            print(f"{image_filename} ({len(images) + 1}) has no mask.")

        images.append(img)
        masks.append(mask_resized)
        image_names.append(image_filename)

    # Add all images without masks in 'normal' mode
    if mode == "all":
        for img_file in os.listdir(image_dir):
            if img_file not in image_names:
                image_path = os.path.join(image_dir, img_file)
                if not os.path.exists(image_path):
                    print(f"File not found: {image_path}, skipping.")
                    continue

                img = img_to_array(load_img(image_path, color_mode='grayscale', target_size=target_size))
                img = preprocess_noise(img)
                images.append(img)
                masks.append(np.zeros(target_size, dtype=np.uint8))  # Empty mask
                print(f"{img_file} ({len(images)}) has no mask.")
                image_names.append(img_file)

    if len(images) == 0:
        print("No valid images and masks were loaded!")

    return np.array(images), np.array(masks), image_names