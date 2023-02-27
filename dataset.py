import txt.etree.ElementTree as ET
from tqdm import tqdm
from PIL import Image
import os
import shutil
import random
import glob
import yaml


def convert_iraniancarnumberplate(input_dir="car",
                                  output_dir="Car-Dataset",
                                  target_size=(512, 512)):
    # Create output directories if they don't exist
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

    # Define label mapping
    label_map = {"car": 0}

    # Loop over all txt files in input directory
    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith(".txt"):
            # Parse txt file
            tree = ET.parse(os.path.join(input_dir, filename))
            root = tree.getroot()

            # Extract image dimensions
            width = int(root.find("size/width").text)
            height = int(root.find("size/height").text)

            # Open output file for YOLO annotation
            out_file = open(os.path.join(output_dir, "labels", filename[:-4] + ".txt"), "w")

            # Loop over all objects in txt file
            for obj in root.iter("object"):
                label = obj.find("name").text
                if label not in label_map:
                    continue
                label_id = label_map[label]
                bndbox = obj.find("bndbox")
                xmin = int(bndbox.find("xmin").text)
                ymin = int(bndbox.find("ymin").text)
                xmax = int(bndbox.find("xmax").text)
                ymax = int(bndbox.find("ymax").text)
                # Convert bbox to YOLO format
                x = (xmin + xmax) / 2 / width
                y = (ymin + ymax) / 2 / height
                w = (xmax - xmin) / width
                h = (ymax - ymin) / height
                # Write annotation to output file
                out_file.write(f"{label_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

            out_file.close()

            # Copy image file to output directory
            image_filename = os.path.splitext(filename)[0] + ".jpg"
            img = Image.open(os.path.join(input_dir, image_filename))
            img = img.resize(target_size)
            img.save(os.path.join(output_dir, "images", image_filename))

def convert_iranvehicleplatedataset(image_dir="Vehicle Plates/Vehicle Plates",
                                    label_dir="Vehicle Plates annotations/Vehicle Plates annotations",
                                    output_dir="Car-Dataset",
                                    target_size=(512, 512)):
    # Create output directories if they don't exist
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

    # Define label mapping
    label_map = {"vehicle plate": 0}  # replace with your own label mapping

    # Loop over all txt files in label directory
    for filename in tqdm(os.listdir(label_dir)):
        if filename.endswith(".txt"):
            # Parse txt file
            tree = ET.parse(os.path.join(label_dir, filename))
            root = tree.getroot()

            # Extract image dimensions
            width = int(root.find("size/width").text)
            height = int(root.find("size/height").text)

            # Open output file for YOLO annotation
            out_file = open(os.path.join(output_dir, "labels", "c2_" + filename[:-4] + ".txt"), "w")

            # Loop over all objects in txt file
            for obj in root.iter("object"):
                label = obj.find("name").text
                if label not in label_map:
                    continue
                label_id = label_map[label]
                bndbox = obj.find("bndbox")
                xmin = int(bndbox.find("xmin").text)
                ymin = int(bndbox.find("ymin").text)
                xmax = int(bndbox.find("xmax").text)
                ymax = int(bndbox.find("ymax").text)
                # Convert bbox to YOLO format
                x = (xmin + xmax) / 2 / width
                y = (ymin + ymax) / 2 / height
                w = (xmax - xmin) / width
                h = (ymax - ymin) / height
                # Write annotation to output file
                out_file.write(f"{label_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

            out_file.close()

    for image_name in tqdm(os.listdir(image_dir)):
        # Copy image file to output directory
        image_path = os.path.join(image_dir, image_name)
        img = Image.open(image_path)
        img = img.resize(target_size)
        img.save(os.path.join(output_dir, "images", "c2_" + image_name))


def separate_test_val(images_dir="Car-Dataset/images",
                      txts_dir="Car-Dataset/labels",
                      dst_dir="Car-Dataset/Yolo/",
                      yaml_dir="Car-Dataset/data.yaml",
                      validation_percentage=0.2):
    """
    Seperating Train and validation to their related directories
    Args:
        txts_dir: all txts files directory.
        images_dir: your images directory.
        dst_dir:destination directory to save validations images and txts after seperating
        dst_dir:destination directory to save train images and txts after seperating
    Returns:
        No return
    """
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    assert os.path.exists(images_dir), 'images path not exist'
    assert os.path.exists(txts_dir), 'txts path not exist'
    a = glob.glob(os.path.join(txts_dir, '*.txt'))
    all_txts = random.sample(a, len(a))
    validation_max_index = int(validation_percentage * len(all_txts))
    validation_txts_path = all_txts[:validation_max_index]
    train_txts_path = all_txts[validation_max_index:]

    dst_validation_path_images = os.path.join(dst_dir, 'images/val')
    dst_validation_path_annotations = os.path.join(dst_dir, 'labels/val')

    dst_train_path_images = os.path.join(dst_dir, 'images/train')
    dst_train_path_annotations = os.path.join(dst_dir, 'labels/train')

    if not os.path.exists(dst_validation_path_images):
        os.makedirs(dst_validation_path_images, exist_ok=True)

    if not os.path.exists(dst_validation_path_annotations):
        os.makedirs(dst_validation_path_annotations, exist_ok=True)

    if not os.path.exists(dst_train_path_images):
        os.makedirs(dst_train_path_images, exist_ok=True)

    if not os.path.exists(dst_train_path_annotations):
        os.makedirs(dst_train_path_annotations, exist_ok=True)

    # seprate tests
    for txt_path in tqdm(validation_txts_path):
        txt_basename = os.path.basename(txt_path)
        jpg_basename = txt_basename[:-4] + ".jpg"
        shutil.copy2(txt_path, dst_validation_path_annotations)
        image_path = glob.glob(os.path.join(images_dir, jpg_basename))
        try:
            image_path = str(image_path[0])
        except IndexError:
            print("related image file {} not exist".format(os.path.join(images_dir, jpg_basename)))
        assert os.path.exists(image_path), 'image file for {} not exist \n'.format(txt_path)
        shutil.copy2(image_path, dst_validation_path_images)

    # seprate validations
    for txt_path in tqdm(train_txts_path):
        txt_basename = os.path.basename(txt_path)
        jpg_basename = txt_basename[:-4] + ".jpg"
        shutil.copy2(txt_path, dst_train_path_annotations)
        image_path = glob.glob(os.path.join(images_dir, jpg_basename))
        image_path = str(image_path[0])
        assert os.path.exists(image_path), 'image file for {} not exist \n'.format(txt_path)
        shutil.copy2(image_path, dst_train_path_images)

    
    data = {}
    data['names'] = ['plate']
    data['train'] = "Car-Dataset/Yolo/images/train"
    data['val'] = "Car-Dataset/Yolo/images/val"
    data['nc'] = 1

    with open(yaml_dir, 'w') as f:
        yaml.dump(data, f)


if __name__ == "__main__":
    convert_iranvehicleplatedataset()
    convert_iraniancarnumberplate()
    separate_test_val()