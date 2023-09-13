import cv2
import argparse
from PIL import Image
import numpy as np
from pathlib import Path

from multiprocessing import Pool 

def split_image(img_path, size, out_dir):
    img = np.array(Image.open(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # make grayscale 
    for i in range(0, img.shape[0]-size, size):
        for j in range(0, img.shape[1]-size, size):
            crop = img[i:i+size, j:j+size]
            crop = Image.fromarray(crop)
            crop.save(out_dir / f"{img_path.stem}_{i}_{j}.png")

def main(args) -> None:

    img_dir = Path(args.data_dir)
    out_dir = Path("/home/art/data_tmp/anomaly_detection")
    out_dir = out_dir / f"split_{args.size}" 
    out_dir.mkdir(exist_ok=True, parents=True)
    
    # read all the images and generate 128 x 128 crops of them in a new directory
    pool = Pool(8) 
    for img_path in img_dir.glob("*.png"):
        pool.apply_async(split_image, args=(img_path, args.size, out_dir))

    pool.close()
    pool.join()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/mnt/data/anomaly_detection/drone_flights/train")
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args()

    main(args)