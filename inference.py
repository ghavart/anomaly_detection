import time
import cv2
import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms

from autoencoder_model import Autoencoder

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# define a private function to batch process the patches
def __batch_process(model, patches, boxes, err_map, count_map):
    
    patches = torch.stack(patches, axis=0)
    patches = patches.to(device=DEVICE)
    rect_patches = model(patches)
    err_patches = torch.linalg.norm(rect_patches - patches, ord=2, axis=1).detach().cpu().numpy()

    for k, box in enumerate(boxes):
        x0, y0, x1, y1 = box
        
        # increment the normalization map 
        count_map[y0:y1, x0:x1] += np.ones_like(err_patches[k], dtype=np.int32)
        
        # online average pixelwise error 
        err_map[y0:y1, x0:x1] += (err_patches[k] - err_map[y0:y1, x0:x1]) / count_map[y0:y1, x0:x1]
    
    del patches, rect_patches, err_patches
    torch.cuda.empty_cache()
    return err_map, count_map


def process_image(model, orig_img, args) -> np.ndarray:

    size = args.size
    stride = args.stride
    err_threshold = args.err_threshold
    batch_size = 2048

    start = time.time()
    # pad the image so that it is divisible by the stride
    h, w = orig_img.shape[:2]
    pad_h = stride - (h - size) % stride 
    pad_w = stride - (w - size) % stride 
    img = np.pad(orig_img, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

    err_map = np.zeros_like(img, dtype=np.float32)
    count_map = np.zeros_like(img, dtype=np.int32)
    patches = []
    boxes = []
    
    # split the image into patches
    for i in range((img.shape[0] - size) // stride):
        for j in range((img.shape[1] - size) // stride):
            
            # run the patch through the model
            x0 = j * stride
            y0 = i * stride
            x1 = x0 + size 
            y1 = y0 + size
            patch = img[y0:y1, x0:x1]
            patches.append(TRANSFORMS(patch).float())
            boxes.append((x0, y0, x1, y1))

            # batch process the patches 
            if len(patches) >= batch_size:
                err_map, count_map = __batch_process(model, patches, boxes, err_map, count_map)
                patches = []
                boxes = []
            
    # process the remaining patches 
    err_map, count_map = __batch_process(model, patches, boxes, err_map, count_map)

    # crop the error map to the original image size
    err_map = err_map[:h, :w]

    # threshold the error map
    err_map = err_map / np.max(err_map)
    err_map[err_map < err_threshold] = 0
    err_map[err_map >= err_threshold] = 1

    end = time.time()
    print(f"Processing elapsed time: {end - start:.2f} seconds / frame")

    # plot the results
    err_map *= 255.0
    err_map = err_map.astype(np.uint8)

    fig, ax = plt.subplots(2, figsize=(16, 9))
    ax[0].imshow(orig_img, cmap='gray') 
    ax[1].imshow(err_map)
    plt.show()

    return err_map 


def main(args):

    # load the model
    model = Autoencoder(bottlencek_dim=64)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device=DEVICE)

    img_dir = Path(args.img_dir)
    for img_path in img_dir.glob("*.png"):
        img = np.array(Image.open(img_path))
        gr_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # to grayscale

        err_map = process_image(model, gr_img, args)
        
        # save the input and the result
        Image.fromarray(img).save("orig.png")
        Image.fromarray(err_map).save("err.png")
        break # NOMERGE


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, default="/mnt/data/anomaly_detection/drone_flights/test")
    parser.add_argument("--model_path", type=str, default="/home/art/code/anomaly_detection/checkpoints/model_10.pth") 
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--err_threshold", type=float, default=0.8)
    args = parser.parse_args()

    main(args)