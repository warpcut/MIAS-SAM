import cv2
import numpy as np

from segment_anything import SamPredictor, sam_model_registry



def get_sam_model(model_type, device, med=False):
    if med:
        print('Loading medsam weights..')
        sam = sam_model_registry[model_type](checkpoint="./checkpoints/medsam_vit_b.pth").to(device=device)
    else:
        print('Loading sam weights..')
        sam = sam_model_registry[model_type](checkpoint="./checkpoints/sam_vit_b_01ec64.pth").to(device=device)
    predictor = SamPredictor(sam)
    return predictor

def get_prompt(anomaly_map):
    #Center of gravity
    gamma = 5
    weighted_image = np.power(anomaly_map, gamma)
    h, w = anomaly_map.shape
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
    total_weight = np.sum(weighted_image)
    if total_weight == 0 or np.isnan(total_weight):
        raise ValueError("Division / 0.")

    x_center = np.sum(x_coords * weighted_image) / total_weight
    y_center = np.sum(y_coords * weighted_image) / total_weight
    point = (x_center, y_center)
    
    #Threshold
    _, thresholded_image = cv2.threshold(anomaly_map, 0.5, 255, cv2.THRESH_BINARY)
    thresholded_image = thresholded_image.astype(np.uint8)
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, x + w, y + h))  # (x1, y1, x2, y2)
    anomaly_map = cv2.resize(anomaly_map, (256, 256), interpolation=cv2.INTER_AREA)
    anomaly_map = np.expand_dims(anomaly_map, axis=2)
    anomaly_map = np.moveaxis(anomaly_map, -1, 0)

    return np.asarray(bounding_boxes), point, anomaly_map, thresholded_image