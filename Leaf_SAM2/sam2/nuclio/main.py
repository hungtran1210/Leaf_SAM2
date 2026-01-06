# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT
import json
import base64
from PIL import Image
import io
import numpy as np
import torch
import cv2
from ultralytics import YOLO
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from skimage.feature import peak_local_max

def init_context(context):
    context.logger.info("Init context...  0%")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sam2_checkpoint = "/opt/nuclio/sam2/sam2.1_hiera_tiny.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    quantized_model = torch.quantization.quantize_dynamic(
        sam2_model, {torch.nn.Linear}, dtype=torch.qint8
    )
    predictor = SAM2ImagePredictor(quantized_model)
    context.user_data.sam2 = predictor
    model = YOLO("/opt/nuclio/sam2/yolo.pt")
    context.user_data.yolo = model
    context.logger.info("Init context...100%")

def check_green(image,min_green = np.array([35,50,50]),max_green = np.array([70,255,255])) :
    # Chuyển ảnh sang HSV
    hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)

    # Tạo mask từ ngưỡng
    mask = cv2.inRange(hsv, min_green, max_green)

    # Xóa nhiễu và làm mịn
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    return mask

def find_leaf_points(mask,min_area = 300,max_area = 10000):
    points = []
    # Co ảnh
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    erosion = cv2.erode(mask,kernel,iterations=1)

    # Tìm từng vùng connected component
    num_labels, labels,stats,centroids = cv2.connectedComponentsWithStats(erosion, connectivity=8)
    for i in range(1,num_labels) :
        _,_,_,_,area = stats[i]
        if area < min_area:
            erosion[labels == i] = 0
        if area > min_area and area < max_area :
            points.append(centroids[i])
            erosion[labels == i] = 0

    # Tạo bản đồ khoảng cách và tìm đỉnh
    dist = cv2.distanceTransform(erosion, cv2.DIST_L2, 5)
    coords = peak_local_max(dist, min_distance=30,threshold_abs=10)
    coords = np.array([(int(x), int(y)) for (y, x) in coords])
    for i in coords :
        points.append(i)

    return np.array(points)

def find_crop(image,results) :
    boxes = results[0].boxes.xyxy

    box_results = []
    crops = []
    positions = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        box_results.append(box)

    for i in box_results :
        x1,y1,x2,y2 = i.int().tolist()
        crop = image[y1:y2, x1:x2].copy()
        crops.append(crop)
        positions.append((y1,y2,x1,x2))
    return crops,positions

def mask_full(mask,h,w,position) :
    mask_f = np.zeros((h, w), dtype=np.uint8)
    y1, y2, x1, x2 = position
    mask_f[y1:y2, x1:x2] = mask
    return mask_f

def sam2_inputs(points, min_dist=100):
    if len(points) == 0: return []
    if len(points) == 1:
        return np.array([points[0]]),np.array([1])

    points = np.array(points)
    num_points = len(points)
    input_coords = []
    input_labels = []

    # Lấy điểm gần nhất > min dist làm negative point
    for i in range(num_points):
        point = points[i]

        min_d_point = None
        min_d = float('inf')

        for j in range(num_points):
            if i == j: continue

            dist = np.linalg.norm(point - points[j])

            if dist < min_d:
                min_d = dist
                min_d_point = points[j]

        if min_d_point is not None and min_d > min_dist:
            input_coord = [point,min_d_point]
            input_label = [1,0]
        else :
                input_coord = [point,point]
                input_label = [1,1]

        input_coords.append(np.array(input_coord))
        input_labels.append(np.array(input_label))

    return np.array(input_coords),np.array(input_labels)

def check_mask(green, s_mask, min_green_ratio=0.5):
    s_mask = (s_mask > 0).astype(np.uint8) * 255

    # Vùng xanh của mask
    overlap = cv2.bitwise_and(green, green, mask=s_mask)

    # Diện tích mask và vùng xanh
    area_mask = cv2.countNonZero(s_mask)
    if area_mask == 0 : return False
    green_area = cv2.countNonZero(overlap)

    # Tỉ lệ vùng xanh
    ratio = green_area / area_mask

    # Lọc theo tỉ lệ và diện tích
    return ratio >= min_green_ratio and area_mask < 30000

def change(point) :
    points = np.array([np.array([i]) for i in point])
    return points

def handler(context, event):
    context.logger.info("call handler")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    image = Image.open(buf)
    image = image.convert("RGB")  #  to make sure image comes in RGB

    model = context.user_data.yolo
    predictor = context.user_data.sam2
    crop_results = model.predict(image, save=False, iou=0.3)

    image = np.array(image)
    h,w = image.shape[:2]
    crops,positions = find_crop(image,crop_results)

    n = len(crops)
    results = []

    for i in range(n) :
        green = check_green(crops[i])
        crop = crops[i]
        hc,wc = crop.shape[:2]
        predictor.set_image(crops[i])
        for a in range(3) :
            points = find_leaf_points(green)
            if (len(points) == 0) : break
            input_points,input_labels = sam2_inputs(points)
            masks, scores, _ = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=False,
            )
            mask_0 = np.zeros((hc, wc), dtype=np.uint8)
            for mask, score in zip(masks, scores):
                mask = np.array(mask.cpu() if hasattr(mask, "cpu") else mask)
                mask = np.squeeze(mask).astype(np.uint8)
                if (check_mask(green,mask) == 0) : continue
                m = mask
                mask = mask_full(mask,h,w,positions[i])

                # Tìm contour (polygon)
                contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
                if len(contours) == 0:
                    continue
                contour = max(contours, key=cv2.contourArea)
                polygon = contour.flatten().tolist()
                results.append({
                    "confidence": str(float(score)),
                    "label": "Spinacia",
                    "points": polygon,
                    "type": "polygon",
                    })
                mask_0[m > 0] = 1
            if cv2.countNonZero(mask_0) == 0 : break
            green[mask_0 > 0] = 0


    return context.Response(body=json.dumps(results), headers={},content_type='application/json', status_code=200)