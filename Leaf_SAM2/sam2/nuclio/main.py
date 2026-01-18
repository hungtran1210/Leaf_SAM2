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
    model1 = YOLO("/opt/nuclio/sam2/yolo1.pt")
    context.user_data.yolo1 = model1
    model2 = YOLO("/opt/nuclio/sam2/yolo2.pt")
    context.user_data.yolo2 = model2
    context.logger.info("Init context...100%")

def check_green(image,min_green = np.array([35,20,20]),max_green = np.array([70,255,255])) :
    # Chuyển ảnh sang HSV
    hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)

    # Tạo mask từ ngưỡng
    mask = cv2.inRange(hsv, min_green, max_green)

    # Xóa nhiễu và làm mịn
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    return mask

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

def check_mask(green, s_mask, min_green_ratio=0.3):
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
    return ratio >= min_green_ratio

def sam2_input(yolo_results,green):
    if len(yolo_results) == 0 or len(yolo_results[0].boxes) == 0:
        return None, None, None

    # Lấy box
    boxes = yolo_results[0].boxes.xyxy.cpu().numpy()

    # Tính tâm các box
    center_x = (boxes[:, 0] + boxes[:, 2]) / 2
    center_y = (boxes[:, 1] + boxes[:, 3]) / 2
    centers = np.stack([center_x, center_y], axis=1)

    # point và label prompt
    points = centers[:, np.newaxis, :]
    # Lấy giá trị pixel tại các điểm tâm
    h, w = green.shape[:2]
    cx = np.clip(center_x.astype(int), 0, w-1)
    cy = np.clip(center_y.astype(int), 0, h-1)

    is_positive = green[cy, cx] > 0
    labels = is_positive.astype(np.float32)[:, np.newaxis]

    return boxes, points, labels

def clean_mask(masks, scores,thresh=0.5):
    scores = scores.flatten()
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    # sắp xếp  theo scores
    sorted_indices = np.argsort(scores)[::-1]
    h, w = masks[0].shape[:2]
    occupied_region = np.zeros((h, w), dtype=bool)

    final_masks = []
    final_indices = []

    # Những vùng trùng sẽ tính theo mask score cao hơn
    for idx in sorted_indices:
        current_mask = masks[idx].astype(bool)
        current_area = np.count_nonzero(current_mask)
        cleaned_mask = current_mask & (~occupied_region)
        cleaned_area = np.count_nonzero(cleaned_mask)
        ratio = cleaned_area / current_area

        if ratio > thresh:
            mask_final = cleaned_mask.astype(np.uint8)
            final_masks.append(mask_final)
            final_indices.append(idx)
            occupied_region = occupied_region | cleaned_mask

    return final_masks, final_indices

def handler(context, event):
    context.logger.info("call handler")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    image = Image.open(buf)
    image = image.convert("RGB")  #  to make sure image comes in RGB

    model1 = context.user_data.yolo1
    model2 = context.user_data.yolo2
    predictor = context.user_data.sam2
    crop_results = model1.predict(image, save=False)

    image = np.array(image)
    h,w = image.shape[:2]
    crops,positions = find_crop(image,crop_results)

    n = len(crops)
    results = []

    for i in range(n) :
        crop = crops[i]
        results2 = model2.predict(crop, save=False)
        green = check_green(crop)
        predictor.set_image(crop)
        boxes, points, labels = sam2_input(results2,green)
        if boxes is None: continue
        masks, scores, _ = predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=boxes,
            multimask_output=False,
        )

        new_masks, index = clean_mask(masks, scores)
        for mask, idx in zip(new_masks, index):
            mask = np.array(mask.cpu() if hasattr(mask, "cpu") else mask)
            mask = np.squeeze(mask).astype(np.uint8)
            if (check_mask(green,mask) == 0) : continue
            mask = mask_full(mask,h,w,positions[i])

            # Tìm contour (polygon)
            contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.1, closed=True) for contour in contours]
            if len(contours) == 0:
                continue
            contour = max(contours, key=cv2.contourArea)
            polygon = contour.flatten().tolist()
            results.append({
                "confidence": str(float(scores[idx])),
                "label": "Spinacia",
                "points": polygon,
                "type": "polygon",
                })
    return context.Response(body=json.dumps(results), headers={},content_type='application/json', status_code=200)