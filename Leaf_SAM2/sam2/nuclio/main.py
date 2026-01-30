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
import math

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

def sam2_input(boxes,green):
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

    return points, labels

def clean_mask(masks, scores,greens, thresh=0.5):
    scores = scores.flatten()

    index_t = [i for i, mask in enumerate(masks) if check_mask(greens, mask)]

    masks = [masks[i] for i in index_t]
    if not masks or len(masks) == 0:
        return [], []
    scores = scores[index_t]
    # sắp xếp  theo scores
    sorted_indices = np.argsort(scores)[::-1]
    h, w = masks[0].shape[:2]
    occupied_region = np.zeros((h, w), dtype=bool)

    final_masks = []
    final_scores = []

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
            final_scores.append(scores[idx])
            occupied_region = occupied_region | cleaned_mask

    return final_masks, final_scores

def new_image(image_list, yolo_results_list):
    num_imgs = len(image_list)
    if num_imgs == 0: return None, [], []

    # Thiết lập ảnh lớn
    row = int(math.ceil(math.sqrt(num_imgs)))

    image_row = []
    for i in range(0, num_imgs, row):
        image_row.append(range(i, min(i + row, num_imgs)))

    row_hw = []
    max_w = 0
    total_h = 0
    offsets = {}

    for images in image_row:
        current_imgs = [image_list[i] for i in images]
        row_w = sum(img.shape[1] for img in current_imgs)
        row_h = max(img.shape[0] for img in current_imgs)
        row_hw.append((row_w, row_h))
        if row_w > max_w: max_w = row_w
        total_h += row_h

    new_img = np.zeros((total_h, max_w, 3), dtype=np.uint8)

    # Dán ảnh
    current_y = 0
    for r, images in enumerate(image_row):
        _, row_h = row_hw[r]
        current_x = 0
        for i in images:
            img = image_list[i]
            h, w = img.shape[:2]
            new_img[current_y : current_y + h, current_x : current_x + w] = img
            offsets[i] = (current_x, current_y, current_x + w,current_y + h)
            current_x += w
        current_y += row_h

    # chuyển box,lưu vị trí crop
    data = []
    global_boxes = []

    for i in range(num_imgs):
        x1, y1, x2, y2 = offsets[i]
        data.append({
            "index": i,
            "box": (x1, y1, x2, y2)
        })

        current_result = yolo_results_list[i]
        current_result = current_result[0]
        if current_result.boxes is None or len(current_result.boxes) == 0:
            continue

        local_boxes = current_result.boxes.xyxy.cpu().numpy()
        if len(local_boxes) > 0:
            g_boxes = local_boxes.copy()
            g_boxes[:, 0] += x1
            g_boxes[:, 2] += x1
            g_boxes[:, 1] += y1
            g_boxes[:, 3] += y1
            global_boxes.extend(g_boxes)

    return new_img, data, np.array(global_boxes)

def process_mask(mask,scores,num_box_crop,data,h,w,positions,greens) :
    mask = mask.squeeze(1)
    n = len(data)
    current = 0
    sum = 0
    masks_all = []
    scores_all = []
    for i in range(n):
        sum += num_box_crop[i]
        x1,y1,x2,y2 = data[i]['box']
        mask_local = [mask[j][y1:y2,x1:x2] for j in range(current,sum)]
        new_masks, new_scores = clean_mask(mask_local, scores[current:sum],greens[i])
        for m,s in zip(new_masks,new_scores) :
            mask_cf = mask_full(m,h,w,positions[i])
            masks_all.append(mask_cf)
            scores_all.append(s)
        current += num_box_crop[i]
    return masks_all, scores_all

def to_cvat_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    contour = max(contours, key=cv2.contourArea)
    xtl, ytl, w, h = cv2.boundingRect(contour)
    xbr = xtl + w - 1
    ybr = ytl + h - 1

    polygon = contour.flatten().tolist()

    roi = mask[ytl : ybr + 1, xtl : xbr + 1]
    flattened = roi.flat[:].tolist()
    flattened.extend([xtl, ytl, xbr, ybr])
    
    return flattened, polygon

def handler(context, event):
    context.logger.info("call handler")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    image = Image.open(buf)
    image = image.convert("RGB")  #  to make sure image comes in RGB

    model1 = context.user_data.yolo1
    model2 = context.user_data.yolo2
    predictor = context.user_data.sam2
    crop_results = model1(image, save=False,iou = 0.3)

    image = np.array(image)
    h,w = image.shape[:2]
    crops,positions = find_crop(image,crop_results)

    n = len(crops)
    img_batch = [crops[i] for i in range(n)]
    yolo_results = [model2(crops[i],save = False,iou = 0.5) for i in range(n)]
    num_box = [len(yolo_results[i][0]) for i in range(n)]
    greens = [check_green(crops[i]) for i in range(n)]
    results = []

    img_sum,data,boxes = new_image(img_batch,yolo_results)
    green = check_green(img_sum)
    points, labels = sam2_input(boxes,green)
    predictor.set_image(img_sum)
    masks, scores, _ = predictor.predict(
        point_coords=points,
        point_labels=labels,
        box=boxes,
        multimask_output=False,
    )

    masks_all,scores_all = process_mask(masks,scores,num_box,data,h,w,positions,greens)

    for mask, score in zip(masks_all, scores_all):
        cvat_mask, polygon = to_cvat_mask(mask)
        if cvat_mask == None : continue
        results.append({
            "confidence": str(float(score)),
            "label": "Spinacia",
            "points": polygon,
            "mask":cvat_mask,
            "type": "mask",
            })
    return context.Response(body=json.dumps(results), headers={},content_type='application/json', status_code=200)