import math
import cv2
import numpy as np

def find_crop(image, boxes):
    crops, positions = [], []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box) 
        crops.append(image[y1:y2, x1:x2].copy())
        positions.append((y1, y2, x1, x2))
    return crops, positions

def mask_full(mask, h, w, position):
    mask_f = np.zeros((h, w), dtype=np.uint8)
    y1, y2, x1, x2 = position
    mask_f[y1:y2, x1:x2] = mask
    return mask_f

def clean_mask(masks, sam_scores, yolo_scores, thresh=0.5):
    sam_scores = np.array(sam_scores).flatten()
    yolo_scores = np.array(yolo_scores).flatten()
    combined_scores = sam_scores * yolo_scores
    
    sorted_indices = np.argsort(combined_scores)[::-1]
    
    h, w = masks[0].shape[:2]
    occupied_region = np.zeros((h, w), dtype=bool)

    final_masks = []
    final_scores = []
    
    for idx in sorted_indices:
        current_mask = masks[idx].astype(bool)
        current_area = np.count_nonzero(current_mask)
        
        cleaned_mask = current_mask & (~occupied_region)
        cleaned_area = np.count_nonzero(cleaned_mask)
        
        if current_area > 0 and (cleaned_area / current_area) > thresh:
            final_masks.append(cleaned_mask.astype(np.uint8))
            final_scores.append(combined_scores[idx]) 
            
            occupied_region = occupied_region | cleaned_mask

    return final_masks, final_scores

def yolo_boxes(image, model):
    h, w = image.shape[:2]
    get_factor = lambda size: 4 if size >= 10000 else (2 if size >= 5000 else 1)
    split_h, split_w = get_factor(h), get_factor(w)

    if split_h == 1 and split_w == 1:
        boxes,_ = model(image)
        return boxes
    
    step_h, step_w = h // split_h, w // split_w
    global_yolo_boxes = []
    
    for i in range(split_h):
        for j in range(split_w):
            y1, x1 = i * step_h, j * step_w
            y2 = h if i == split_h - 1 else (i + 1) * step_h
            x2 = w if j == split_w - 1 else (j + 1) * step_w
            
            crop_img = image[y1:y2, x1:x2]
            boxes,_ = model(crop_img) 
            
            if len(boxes) > 0:
                boxes[:, [0, 2]] += x1  
                boxes[:, [1, 3]] += y1  
                global_yolo_boxes.append(boxes)
                
    if len(global_yolo_boxes) > 0:
        return np.vstack(global_yolo_boxes)
    return np.empty((0, 4))

def to_cvat_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None, None
    contour = max(contours, key=cv2.contourArea)
    xtl, ytl, w, h = cv2.boundingRect(contour)
    xbr, ybr = xtl + w - 1, ytl + h - 1

    polygon = contour.flatten().tolist()
    roi = mask[ytl : ybr + 1, xtl : xbr + 1]
    flattened = roi.flat[:].tolist()
    flattened.extend([xtl, ytl, xbr, ybr])
    return flattened, polygon

def run(image, model1, model2, sam2_onnx):
    h, w = image.shape[:2]
    final_boxes = yolo_boxes(image, model1)
    crops, positions = find_crop(image, final_boxes)

    results = []
    masks_all = []
    scores_all = []
    masks_all = []

    for i in range(len(crops)):
        embeddings = sam2_onnx.encode(crops[i])
        boxes,scores_yolo = model2(crops[i],conf =0.5, iou =0.7)

        if len(boxes) == 0: continue
        
        masks = []
        scores_sam = []
        for box in boxes:
            prompt = [{"type": "rectangle", "data": box.tolist()}]
            mask_result, score = sam2_onnx.predict_masks(embeddings, prompt)
            masks.append(mask_result[0, 0])
            scores_sam.append(score)

        final_masks, final_scores = clean_mask(masks, scores_sam, scores_yolo)

        for m,s in zip(final_masks,final_scores) :
            mask_cf = mask_full(m,h,w,positions[i])
            masks_all.append(mask_cf)
            scores_all.append(s)

    for mask, score in zip(masks_all, scores_all):
        cvat_mask, polygon = to_cvat_mask(mask)
        if cvat_mask is None: continue
        results.append({
            "confidence": str(float(score)),
            "label": "Spinacia",
            "points": polygon,
            "mask": cvat_mask,
            "type": "mask",
        })
        
    return results