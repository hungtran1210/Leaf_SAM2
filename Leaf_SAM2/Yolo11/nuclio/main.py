import base64
import io
import json
from PIL import Image
from ultralytics import YOLO


def init_context(context):
    context.logger.info("Init context...  0%")
    model = YOLO("/opt/nuclio/sam2/yolo1.pt") # yolo1 : cây rau ,yolo2 : lá rau
    context.user_data.yolo = model
    context.logger.info("Init context...100%")


def handler(context, event):
    context.logger.info("Yolo11")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    image = Image.open(buf).convert("RGB")

    model = context.user_data.yolo
    crops = model.predict(image, save=False,iou = 0.3)
    results = []

    for i in crops:
        for box in i.boxes:
            xtl, ytl, xbr, ybr = box.xyxy[0].tolist()
            score = float(box.conf[0])
            results.append({
                "confidence": str(score),
                "label": "Spinacia",
                "points": [xtl, ytl, xbr, ybr],
                "type": "rectangle",
            })

    return context.Response(body=json.dumps(results), headers={},
        content_type='application/json', status_code=200)