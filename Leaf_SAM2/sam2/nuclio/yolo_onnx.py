import cv2
import numpy as np
import onnxruntime as ort

class YOLO_ONNX:
    def __init__(self, model_path, conf=0.25, iou=0.7):
        self.conf = conf
        self.iou = iou
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        so = ort.SessionOptions()
        so.log_severity_level = 3

        self.model = ort.InferenceSession(model_path, providers=providers, sess_options=so)
        self.output_details = [i.name for i in self.model.get_outputs()]
        self.input_details = [i.name for i in self.model.get_inputs()]
        
        # Tự động lấy kích thước đầu vào từ file ONNX
        input_shape = self.model.get_inputs()[0].shape
        self.input_h = input_shape[2] 
        self.input_w = input_shape[3] 

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleup=True, stride=32):
        # Resize ảnh giữ đúng tỷ lệ khung hình
        shape = im.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

        if auto:
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)

        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return im, r, (dw, dh)

    # Đã thêm conf và iou vào đây để truyền động
    def __call__(self, image: np.ndarray, conf=None, iou=None):
        current_conf = conf if conf is not None else self.conf
        current_iou = iou if iou is not None else self.iou
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        img, ratio, dwdh = self.letterbox(img, new_shape=(self.input_h, self.input_w), auto=False)
        
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, 0)
        img = np.ascontiguousarray(img)

        im = img.astype(np.float32)
        im /= 255.0

        inp = {self.input_details[0]: im}
        preds = self.model.run(self.output_details, inp)[0]
        preds = np.squeeze(preds) 
        preds = preds.T           

        boxes_data = preds[:, :4] 
        scores_data = preds[:, 4:]

        class_ids = np.argmax(scores_data, axis=1)
        confidences = np.max(scores_data, axis=1)

        mask = confidences > current_conf
        boxes_data = boxes_data[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        output_boxes = []
        output_scores = [] 

        if len(boxes_data) > 0:
            cx, cy, w, h = boxes_data[:, 0], boxes_data[:, 1], boxes_data[:, 2], boxes_data[:, 3]
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            
            boxes_xyxy = np.column_stack((x1, y1, x2, y2))

            nms_indices = cv2.dnn.NMSBoxes(boxes_xyxy.tolist(), confidences.tolist(), current_conf, current_iou)

            if len(nms_indices) > 0:
                for i in nms_indices.flatten():
                    box = boxes_xyxy[i]
                    box -= np.array(dwdh * 2) 
                    box /= ratio
                    box = box.round().astype(np.int32)
                    h_img, w_img = image.shape[:2]
                    box[0] = max(0, box[0])
                    box[1] = max(0, box[1])
                    box[2] = min(w_img, box[2])
                    box[3] = min(h_img, box[3])

                    output_boxes.append(box)
                    output_scores.append(confidences[i])
        return np.array(output_boxes), np.array(output_scores)