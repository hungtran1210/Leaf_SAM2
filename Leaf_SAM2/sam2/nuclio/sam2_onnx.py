import time
from typing import Any

import cv2
import numpy as np
import onnxruntime
from numpy import ndarray


class SegmentAnything2ONNX:
    """Segmentation model using Segment Anything 2 (SAM2)"""

    def __init__(self, encoder_model_path, decoder_model_path) -> None:
        self.encoder = SAM2ImageEncoder(encoder_model_path)
        self.decoder = SAM2ImageDecoder(
            decoder_model_path, self.encoder.input_shape[2:]
        )

    def encode(self, cv_image: np.ndarray) -> dict:
        original_size = cv_image.shape[:2]
        high_res_feats_0, high_res_feats_1, image_embed = self.encoder(cv_image)
        return {
            "high_res_feats_0": high_res_feats_0,
            "high_res_feats_1": high_res_feats_1,
            "image_embedding": image_embed,
            "original_size": original_size,
        }

    def predict_masks(self, embedding, prompt) -> list[np.ndarray]:
        points = []
        labels = []
        for mark in prompt:
            if mark["type"] == "point":
                points.append(mark["data"])
                labels.append(mark["label"])
            elif mark["type"] == "rectangle":
                points.append([mark["data"][0], mark["data"][1]])  # top left
                points.append([mark["data"][2], mark["data"][3]])  # bottom right
                labels.append(2)
                labels.append(3)
        points, labels = np.array(points), np.array(labels)

        image_embedding = embedding["image_embedding"]
        high_res_feats_0 = embedding["high_res_feats_0"]
        high_res_feats_1 = embedding["high_res_feats_1"]
        original_size = embedding["original_size"]
        self.decoder.set_image_size(original_size)
        masks, scores = self.decoder(
            image_embedding,
            high_res_feats_0,
            high_res_feats_1,
            points,
            labels,
        )

        return masks, scores

    def transform_masks(self, masks, original_size, transform_matrix):
        output_masks = []
        for batch in range(masks.shape[0]):
            batch_masks = []
            for mask_id in range(masks.shape[1]):
                mask = masks[batch, mask_id]
                mask = cv2.warpAffine(
                    mask,
                    transform_matrix[:2],
                    (original_size[1], original_size[0]),
                    flags=cv2.INTER_LINEAR,
                )
                batch_masks.append(mask)
            output_masks.append(batch_masks)
        return np.array(output_masks)


class SAM2ImageEncoder:
    def __init__(self, path: str) -> None:
        requested_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(path, providers=requested_providers)
        active_providers = self.session.get_providers()
        self.use_gpu = 'CUDAExecutionProvider' in active_providers
        
        if not self.use_gpu:
            print("No GPU, use CPU.")

        self.get_input_details()
        self.get_output_details()

    def __call__(self, image: np.ndarray) -> tuple[Any, Any, Any]:
        return self.encode_image(image)

    def encode_image(self, image: np.ndarray):
        input_tensor = self.prepare_input(image)
        outputs = self.infer(input_tensor)
        return self.process_output(outputs)

    def prepare_input(self, image: np.ndarray) -> np.ndarray:
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        input_img = (input_img / 255.0 - mean) / std
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def infer(self, input_tensor: np.ndarray) -> list:
        start = time.perf_counter()
        
        if self.use_gpu:
            io_binding = self.session.io_binding()
            input_ort = onnxruntime.OrtValue.ortvalue_from_numpy(input_tensor, 'cuda', 0)
            io_binding.bind_ortvalue_input(self.input_names[0], input_ort)
            
            for name in self.output_names:
                io_binding.bind_output(name, 'cuda')
                
            self.session.run_with_iobinding(io_binding)
            outputs = io_binding.get_outputs()
            print(f"Encoder infer time: {(time.perf_counter() - start) * 1000:.2f} ms")
        else:
            outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
            print(f"Encoder infer time: {(time.perf_counter() - start) * 1000:.2f} ms")
            
        return outputs

    def process_output(self, outputs: list):
        return outputs[0], outputs[1], outputs[2]

    def get_input_details(self) -> None:
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self) -> None:
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]


class SAM2ImageDecoder:
    def __init__(
        self,
        path: str,
        encoder_input_size: tuple[int, int],
        orig_im_size: tuple[int, int] = None,
        mask_threshold: float = 0.0,
    ) -> None:
        requested_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(path, providers=requested_providers)        
        active_providers = self.session.get_providers()
        self.use_gpu = 'CUDAExecutionProvider' in active_providers

        if not self.use_gpu:
            print("No GPU, use CPU")

        self.orig_im_size = orig_im_size if orig_im_size is not None else encoder_input_size
        self.encoder_input_size = encoder_input_size
        self.mask_threshold = mask_threshold
        self.scale_factor = 4

        self.get_input_details()
        self.get_output_details()

    def __call__(
        self,
        image_embed,
        high_res_feats_0,
        high_res_feats_1,
        point_coords,
        point_labels,
    ):
        return self.predict(
            image_embed, high_res_feats_0, high_res_feats_1, point_coords, point_labels
        )

    def predict(
        self,
        image_embed,
        high_res_feats_0,
        high_res_feats_1,
        point_coords,
        point_labels,
    ):
        inputs = self.prepare_inputs(
            image_embed, high_res_feats_0, high_res_feats_1, point_coords, point_labels
        )
        outputs = self.infer(inputs)
        return self.process_output(outputs)

    def prepare_inputs(
        self,
        image_embed,
        high_res_feats_0,
        high_res_feats_1,
        point_coords,
        point_labels,
    ):
        input_point_coords, input_point_labels = self.prepare_points(point_coords, point_labels)

        num_labels = input_point_labels.shape[0]
        mask_input = np.zeros(
            (
                num_labels,
                1,
                self.encoder_input_size[0] // self.scale_factor,
                self.encoder_input_size[1] // self.scale_factor,
            ),
            dtype=np.float32,
        )
        has_mask_input = np.array([0], dtype=np.float32)

        return (
            image_embed,
            high_res_feats_0,
            high_res_feats_1,
            input_point_coords,
            input_point_labels,
            mask_input,
            has_mask_input,
        )

    def prepare_points(
        self,
        point_coords: list[np.ndarray] | np.ndarray,
        point_labels: list[np.ndarray] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        if isinstance(point_coords, np.ndarray):
            input_point_coords = point_coords[np.newaxis, ...]
            input_point_labels = point_labels[np.newaxis, ...]
        else:
            max_num_points = max([coords.shape[0] for coords in point_coords])
            input_point_coords = np.zeros((len(point_coords), max_num_points, 2), dtype=np.float32)
            input_point_labels = np.ones((len(point_coords), max_num_points), dtype=np.float32) * -1

            for i, (coords, labels) in enumerate(zip(point_coords, point_labels)):
                input_point_coords[i, : coords.shape[0], :] = coords
                input_point_labels[i, : labels.shape[0]] = labels

        input_point_coords[..., 0] = (input_point_coords[..., 0] / self.orig_im_size[1] * self.encoder_input_size[1])
        input_point_coords[..., 1] = (input_point_coords[..., 1] / self.orig_im_size[0] * self.encoder_input_size[0])

        return input_point_coords.astype(np.float32), input_point_labels.astype(np.float32)

    def infer(self, inputs) -> list:
        start = time.perf_counter()

        if self.use_gpu:
            io_binding = self.session.io_binding()
            for i, inp in enumerate(inputs):
                name = self.input_names[i]
                if isinstance(inp, onnxruntime.OrtValue):
                    io_binding.bind_ortvalue_input(name, inp)
                else:
                    inp_ort = onnxruntime.OrtValue.ortvalue_from_numpy(inp, 'cuda', 0)
                    io_binding.bind_ortvalue_input(name, inp_ort)
                    
            for name in self.output_names:
                io_binding.bind_output(name, 'cuda')
                
            self.session.run_with_iobinding(io_binding)
            outputs = io_binding.get_outputs()
            print(f"Decoder infer time: {(time.perf_counter() - start) * 1000:.2f} ms")
        else:
            outputs = self.session.run(
                self.output_names,
                {self.input_names[i]: inputs[i] for i in range(len(self.input_names))},
            )
            print(f"Decoder infer time: {(time.perf_counter() - start) * 1000:.2f} ms")
            
        return outputs

    def process_output(self, outputs: list):
        if self.use_gpu:
            scores = outputs[1].numpy().squeeze()
            masks = outputs[0].numpy()[0]
        else:
            scores = outputs[1].squeeze()
            masks = outputs[0][0]
        
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        best_mask = cv2.resize(best_mask, (self.orig_im_size[1], self.orig_im_size[0]))
        
        binary_mask = (best_mask > 0.0).astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        smoothed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        return np.array([[smoothed_mask]]), float(scores[best_idx])

    def set_image_size(self, orig_im_size: tuple[int, int]) -> None:
        self.orig_im_size = orig_im_size

    def get_input_details(self) -> None:
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

    def get_output_details(self) -> None:
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]