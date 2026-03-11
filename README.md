# Leaf_SAM2

## Quy trình xử lý
1. **Phát hiện cây** : Dùng Yolo1 phát hiện crop chứa cây rau trong ảnh. Nếu ảnh lớn sẽ chia ảnh thành nhiều ảnh nhỏ. 
2. **Phát hiện lá** : Dùng Yolo2 phát hiện box chứa lá cây trong cây rau.
3. **Phân đoạn** : Tạo prompt cho SAM2 phân đoạn từ box của yolo2
4. **Hậu xủ lý** : Lọc mask hợp lệ, làm sạch và ghép về ảnh gốc

*Yolo1 : mô hình yolo11n phát hiện cây rau trong ảnh

*Yolo2 : mô hình yolo11n phát hiện lá rau trong 1 cây 

## Triển khai 
1. Cài đặt CVAT :

    Làm theo hướng dẫn của cvat :

    - [doc cvat](https://docs.cvat.ai/docs/getting_started/)

    - [github cvat](https://github.com/cvat-ai/cvat)

2. Triển khai model :

    Copy [sam2/nuclio](https://github.com/hungtran1210/Leaf_SAM2/tree/main/Leaf_SAM2/sam2/nuclio) vào phần [serverless](https://docs.cvat.ai/docs/getting_started) của CVAT

    Chạy cvat
    ```bash
    docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml up -d
    ```

    Chạy model bằng lệnh :
    ```bash
    ./serverless/deploy_cpu.sh serverless/sam2
    ```
    ```bash
    ./serverless/deploy_gpu.sh serverless/sam2
    ```

## Notebooks demo

1. Minh họa phương pháp : [Leaf](https://github.com/hungtran1210/Leaf_SAM2/blob/main/Leaf_SAM2/notebook/leaf_onnx.ipynb)

2. Train yolo : [Yolo11](https://github.com/hungtran1210/Leaf_SAM2/blob/main/Leaf_SAM2/notebook/Yolo11.ipynb), [Data_crop](https://drive.google.com/drive/folders/1e1oyn57vWUYVmAbeYK6VWvUvFHzDKQ1o?usp=sharing), [Data_leaf](https://drive.google.com/drive/folders/1fGZf1mAJlIW7NxIgkqHs6_R7SPrUFu7Y?usp=drive_link) 

3. Kết quả : [Test](https://github.com/hungtran1210/Leaf_SAM2/blob/main/Leaf_SAM2/notebook/test.ipynb)

## Hình ảnh minh họa 
![alt text](image1.png)
![alt text](image2.png)
![alt text](image3.png)