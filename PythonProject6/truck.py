import cv2
import numpy as np

# 加载 YOLO 配置文件和权重文件
cfg_file = 'yolov4.cfg'  # YOLO 配置文件路径
weights_file = 'yolov4.weights'  # YOLO 权重文件路径
net = cv2.dnn.readNetFromDarknet(cfg_file, weights_file)

# 设置类别（CAR, truck, pedestrian）
classes = ["car", "truck", "pedestrian", "bicycle", "bus", "cat", "dog", "chair", "potted plant", "bench"]

# 打开视频文件
video_path = 'video'  # 视频文件路径
cap = cv2.VideoCapture(video_path)

# 获取视频的帧数
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 处理每一帧
for i in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break

    # 转换为 YOLO 输入格式
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), (0, 0, 0), swapRB=True)
    net.setInput(blob)

    # 获取 YOLO 输出层
    output_layers = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers)

    # 检测物体并标记障碍物
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id in [8, 9, 10]:  # 假设障碍物类别为 'bench', 'potted plant', 'chair' (类别编号8,9,10)
                box = detection[:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                x, y, w, h = box.astype("int")
                center_x = x + w // 2
                center_y = y + h // 2

                # 假设检测到障碍物占道，可以进行标记
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 绘制边框
                cv2.putText(frame, "Obstacle on the road", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 显示结果
    cv2.imshow("Obstacle Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
