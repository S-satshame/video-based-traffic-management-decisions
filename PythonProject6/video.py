import cv2
import numpy as np

# 加载训练好的 YOLO 模型
cfg_file = 'yolov4.cfg'  # YOLO配置文件
weights_file = 'yolov4.weights'  # 训练好的权重文件
net = cv2.dnn.readNetFromDarknet(cfg_file, weights_file)

# 设置类别（假设有3个类别：car, truck, person）
classes = ["car", "truck", "person"]

# 打开视频文件
video_path = 'videos'
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

    # 绘制检测框并标注
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # 设置置信度阈值
                box = detection[:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                x, y, w, h = box.astype("int")
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{classes[class_id]}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 显示处理后的每一帧
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
