import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 加载 YOLO 配置文件和权重文件
cfg_file = 'yolov4.cfg'  # YOLO 配置文件路径
weights_file = 'yolov4.weights'  # YOLO 权重文件路径
net = cv2.dnn.readNetFromDarknet(cfg_file, weights_file)

# 设置类别（CAR, truck, pedestrian）
classes = ["car", "truck", "pedestrian"]

# 打开视频文件
video_path = 'path_to_your_video.mp4'  # 视频文件路径
cap = cv2.VideoCapture(video_path)

# 获取视频的帧数
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 存储车辆的中心点位置
vehicle_centers = []

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

    # 检测车辆并提取车辆中心点
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # 0 对应车辆（car）
                box = detection[:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                x, y, w, h = box.astype("int")
                center_x = x + w // 2
                center_y = y + h // 2
                vehicle_centers.append((center_x, center_y))

# 使用 KMeans 聚类来识别排队长度
vehicle_centers_np = np.array(vehicle_centers)

# 假设车队只有一个簇
kmeans = KMeans(n_clusters=1)  # 假设聚成一个簇，表示排队区域
kmeans.fit(vehicle_centers_np)

# 获取聚类标签和聚类中心
labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

# 计算排队长度（车队中的车辆数量）
queue_length = len(vehicle_centers_np)  # 车辆数量即为排队长度
print(f"Queue Length: {queue_length} vehicles")

# 可视化排队区域
plt.scatter(vehicle_centers_np[:, 0], vehicle_centers_np[:, 1], c=labels, cmap='viridis')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='X', color='red')  # 聚类中心
plt.title("Queue Length Detection")
plt.show()

# 在视频中显示排队区域
cap = cv2.VideoCapture(video_path)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 绘制聚类中心
    for center in cluster_centers:
        cv2.circle(frame, (int(center[0]), int(center[1])), 10, (0, 0, 255), -1)

    # 显示处理后的帧
    cv2.imshow("Queue Length Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
