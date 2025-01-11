import cv2
import numpy as np
import os

image_path = "D:/picture/dcj.jpg"
# 确保路径标准化
image_path = os.path.abspath(image_path)

# 读取图像
image = cv2.imread(image_path)

# 加载 YOLO 配置和预训练权重文件
config_path = "yolov3.cfg"  # 替换为你的配置文件路径
weights_path = "yolov3.weights"  # 替换为你的权重文件路径
labels_path = "coco.names"  # 替换为你的标签文件路径

# 读取类别标签
with open(labels_path, 'r') as f:
    labels = f.read().strip().split('\n')

# 为 YOLO 模型加载网络
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # 如果有 GPU，可以改为 DNN_TARGET_CUDA

# 获取 YOLO 输出层的名称
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# 加载输入图像
image = cv2.imread(image_path)
(H, W) = image.shape[:2]

# 将图像转换为 blob（YOLO 的输入格式）
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

# 运行前向传播，获得检测结果
layer_outputs = net.forward(output_layers)

# 初始化检测框、置信度和类别列表
boxes = []
confidences = []
class_ids = []

# 遍历每个输出层的结果
for output in layer_outputs:
    for detection in output:
        # 提取类别分数和类别 ID
        scores = detection[5:]  # 类别分数从第 6 个值开始
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        # 过滤掉低置信度的检测
        if confidence > 0.5:  # 置信度阈值
            # 获取检测框的中心坐标和宽高
            box = detection[:4] * np.array([W, H, W, H])
            (center_x, center_y, width, height) = box.astype("int")

            # 计算检测框的左上角坐标
            x = int(center_x - (width / 2))
            y = int(center_y - (height / 2))

            # 保存检测框、置信度和类别 ID
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# 应用非极大值抑制（NMS）以去除冗余框
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# 在图像上绘制检测框
if len(indices) > 0:
    for i in indices.flatten():
        (x, y, w, h) = boxes[i]
        color = [int(c) for c in np.random.randint(0, 255, size=(3,), dtype="uint8")]

        # 绘制边界框和标签
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 输出检测结果
        print(f"检测到：{labels[class_ids[i]]}, 置信度：{confidences[i]:.2f}, 边框：[{x}, {y}, {w}, {h}]")
else:
    print("没有检测到物体！")

# 显示结果图像
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存结果图像
cv2.imwrite("output.jpg", image)
