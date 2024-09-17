import cv2
import numpy as np


def intersection_area(a, b):
    inter = a['rect'] & b['rect']
    return inter.area()


def qsort_descent_inplace(faceobjects):
    if not faceobjects:
        return

    faceobjects.sort(key=lambda obj: obj['prob'], reverse=True)  # 按概率排序
    return faceobjects


def nms_sorted_bboxes(faceobjects, nms_threshold):
    picked = []
    n = len(faceobjects)
    areas = [obj['rect'].area() for obj in faceobjects]

    for i in range(n):
        a = faceobjects[i]
        keep = True
        for j in picked:
            b = faceobjects[j]
            inter_area = intersection_area(a, b)
            union_area = areas[i] + areas[j] - inter_area

            if inter_area / union_area > nms_threshold:
                keep = False
                break

        if keep:
            picked.append(i)

    return picked


def generate_anchors(base_size, ratios, scales):
    num_ratio = len(ratios)
    num_scale = len(scales)

    anchors = np.zeros((num_ratio * num_scale, 4), dtype=np.float32)
    cx = cy = 0

    for i in range(num_ratio):
        ar = ratios[i]
        r_w = round(base_size / (ar ** 0.5))
        r_h = round(r_w * ar)

        for j in range(num_scale):
            scale = scales[j]
            rs_w = r_w * scale
            rs_h = r_h * scale

            index = i * num_scale + j
            anchors[index, 0] = cx - rs_w * 0.5
            anchors[index, 1] = cy - rs_h * 0.5
            anchors[index, 2] = cx + rs_w * 0.5
            anchors[index, 3] = cy + rs_h * 0.5

    return anchors


def generate_proposals(anchors, feat_stride, score_blob, bbox_blob, kps_blob, prob_threshold):
    h, w = score_blob.shape[1], score_blob.shape[2]
    num_anchors = anchors.shape[0]
    faceobjects = []

    for q in range(num_anchors):
        anchor = anchors[q]

        score = score_blob[q]  # 频道的得分
        bbox = bbox_blob[q]  # 频道的边框

        anchor_y = anchor[1]
        anchor_w = anchor[2] - anchor[0]
        anchor_h = anchor[3] - anchor[1]

        for i in range(h):
            anchor_x = anchor[0]
            for j in range(w):
                index = i * w + j
                prob = score[i, j]

                if prob >= prob_threshold:
                    dx = bbox[0, index] * feat_stride
                    dy = bbox[1, index] * feat_stride
                    dw = bbox[2, index] * feat_stride
                    dh = bbox[3, index] * feat_stride

                    cx = anchor_x + anchor_w * 0.5
                    cy = anchor_y + anchor_h * 0.5

                    x0 = cx - dx
                    y0 = cy - dy
                    x1 = cx + dw
                    y1 = cy + dh

                    obj = {
                        'rect': cv2.Rect(x0, y0, x1 - x0, y1 - y0),
                        'prob': prob,
                        'landmark': []  # 假设没有关键点信息
                    }

                    if kps_blob is not None:
                        # 在这里处理关键点数据（如适用）
                        pass

                    faceobjects.append(obj)

                anchor_x += feat_stride
            anchor_y += feat_stride

    return faceobjects


class SCRFD:
    def __init__(self):
        self.scrfd = None  # Placeholder for the NCNN model
        self.has_kps = False  # Indicates if keypoints are available

    def load(self, parampath, modelpath, use_gpu=False):
        # 载入模型参数和模型
        import ncnn  # 确保已经安装 ncnn 的 Python 包

        # 假设 load_model 和 load_param 是 ncnn 提供的方法
        self.scrfd = ncnn.Net()  # Placeholder for NCNN model
        self.scrfd.load_param(parampath)
        self.scrfd.load_model(modelpath)

    def detect(self, rgb, prob_threshold, nms_threshold):
        height, width, _ = rgb.shape
        target_size = 640

        w, h, scale = width, height, 1.0
        if w > h:
            scale = target_size / w
            w, h = target_size, int(h * scale)
        else:
            scale = target_size / h
            h, w = target_size, int(w * scale)

        in_mat = cv2.resize(rgb, (w, h))  # 假设使用 OpenCV 进行预处理

        # 归一化和边框处理
        in_mat = in_mat.astype(np.float32)
        in_mat -= np.array([127.5, 127.5, 127.5])
        in_mat *= (1 / 128)

        faceproposals = []

        # 应用 NCNN 模型进行检测，省略模型提取和便于理解的部分（待实现）

        # 此处需调用 NCNN 进行前向推理，获得 score_blob, bbox_blob 和 kps_blob，然后生成提案
        anchors = generate_anchors(16, [1.0], [1.0, 2.0])
        faceproposals += generate_proposals(anchors, 8, score_blob, bbox_blob, None, prob_threshold)

        # 进行 NMS
        faceproposals = qsort_descent_inplace(faceproposals)
        picked = nms_sorted_bboxes(faceproposals, nms_threshold)

        return [faceproposals[i] for i in picked]

    def draw(self, rgb, faceobjects):
        for obj in faceobjects:
            cv2.rectangle(rgb, (obj['rect'].x, obj['rect'].y),
                          (obj['rect'].x + obj['rect'].width, obj['rect'].y + obj['rect'].height), (0, 0, 255), 2)

            if self.has_kps:
                for lm in obj['landmark']:
                    cv2.circle(rgb, (int(lm.x), int(lm.y)), 2, (255, 255, 0), -1)

            text = f"{obj['prob'] * 100:.1f}%"
            cv2.putText(rgb, text, (obj['rect'].x, obj['rect'].y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return rgb

# 示例使用
# scrfd_detector = SCRFD()
# scrfd_detector.load("model.param", "model.bin")
# results = scrfd_detector.detect(cv2.imread('image.jpg'), 0.5, 0.4)
# scrfd_detector.draw(cv2.imread('image.jpg'), results)
