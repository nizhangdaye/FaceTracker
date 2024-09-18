import cv2
import numpy as np
import ncnn


class FaceObject:
    def __init__(self):
        self.rect = None  # [x, y, width, height]
        self.prob = 0.0
        self.landmark = [None] * 5  # 关键点位置


def intersection_area(a, b):
    inter = (max(0, min(a.rect[0] + a.rect[2], b.rect[0] + b.rect[2]) - max(a.rect[0], b.rect[0])),
             max(0, min(a.rect[1] + a.rect[3], b.rect[1] + b.rect[3]) - max(a.rect[1], b.rect[1])))
    return inter[0] * inter[1]


def qsort_descent_inplace(faceobjects):
    if not faceobjects:
        return
    faceobjects.sort(key=lambda x: x.prob, reverse=True)


def nms_sorted_bboxes(faceobjects, nms_threshold):
    picked = []
    n = len(faceobjects)
    areas = [obj.rect[2] * obj.rect[3] for obj in faceobjects]

    for i in range(n):
        a = faceobjects[i]
        keep = True
        for j in picked:
            b = faceobjects[j]
            inter_area = intersection_area(a, b)
            union_area = areas[i] + areas[picked[j]] - inter_area
            if inter_area / union_area > nms_threshold:
                keep = False
                break
        if keep:
            picked.append(i)
    return picked


def generate_anchors(base_size, ratios, scales):
    anchors = []
    cx, cy = 0, 0
    for ar in ratios:
        r_w = round(base_size / np.sqrt(ar))
        r_h = round(r_w * ar)
        for scale in scales:
            rs_w = r_w * scale
            rs_h = r_h * scale
            anchors.append([cx - rs_w * 0.5, cy - rs_h * 0.5, cx + rs_w * 0.5, cy + rs_h * 0.5])
    return np.array(anchors)


def generate_proposals(anchors, feat_stride, score_blob, bbox_blob, kps_blob, prob_threshold, faceobjects):
    h, w = score_blob.shape()[1:3]
    for q in range(anchors.shape[0]):
        anchor = anchors[q]
        score = score_blob[q]
        bbox = bbox_blob[q]
        anchor_y = anchor[1]

        for i in range(h):
            anchor_x = anchor[0]
            for j in range(w):
                index = i * w + j

                prob = score[index]
                if prob >= prob_threshold:
                    dx = bbox[0][index] * feat_stride
                    dy = bbox[1][index] * feat_stride
                    dw = bbox[2][index] * feat_stride
                    dh = bbox[3][index] * feat_stride

                    cx = anchor_x + (anchor[2] - anchor[0]) * 0.5
                    cy = anchor_y + (anchor[3] - anchor[1]) * 0.5

                    x0 = cx - dx
                    y0 = cy - dy
                    x1 = cx + dw
                    y1 = cy + dh

                    obj = FaceObject()
                    obj.rect = [x0, y0, x1 - x0, y1 - y0]
                    obj.prob = prob

                    if kps_blob is not None and len(kps_blob) > 0:
                        for kp in range(5):
                            obj.landmark[kp] = (int(cx + kps_blob[q][kp * 2] * feat_stride),
                                                int(cy + kps_blob[q][kp * 2 + 1] * feat_stride))

                    faceobjects.append(obj)
                anchor_x += feat_stride
            anchor_y += feat_stride


class SCRFD:
    def __init__(self):
        self.scrfd = ncnn.Net()
        self.has_kps = True  # 假设有关键点

    def load(self, parampath, modelpath, use_gpu=True):
        # 加载模型
        if use_gpu:
            self.scrfd.load_param(parampath)
            self.scrfd.load_model(modelpath)
        else:
            self.scrfd.load_param_cpu(parampath)
            self.scrfd.load_model_cpu(modelpath)

    def detect(self, rgb, faceobjects, prob_threshold, nms_threshold):
        height, width = rgb.shape[:2]
        target_size = 640

        scale = min(target_size / height, target_size / width)
        w, h = int(width * scale), int(height * scale)

        in_pad = cv2.resize(rgb, (w, h))
        in_pad = cv2.cvtColor(in_pad, cv2.COLOR_BGR2RGB)
        in_pad = in_pad.astype(np.float32)

        mean_vals = np.array([127.5, 127.5, 127.5], dtype=np.float32)
        norm_vals = np.array([1 / 128.0, 1 / 128.0, 1 / 128.0], dtype=np.float32)

        in_pad = (in_pad - mean_vals) * norm_vals

        faceproposals = []
        extractor = self.scrfd.create_extractor()

        for stride in [8, 16, 32]:
            score_blob = ncnn.Mat()
            bbox_blob = ncnn.Mat()
            kps_blob = ncnn.Mat() if self.has_kps else None  # 确保 kps_blob 被初始化

            extractor.input("input.1", ncnn.Mat(in_pad))
            print(f"尝试提取 score 和 bbox，当前 stride: {stride}")

            try:
                extractor.extract(f"score_{stride}", score_blob)
                extractor.extract(f"bbox_{stride}", bbox_blob)
                if self.has_kps:
                    print(f"尝试提取 kps_{stride}")
                    extractor.extract(f"kps_{stride}", kps_blob)
            except Exception as e:
                print(f"提取过程中发生错误: {e}")

            # 检查提取的 blob 是否有效
            if score_blob.empty() or bbox_blob.empty():
                print(f"提取的 score_blob 或 bbox_blob 在 stride {stride} 上为空，可能构建失败")
                continue  # 继续到下一个 stride

            # 从 score_blob 获取大小
            h, w = score_blob.shape()[1:3]  # 确保使用括号调用 shape

            # 生成提案逻辑
            base_size = 16 if stride == 8 else 64 if stride == 16 else 256
            anchors = generate_anchors(base_size, [1.0], [1.0, 2.0])
            generate_proposals(anchors, stride, score_blob, bbox_blob, kps_blob, prob_threshold, faceproposals)

        qsort_descent_inplace(faceproposals)
        picked = nms_sorted_bboxes(faceproposals, nms_threshold)

        for i in picked:
            faceobjects.append(faceproposals[i])
        return faceobjects

    def draw(self, rgb, faceobjects):
        for obj in faceobjects:
            cv2.rectangle(rgb, (int(obj.rect[0]), int(obj.rect[1])),
                          (int(obj.rect[0] + obj.rect[2]), int(obj.rect[1] + obj.rect[3])),
                          (0, 0, 255), 2)
            if self.has_kps:
                for lm in obj.landmark:
                    if lm is not None:
                        cv2.circle(rgb, lm, 2, (255, 255, 0), -1)

            text = f"{obj.prob * 100:.1f}%"
            cv2.putText(rgb, text, (int(obj.rect[0]), int(obj.rect[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


# 实际使用示例
if __name__ == '__main__':
    scrfd = SCRFD()
    scrfd.load('path_to_param', 'path_to_model')
    rgb_image = cv2.imread('path_to_image')
    face_objects = []
    scrfd.detect(rgb_image, face_objects, prob_threshold=0.5, nms_threshold=0.45)
    scrfd.draw(rgb_image, face_objects)
    cv2.imshow('Detection', rgb_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
