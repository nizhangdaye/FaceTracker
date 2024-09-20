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
    num_ratio = len(ratios)
    num_scale = len(scales)
    # 创建一个用于存储锚框的数组，形状为(num_ratio * num_scale, 4)
    anchors = np.zeros((num_ratio * num_scale, 4), dtype=np.float32)
    cx = 0.0
    cy = 0.0
    for i in range(num_ratio):
        ar = ratios[i]
        # 计算基础宽度和高度
        r_w = round(base_size / np.sqrt(ar))
        r_h = round(r_w * ar)
        for j in range(num_scale):
            scale = scales[j]
            # 计算缩放后的宽度和高度
            rs_w = r_w * scale
            rs_h = r_h * scale
            # 计算锚框的四个边界
            index = i * num_scale + j
            anchors[index, 0] = cx - rs_w * 0.5  # 左边界
            anchors[index, 1] = cy - rs_h * 0.5  # 上边界
            anchors[index, 2] = cx + rs_w * 0.5  # 右边界
            anchors[index, 3] = cy + rs_h * 0.5  # 下边界
    return anchors


def generate_proposals(anchors: np.ndarray, feat_stride: int, score_blob: ncnn.Mat, bbox_blob: ncnn.Mat,
                       kps_blob: ncnn.Mat, prob_threshold: float, faceobjects: list):
    h, w = score_blob.h, score_blob.w
    num_anchors = anchors.shape[0]
    print(f"锚框数量：{num_anchors}, 特征图大小：{w}x{h}")

    for q in range(num_anchors):
        anchor = anchors[q, :]

        score = score_blob.channel(q)
        bbox = [bbox_blob.channel(q * 4 + i) for i in range(4)]  # 手动提取 bbox

        anchor_y = anchor[1]
        anchor_w = anchor[2] - anchor[0]
        anchor_h = anchor[3] - anchor[1]

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

                    cx = anchor_x + anchor_w * 0.5
                    cy = anchor_y + anchor_h * 0.5

                    x0 = cx - dx
                    y0 = cy - dy
                    x1 = cx + dw
                    y1 = cy + dh

                    obj = FaceObject()
                    obj.rect.x = x0
                    obj.rect.y = y0
                    obj.rect.width = x1 - x0 + 1
                    obj.rect.height = y1 - y0 + 1
                    obj.prob = prob

                    if kps_blob.size > 0:
                        kps = [kps_blob[q * 10 + i] for i in range(10)]  # 手动提取 kps 值

                        obj.landmark[0].x = cx + kps[0][index] * feat_stride
                        obj.landmark[0].y = cy + kps[1][index] * feat_stride
                        obj.landmark[1].x = cx + kps[2][index] * feat_stride
                        obj.landmark[1].y = cy + kps[3][index] * feat_stride
                        obj.landmark[2].x = cx + kps[4][index] * feat_stride
                        obj.landmark[2].y = cy + kps[5][index] * feat_stride
                        obj.landmark[3].x = cx + kps[6][index] * feat_stride
                        obj.landmark[3].y = cy + kps[7][index] * feat_stride
                        obj.landmark[4].x = cx + kps[8][index] * feat_stride
                        obj.landmark[4].y = cy + kps[9][index] * feat_stride

                    faceobjects.append(obj)
                anchor_x += feat_stride
            anchor_y += feat_stride


class SCRFD:
    def __init__(self, num_threads=4, use_gpu=False):
        self.scrfd = ncnn.Net()
        # 打印模型结构
        self.has_kps = False  # 假设有关键点

    def load(self, parampath, modelpath):
        # 加载模型
        self.scrfd.load_param(parampath)
        self.scrfd.load_model(modelpath)
        # # 打印模型结构
        # for i, layer in enumerate(self.scrfd.layers()):
        #     print(f"Layer {i}: Name = {layer.name},"
        #           f"Type = {layer.type}, Input = {layer.bottoms},"
        #           f"Output = {layer.tops}")

    def detect(self, rgb: cv2.Mat, faceobjects: list, prob_threshold=0.5, nms_threshold=0.45) -> None:
        height, width = rgb.shape[:2]

        # insightface/detection/scrfd/configs/scrfd/scrfd_500m.py
        target_size = 640

        # pad to multiple of 32
        w, h = width, height
        scale = 1.0
        if w > h:
            scale = target_size / float(w)
            w = target_size
            h = int(h * scale)
        else:
            scale = target_size / float(h)
            h = target_size
            w = int(w * scale)

        # 转化为 NCNN 格式
        resized_rgb = cv2.resize(rgb, (w, h))
        rgb_resized = cv2.cvtColor(resized_rgb, cv2.COLOR_BGR2RGB)
        ncnn_mat = np.array(rgb_resized, dtype=np.float32)

        # pad to target_size rectangle
        wpad = (w + 31) // 32 * 32 - w
        hpad = (h + 31) // 32 * 32 - h
        in_pad = cv2.copyMakeBorder(ncnn_mat, 0, hpad, 0, wpad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        mean_vals = [127.5, 127.5, 127.5]
        norm_vals = [1 / 128.0, 1 / 128.0, 1 / 128.0]

        # Normalize the input
        in_pad = (in_pad - np.array(mean_vals)) * np.array(norm_vals)

        in_ncnn = ncnn.Mat.from_pixels_resize(in_pad, ncnn.Mat.PixelType.PIXEL_RGB, w, h, target_size, target_size)

        ex = self.scrfd.create_extractor()

        ex.input("input.1", in_ncnn)

        print(type(ex))

        faceproposals = []

        # stride 8, 16, 32 handling
        for stride in [8, 16, 32]:
            score_blob = ncnn.Mat()
            bbox_blob = ncnn.Mat()
            kps_blob = ncnn.Mat() if self.has_kps else None

            ex.extract(f"score_{stride}", score_blob)
            ex.extract(f"bbox_{stride}", bbox_blob)
            # 打印信息个数
            print(
                f"stride = {stride}, score_blob.c = {score_blob.c}, score_blob.h = {score_blob.h}, score_blob.w = {score_blob.w}")
            print(
                f"stride = {stride}, bbox_blob.c = {bbox_blob.c}, bbox_blob.h = {bbox_blob.h}, bbox_blob.w = {bbox_blob.w}")

            if self.has_kps:
                ex.extract(f"kps_{stride}", kps_blob)

            base_size = 16 if stride == 8 else 64 if stride == 16 else 256
            feat_stride = stride
            ratios = np.array([1.0])
            scales = np.array([1.0, 2.0])

            # 生成锚框
            anchors = generate_anchors(base_size, ratios, scales)
            # print(anchors)
            # 生成候选框
            generate_proposals(anchors, feat_stride, score_blob, bbox_blob, kps_blob, prob_threshold, faceobjects)
            # 打印候选框信息
            print(f"stride = {stride}, {len(faceobjects)} faces")
            faceproposals.extend(faceobjects)

        # sort all proposals by score from highest to lowest
        qsort_descent_inplace(faceproposals)

        # apply nms with nms_threshold
        picked = nms_sorted_bboxes(faceproposals, nms_threshold)

        face_count = len(picked)
        print(f"Detected {face_count} faces")
        faceobjects = [faceproposals[i] for i in picked]

        for face in faceobjects:
            # adjust offset to original unpadded
            face['rect']['x'] = (face['rect']['x'] - wpad / 2) / scale
            face['rect']['y'] = (face['rect']['y'] - hpad / 2) / scale
            face['rect']['width'] = (face['rect']['width'] - wpad / 2) / scale
            face['rect']['height'] = (face['rect']['height'] - hpad / 2) / scale

            # handle landmarks if available
            if self.has_kps:
                for i in range(5):
                    face['landmark'][i]['x'] = (face['landmark'][i]['x'] - wpad / 2) / scale
                    face['landmark'][i]['y'] = (face['landmark'][i]['y'] - hpad / 2) / scale

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
    scrfd.load('../assets/scrfd_10g-opt2.param',
               '../assets/scrfd_10g-opt2.bin')
    rgb_image = cv2.imread('../test/1.png')
    face_objects = []
    scrfd.detect(rgb_image, face_objects, prob_threshold=0.5, nms_threshold=0.45)
    scrfd.draw(rgb_image, face_objects)
    cv2.imwrite('../test/result.png', rgb_image)
