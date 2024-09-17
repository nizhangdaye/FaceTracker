import os
import cv2
import numpy as np
from pathlib import Path
from scrfd_detector import SCRFD
from multiobject_tracker import MultiObjectTracker
from utils import is_image_file, is_video_file, print_progress_bar
from cluster import cluster_faces_for_id

clustering_done = False
initial_face_boxes = []
clustered_rects = []
face_trackers = {}
face_tracker_ids = {}


def process_file(file_path, detector, result_dir):
    if is_image_file(str(file_path)):
        img = cv2.imread(str(file_path))
        print(f"[INFO] Processing image: {file_path}")
        if img is None:
            print(f"[ERROR] cv2.imread {file_path} failed")
            return

        prob_threshold = 0.5  # 人脸置信度阈值
        nms_threshold = 0.4  # 非极大值抑制阈值

        face_objects = detector.detect(img, [], prob_threshold, nms_threshold)
        detector.draw(img, face_objects)  # 绘制人脸框

        output_path = result_dir / (Path(file_path).stem + ".png")
        cv2.imwrite(str(output_path), img)

        print(f"[INFO] Image saved to: {output_path}")

    elif is_video_file(file_path):
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return

        interval = 1
        standard = 60  # 设定的标准帧数量
        processed_frames = 0
        update_interval = 10  # 更新进度条

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_path = result_dir / (Path(file_path).stem + ".mp4")
        video = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'MP4V'),
                                cap.get(cv2.CAP_PROP_FPS), (frame_width, frame_height))

        num_clusters = 0
        prob_threshold = 0.3  # 人脸置信度阈值
        nms_threshold = 0.3  # 非极大值抑制阈值

        tracker = MultiObjectTracker()
        id_to_boxes = {}
        id_to_standard_box = {}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            det_frame_data = []

            if processed_frames <= standard:
                face_objects = detector.detect(frame, [], prob_threshold, nms_threshold)
                detector.draw(frame, face_objects)

                detected_boxes = [face.rect for face in face_objects]

                for i, box in enumerate(detected_boxes):
                    cur_box = {
                        'box': box,
                        'id': i,
                        'frame': processed_frames
                    }
                    det_frame_data.append(cur_box)

                tracking_results = tracker.update(det_frame_data)

                for it in tracking_results:
                    cv2.rectangle(frame, it['box'], (255, 0, 255), 2)
                    label = f"ID: {it['id']}"
                    cv2.putText(frame, label, (it['box'].x, it['box'].y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255))
                    id_to_boxes.setdefault(it['id'], []).append(it['box'])

                initial_face_boxes.append(face_objects)

                if processed_frames == standard:
                    for id, boxes in id_to_boxes.items():
                        num_clusters = 1  # 根据ID的检测框进行聚类
                        standard_box = cluster_faces_for_id(boxes, num_clusters)
                        id_to_standard_box[id] = standard_box
                    global clustering_done
                    clustering_done = True
            else:
                if (processed_frames - standard) % interval == 0:
                    detector.detect(frame, face_objects, prob_threshold, nms_threshold)
                    detector.draw(frame, face_objects)

                if clustering_done:
                    for id, standard_box in id_to_standard_box.items():
                        cv2.rectangle(frame, standard_box, (0, 255, 0), 2)
                        label = f"Standard ID: {id}"
                        cv2.putText(frame, label, (standard_box.x, standard_box.y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            video.write(frame)

            # 更新进度条
            processed_frames += 1
            if processed_frames % update_interval == 0:
                progress = processed_frames / total_frames
                print_progress_bar(progress)

        cap.release()
        video.release()

        print(f"[INFO] Video saved to: {output_path}")


def main(dir_path):
    param_path = "../assets/scrfd_10g-opt2.param"
    model_path = "../assets/scrfd_10g-opt2.bin"

    if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
        print(f"[ERROR] File does not exist: {dir_path}")
        return

    if not os.path.exists(param_path) or not os.path.exists(model_path):
        print("[ERROR] Model File or Param File does not exist")
        return

    result_dir = Path(dir_path) / "result"
    if not result_dir.exists():
        result_dir.mkdir()

    detector = SCRFD()
    detector.load(param_path, model_path, True)

    for entry in Path(dir_path).iterdir():
        if entry.is_file():
            process_file(entry, detector, result_dir)


if __name__ == "__main__":
    import sys
    # 打印当前工作目录
    print(os.getcwd())
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} [directory_path]")
    else:
        main(sys.argv[1])
