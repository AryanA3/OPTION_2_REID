from ultralytics import YOLO
import numpy as np
import cv2 as cv
from typing import List, Tuple
from torchreid.utils import FeatureExtractor



def print_matrix(arr) -> None:
    formatted = np.array2string(arr, precision=2, suppress_small=True, separator=', ')
    print(formatted)


def get_features(image, extractor: FeatureExtractor) -> np.ndarray:
    return extractor(image).cpu().numpy()


def get_cosine_mat(features: np.ndarray) -> List[List]:
    dot = np.matmul(features, features.T)
    mags = np.linalg.norm(features, axis=1)
    mags_mul = np.outer(mags, mags)

    return (dot / mags_mul).tolist()


def get_similar_images_for_each(matrix: List[List]) -> np.ndarray:

    sparse_mat = np.where(np.array(matrix) >= 0.83, 1, 0)
    sets = np.array([set() for _ in range(sparse_mat.shape[0])])

    for i in range(sparse_mat.shape[0]):
        for j in range(sparse_mat.shape[1]):
            if sparse_mat[i, j]:
                sets[i].add(j)
    
    return sets


def make_groups(sets:np.ndarray) -> List[set]:
    groups = [set()]
    for i in range(0, len(sets)):
            found_place = False
            for j in range(len(groups)):
                if len(groups[j]) == 0:
                    groups[j] = sets[i]
                    found_place = True
                    break
                
                if groups[j].intersection(sets[i]) == groups[j]:
                    found_place = True
                    break
            
            if not found_place:
                groups.append(sets[i])
    return groups



def find_most_similar_group_idx(data_base: dict, feature_vector: np.ndarray,thresh = 0.85) -> int:
    max_sim = 0.0
    max_key = 0
    for key, value in data_base.items():
        norm_1 = np.linalg.norm(value[0, :])
        norm_2 = np.linalg.norm(feature_vector)
        dot = np.dot(value[0, :].reshape((1, -1)), feature_vector.T)
        cos = dot / (norm_1 * norm_2)
        if cos >= thresh:
            if max(max_sim, cos) == cos:
                max_sim = max(max_sim, cos)
                max_key = key
    
    return max_key


def get_ids_for_images(data_base: dict, max_key: int, group_arr: np.ndarray, similarity_thresh = 0.93) -> List[int]:
    n = data_base[max_key][1:, :].shape[0]
    db_mags = np.linalg.norm(data_base[max_key][1:, :], axis=1, keepdims=True)
    group_arr_mags = np.linalg.norm(group_arr, axis=1, keepdims=True) 

    dot = np.dot(group_arr, data_base[max_key][1:, :].T)
    mags = np.matmul(group_arr_mags, db_mags.T)

    sparse_index_mat = np.where((dot / mags) >= similarity_thresh, 1, 0)

    query_ids = [-1] * sparse_index_mat.shape[0]
    for row in range(sparse_index_mat.shape[0]):
        for col in range(sparse_index_mat.shape[1]):
            if sparse_index_mat[row, col] and col not in query_ids:
                query_ids[row] = col + 1
                break
    
    for idx in range(sparse_index_mat.shape[0]):
        if query_ids[idx] == -1 and n < 12:
            query_ids[idx] = n + 1
            data_base[max_key] = np.vstack((data_base[max_key], group_arr[idx]))
            data_base[max_key][0, :] = (data_base[max_key][0, :] * n + group_arr[idx]) / (n + 1)
            n += 1

    
    return query_ids



def update_data_base(data_base: dict, features: np.ndarray, groups: List[set], fresh = False) -> List[List[Tuple[int, int]]]:
    i = 0
    ids = []
    if fresh:
        ids = [list(zip(list(groups[i]), range(1, len(groups[i]) + 1))) for i in range(len(groups))]

    for group in groups:
        group = list(group)
        idx = group[0]
        group_arr = features[idx, :]
        for idx in group[1:]:
            group_arr = np.vstack((group_arr, features[idx, :]))
        
        if group_arr.ndim != 2:
            group_arr = group_arr.reshape((1 , -1))

        if fresh:
            data_base[i] = np.vstack((np.sum(group_arr, axis = 0, keepdims=True) / group_arr.shape[0], group_arr))
            i += 1
            continue
        
        else:
            feature_vector = np.sum(group_arr, axis = 0, keepdims=True) / group_arr.shape[0]
            max_key = find_most_similar_group_idx(data_base, feature_vector)
            print(max_key)
            data_base[max_key][0, :] += feature_vector.flatten()
            new_ids_group_arr_idx = get_ids_for_images(data_base, max_key, group_arr)
            ids.append(list(zip(group, new_ids_group_arr_idx)))
    
    return ids
            
    

def draw_boxes(frame, detections: List, data_base: dict, extractor: FeatureExtractor):
    images = np.array([])
    cords = []
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        cls_id = int(cls_id)

        if cls_id != 2 or (x2 - x1 > 40 and y2 - y1 > 80):
            continue

        # Convert to int
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        image = frame[y1:y2, x1:x2]

        if images.size == 0:
            images = get_features(image, extractor)
            cords.append(((x1, y1), (x2, y2)))
            continue

        feature = get_features(image, extractor)
        images = np.vstack([images, feature])
        cords.append(((x1, y1), (x2, y2)))
    
    if images.size == 0:
        return frame

    matrix = get_cosine_mat(images)
    sets = get_similar_images_for_each(matrix)
    groups_test = make_groups(sets)
    groups = update_data_base(data_base, images, groups_test, True if not data_base else False)

    for group in groups:
        for idx, id in group:

            color = (0, 255, 0)  # Green box
            cv.rectangle(frame, cords[idx][0], cords[idx][1], color, 2)

            # Draw label
            label = f"ID = {id}"
            frame = cv.putText(frame, label, (cords[idx][0][0], cords[idx][0][1] - 5), cv.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)
        

    return frame



def main() -> None:

    extractor = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path='./REID_ASSIGNMENT_MODELS_VIDEOS/osnet_x1_0.pth',
        device='cuda'
    )

    data_base = {}


    cap = cv.VideoCapture("./REID_ASSIGNMENT_MODELS_VIDEOS/15_sec.mp4")
    model = YOLO("./REID_ASSIGNMENT_MODELS_VIDEOS/best.pt")
    model.to("cuda:0")
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    if not cap.isOpened():
        print("cant open")
        exit(1)

    fps = cap.get(cv.CAP_PROP_FPS)
    delay = int(1000/fps) if fps > 0 else 1

    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    out = cv.VideoWriter("output.mp4", fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, conf = 0.91)

        annotated_frame = draw_boxes(
        frame,
        results[0].boxes.data.cpu().numpy(),
        data_base, extractor)

        if annotated_frame.dtype != np.uint8:
            annotated_frame = annotated_frame.astype(np.uint8)

        out.write(annotated_frame)
        
        if cv.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv.destroyAllWindows()



if __name__ == "__main__":
    import gdown
    import zipfile
    import os

    zip_link = 'https://drive.google.com/file/d/1dlndXhmElWX-q3SjsNrLKW9oOio0_9zK/view?usp=drive_link'

    file_id = zip_link.split('/d/')[1].split('/')[0]

    download_url = f'https://drive.google.com/uc?id={file_id}'

    output_path = os.getcwd() + "/folder.zip"
    gdown.download(download_url, output_path, quiet=False)


    with zipfile.ZipFile(output_path, 'r') as zip_ref:
        zip_ref.extractall(os.getcwd())


    os.remove(output_path)
    main()