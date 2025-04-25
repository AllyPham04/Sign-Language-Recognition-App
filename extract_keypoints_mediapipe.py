import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

path = 'D:\\Admin\\ML-Ops\\Video' #thay cái path dẫn tới cái folder chứa video vô đây nha

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_pose_keypoints(results):
    if results.pose_landmarks:
        return np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
    else:
        return np.zeros(33 * 3)

def extract_hand_keypoints(results):
    left = np.zeros(21 * 3)
    right = np.zeros(21 * 3)
    
    if results.multi_handedness and results.multi_hand_landmarks:
        for idx, hand_info in enumerate(results.multi_handedness):
            label = hand_info.classification[0].label
            landmarks = results.multi_hand_landmarks[idx]
            keypoints = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()
            if label == 'Left':
                left = keypoints
            else:
                right = keypoints
    
    return left, right


def process_video(source='webcam', show=True):
    cap = cv2.VideoCapture(0 if source == 'webcam' else source)
    keypoints_list = []

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            pose_results = pose.process(image)
            hands_results = hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if show and source == 'webcam':
                mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                if hands_results.multi_hand_landmarks:
                    for hand_landmarks in hands_results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.imshow('Pose + Hands', image)

            pose_kp = extract_pose_keypoints(pose_results)
            lh_kp, rh_kp = extract_hand_keypoints(hands_results)
            all_kp = np.concatenate([pose_kp, lh_kp, rh_kp])
            keypoints_list.append(all_kp)

            if source == 'webcam' and (cv2.waitKey(1) & 0xFF == ord('q')):
                break

    cap.release()
    if show and source == 'webcam':
        cv2.destroyAllWindows() #destroy (close) all windows that appear in the time of running code
    return keypoints_list

def merge_keypoints(keypoints_list):
    return np.array(keypoints_list)

def extract_keypoints_from_video(path, start_index, end_index):
    all_files = os.listdir(path)
    keypoints_and_id = pd.DataFrame(columns=['video_id', 'keypoints'])

    os.chdir(path)
    
    for file in all_files[start_index:end_index]: #lấy từ file nào đến file nào thì tự sửa vào đây
        video_id = file.split('.')[0]
        print(f'Extracting keypoints from video {file}')
        keypoints = process_video(file)
        merged = merge_keypoints(keypoints)
        print("Shape of merged keypoints:", merged.shape)  # (num_frames, total_features)
        keypoints_and_id.loc[len(keypoints_and_id)] = [video_id, keypoints]
    return keypoints_and_id
