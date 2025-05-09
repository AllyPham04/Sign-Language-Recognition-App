import gradio as gr
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pandas as pd
import os
import glob

interpreter = tf.lite.Interpreter("model/model_sign_language.tflite")
interpreter.allocate_tensors()
# Get input and output tensors
found_signatures = list(interpreter.get_signature_list().keys())
print("Found signatures:", found_signatures)
prediction_fn = interpreter.get_signature_runner("serving_default")

ord2sign_df = pd.read_csv('data/ord2sign.csv')
ord2sign = ord2sign_df.set_index('sign_ord')['sign'].to_dict()

def get_sign_videos():
    video_files = glob.glob("data/videos/*.mp4")
    sign_videos = {}
    
    for video_file in video_files:
        filename = os.path.basename(video_file)
        parts = filename.split('_')
        if len(parts) >= 2:
            sign_ord = int(parts[0])
            sign_name = parts[1]
            if sign_ord in ord2sign:
                if sign_ord not in sign_videos:
                    sign_videos[sign_ord] = []
                sign_videos[sign_ord].append(video_file)
    
    return sign_videos

def get_video_for_sign(sign_ord):
    sign_videos = get_sign_videos()
    if sign_ord in sign_videos and sign_videos[sign_ord]:
        return sign_videos[sign_ord] # Return all videos for this sign
    return None

sign_names = [f"{ord2sign[ord]} ({ord})" for ord in sorted(ord2sign.keys())]

ROWS_PER_FRAME = 75

def load_relevant_data_subset(df):
    data_columns = ['x', 'y', 'z']
    data = df[data_columns]
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)

def create_frame_landmarks(results, frame, xyz):
    pose = pd.DataFrame()
    left_hand = pd.DataFrame()
    right_hand = pd.DataFrame()

    if results.pose_landmarks:
        for i, point in enumerate(results.pose_landmarks.landmark):
            pose.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]
    if results.left_hand_landmarks:
        for i, point in enumerate(results.left_hand_landmarks.landmark):
            left_hand.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]
    if results.right_hand_landmarks:
        for i, point in enumerate(results.right_hand_landmarks.landmark):
            right_hand.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]

    pose = pose.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='body')
    left_hand = left_hand.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='left_hand')
    right_hand = right_hand.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='right_hand')

    landmarks = pd.concat([pose, left_hand, right_hand]).reset_index(drop=True)
    landmarks = xyz.merge(landmarks, on=['type', 'landmark_index'], how='left')
    landmarks = landmarks.assign(frame=frame)
    return landmarks

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

def extract_landmarks_from_video(video_path):
    xyz = pd.DataFrame({
        'type': (['pose'] * 33) + (['left_hand'] * 21) + (['right_hand'] * 21),
        'landmark_index': list(range(33)) + list(range(21)) + list(range(21))
    })
    
    mp_holistic = mp.solutions.holistic
    xyz = pd.DataFrame({
        'type': (['pose'] * 33) + (['left_hand'] * 21) + (['right_hand'] * 21),
        'landmark_index': list(range(33)) + list(range(21)) + list(range(21))
    })
    all_landmarks = []
    cap = cv2.VideoCapture(video_path)
    with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:
        frame = 0
        while cap.isOpened():
            frame += 1
            success, image = cap.read()
            if not success:
                break
            image = cv2.flip(image, 1)
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            if not (results.pose_landmarks or results.left_hand_landmarks or results.right_hand_landmarks):
                continue
            landmarks = create_frame_landmarks(results, frame, xyz)
            if not landmarks.empty:
                all_landmarks.append(landmarks)
            
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            mp_drawing.draw_landmarks(
                image,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
            mp_drawing.draw_landmarks(
                image,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())

    cap.release()
    if all_landmarks:
        df = pd.concat(all_landmarks).reset_index(drop=True)
        return df
    else:
        return None

def process_video_to_csv(video):
    df = extract_landmarks_from_video(video)
    if df is None or len(df) < ROWS_PER_FRAME:
        return None, "No landmarks detected or video too short."
    csv_path = "temp_landmarks.csv"
    df.to_csv(csv_path, index=False)
    return csv_path, "Landmarks extracted and saved to CSV."

def predict_from_csv(csv_path):
    if not csv_path or not os.path.exists(csv_path):
        return "CSV file not found. Please process a video first."
    df = pd.read_csv(csv_path)
    xyz_np = load_relevant_data_subset(df)
    prediction = prediction_fn(inputs=xyz_np)
    outputs = prediction['outputs']
    pred = outputs.argmax()
    sign = ord2sign.get(pred, "Unknown")
    conf = outputs[pred]
    top_10_indices = outputs.argsort()[-10:][::-1]
    top_10_signs = [ord2sign.get(idx, "Unknown") for idx in top_10_indices]
    top_10_confidences = outputs[top_10_indices]
    result = f"Predicted sign: {sign} (Confidence: {conf:.4f})\n\nTop 10:\n"
    for s, c in zip(top_10_signs, top_10_confidences):
        result += f"{s}: {c:.4f}\n"
    return result

with gr.Blocks() as demo:
    gr.Markdown("# Sign Language Recognition\nUpload or record a video, then process to extract landmarks, then predict.")

    with gr.Tab("Video Tutorials"):
        with gr.Row():
            with gr.Column():
                sign_search = gr.Dropdown(
                    choices=sign_names,
                    label="Search for a sign to learn",
                    value=sign_names[0] if sign_names else None,
                    allow_custom_value=True,
                    filterable=True,
                    info="Type to search for a sign"
                )
                tutorial_videos = gr.Gallery(
                    label="Tutorial Videos",
                    show_label=True,
                    elem_id="gallery",
                    columns=[2],
                    rows=[2],
                    height="auto"
                )
                
                def update_tutorial_videos(sign_name):
                    if sign_name:
                        try:
                            sign_ord = int(sign_name.split("(")[1].strip(")"))
                            video_path = get_video_for_sign(sign_ord)
                            return video_path if video_path else None
                        except:
                            return None
                    return None
                
                sign_search.change(
                    fn=update_tutorial_videos,
                    inputs=sign_search,
                    outputs=tutorial_videos
                )

    with gr.Tab("Practice with Camera"):
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Upload or record a video")
                process_btn = gr.Button("Process Video (Extract Landmarks)")
            with gr.Column():
                csv_output = gr.Textbox(label="CSV Path", visible=False)
                process_status = gr.Textbox(label="Status")
                predict_btn = gr.Button("Predict")
                prediction_output = gr.Textbox(label="Prediction")

    process_btn.click(
        process_video_to_csv,
        inputs=video_input,
        outputs=[csv_output, process_status]
    )
    predict_btn.click(
        predict_from_csv,
        inputs=csv_output,
        outputs=prediction_output
    )

if __name__ == "__main__":
    demo.launch()