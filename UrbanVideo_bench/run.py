import cv2
import base64
import time
import os
import pandas as pd
import warnings
import math
import copy
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

warnings.filterwarnings("ignore")

def infer_with_acebrain(vlm, processor, prompt, frame_paths, max_new_tokens=256):
    messages = [
        {
            "role": "user",
            "content": (
                [{"type": "image", "image": img_path} for img_path in frame_paths]
                + [{"type": "text", "text": prompt}]
            ),
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(vlm.device)

    with torch.no_grad():
        generated_ids = vlm.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return output_text


# Main execution block
if __name__ == '__main__':
    # Need to input
    # Define the model name and initialize the OpenAI client with API credentials.
    model = "ace-brain"
    model_path = "/root/autodl-tmp/ace_brain_0_8b"

    vlm = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)

    # Dataset path
    folder_path = '/root/autodl-tmp/urbanvideo_bench_dataset/videos'  # Define the folder path where video files are stored.
    QA_df = pd.read_parquet('/root/autodl-tmp/urbanvideo_bench_dataset/MCQ.parquet')  # Read the dataset containing questions and metadata from a Parquet file.

    # Define the folder path for saving results and create it if it doesn't exist.
    folder_path_result = 'result'
    if not os.path.exists(folder_path_result):
        os.makedirs(folder_path_result)

    # Define the path for the result CSV file.
    res_path = os.path.join(folder_path_result, '%s_output.csv' % model)

    # Check if the result file already exists.
    if os.path.exists(res_path):
        # If the file exists, load it and find the last valid index in the 'Output' column.
        res = pd.read_csv(res_path, index_col=0)
        last_valid_index = int(res['Output'].last_valid_index())
        last_valid_index += 1  # Start processing from the next index.
    else:
        # If the file doesn't exist, create a new DataFrame based on the QA dataset.
        res = copy.deepcopy(QA_df)
        res['Output'] = None  # Add an 'Output' column initialized to None.
        last_valid_index = 0  # Start processing from the first index.

    # Iterate through each question starting from the last valid index.
    for qa_idx in range(last_valid_index, res.shape[0]):
        print('Processing index: %d' % qa_idx)

        # Get the video ID for the current question.
        select_vid_name = res['video_id'].iloc[qa_idx]

        # Open the video file using OpenCV.
        video = cv2.VideoCapture(os.path.join(folder_path, str(select_vid_name)))
        video_fps = video.get(cv2.CAP_PROP_FPS)  # Get the frames per second (FPS) of the video.

        # Initialize a list to store base64-encoded frames.
        all_frames = []
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
            all_frames.append(frame)

        # Release the video file and print the number of frames read.
        video.release()
        print(len(all_frames), "frames read.")

        # Create a prompt for the GPT model to answer questions based on the video.
        prompt = "This video (captured into multiple frames of images as follows) presents the perception data of an agent moving in the environment from a first person perspective. Please answer the following questions: \n"
        prompt += "The template for the answer is: \n\
                        Option: []; Reason: []\n\
                        where the Option only outputs one option from 'A' to 'E' here, do not output redundant content. Reason explains why you choose this option."

        # Add the question from the dataset to the prompt.
        qa = res['question'].iloc[qa_idx]
        prompt += '\n' + qa

        try:
            div_num = max(1, math.ceil(len(all_frames) / 32))
            selected_frames = all_frames[0::div_num][:32]

            temp_frame_dir = "temp_frames"
            os.makedirs(temp_frame_dir, exist_ok=True)

            frame_paths = []
            for i, frame in enumerate(selected_frames):
                frame_file = os.path.join(temp_frame_dir, f"qa_{qa_idx}_{i}.jpg")
                cv2.imwrite(frame_file, frame)
                frame_paths.append(frame_file)

            res_str = infer_with_acebrain(
                vlm=vlm,
                processor=processor,
                prompt=prompt,
                frame_paths=frame_paths,
                max_new_tokens=256
            )

            print(res_str)

            # Save the response in the 'Output' column of the result DataFrame.
            res.loc[qa_idx, 'Output'] = res_str

        except Exception as e:
            # Handle errors and wait for 60 seconds before retrying.
            print(f"An error occurred: {e}")
            time.sleep(60)

        # Save the updated result DataFrame to the CSV file.
        res.to_csv(res_path, index=True, encoding='utf-8-sig')
