import os
import gc
import cv2
import copy
import torch
import pandas as pd
import warnings
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

warnings.filterwarnings("ignore")


def infer_with_acebrain(vlm, processor, prompt, frame_paths, max_new_tokens=64):
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

    del inputs
    del generated_ids
    del generated_ids_trimmed
    torch.cuda.empty_cache()

    return output_text


def sample_video_frames(video_path, num_samples=16):
    video = cv2.VideoCapture(video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count <= 0:
        video.release()
        raise ValueError(f"Cannot read video or frame_count <= 0: {video_path}")

    num_samples = min(num_samples, frame_count)
    sample_indices = [int(i * frame_count / num_samples) for i in range(num_samples)]

    selected_frames = []
    for idx in sample_indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = video.read()
        if success and frame is not None:
            selected_frames.append(frame)

    video.release()

    if len(selected_frames) == 0:
        raise ValueError(f"No frames sampled from video: {video_path}")

    return selected_frames


if __name__ == "__main__":
    model = "ace-brain"
    model_path = "/root/autodl-tmp/ace_brain_0_8b"

    folder_path = "/root/autodl-tmp/urbanvideo_bench_dataset/videos"
    qa_path = "/root/autodl-tmp/urbanvideo_bench_dataset/MCQ.parquet"

    folder_path_result = "result"
    os.makedirs(folder_path_result, exist_ok=True)
    res_path = os.path.join(folder_path_result, f"{model}_output.csv")

    target_idx = 5343
    target_video = "RealWorld_49_1.mp4"

    print("Loading model...")
    vlm = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)
    print("Model loaded.")

    qa_df = pd.read_parquet(qa_path)

    if os.path.exists(res_path):
        res = pd.read_csv(res_path, index_col=0)
    else:
        res = copy.deepcopy(qa_df)
        res["Output"] = None

    if target_idx not in res.index:
        raise ValueError(f"target_idx {target_idx} not found in result dataframe index")

    actual_video = str(res.loc[target_idx, "video_id"])
    print(f"qa_idx={target_idx}, video_id={actual_video}")

    if actual_video != target_video:
        print(f"Warning: expected {target_video}, but dataframe has {actual_video}")

    video_path = os.path.join(folder_path, actual_video)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    prompt = (
        "This video (captured into multiple frames of images as follows) presents "
        "the perception data of an agent moving in the environment from a first "
        "person perspective. Please answer the following questions:\n"
    )
    prompt += (
        "The template for the answer is:\n"
        "Option: []; Reason: []\n"
        "where the Option only outputs one option from 'A' to 'E' here, "
        "do not output redundant content. Reason explains why you choose this option."
    )

    qa = str(res.loc[target_idx, "question"])
    prompt += "\n" + qa

    temp_frame_dir = "temp_frames_debug5315"
    os.makedirs(temp_frame_dir, exist_ok=True)

    frame_paths = []
    selected_frames = []

    try:
        print(f"Sampling frames from: {video_path}")
        selected_frames = sample_video_frames(video_path, num_samples=16)
        print(f"{len(selected_frames)} frames selected.")

        for i, frame in enumerate(selected_frames):
            frame_file = os.path.join(temp_frame_dir, f"qa_{target_idx}_{i}.jpg")
            ok = cv2.imwrite(frame_file, frame)
            if not ok:
                raise RuntimeError(f"Failed to save frame: {frame_file}")
            frame_paths.append(frame_file)

        print("Start inference...")
        res_str = infer_with_acebrain(
            vlm=vlm,
            processor=processor,
            prompt=prompt,
            frame_paths=frame_paths,
            max_new_tokens=64
        )

        print("Inference result:")
        print(res_str)

        res.loc[target_idx, "Output"] = res_str
        res.to_csv(res_path, index=True, encoding="utf-8-sig")
        print(f"Saved result to: {res_path}")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        for p in frame_paths:
            if os.path.exists(p):
                os.remove(p)

        if os.path.exists(temp_frame_dir) and len(os.listdir(temp_frame_dir)) == 0:
            os.rmdir(temp_frame_dir)

        del frame_paths
        del selected_frames
        gc.collect()
        torch.cuda.empty_cache()

        print("Cleanup done.")