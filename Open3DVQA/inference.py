import json
import os
from collections import defaultdict
import glob
import gc

from tqdm import tqdm
from PIL import Image
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


def main():
    data_dir = "/root/autodl-tmp/Open3D-VQA.code/O3DVQA"
    result_dir = "/root/autodl-tmp/Open3D-VQA.code/response_result"

    qa_files = glob.glob(os.path.join(data_dir, "*", "*", "merged_qa.json"))
    if not qa_files:
        raise FileNotFoundError(f"No merged_qa.json found under {data_dir}")

    print("found qa files:", len(qa_files))
    for p in qa_files:
        print(p)

    model_path = "/root/autodl-tmp/ace_brain_0_8b"
    model_name = "ace-brain"
    result_path = os.path.join(result_dir, f"{model_name}_responses.json")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)

    responses = defaultdict(lambda: defaultdict(list))
    done_keys = set()

    if os.path.exists(result_path):
        print(f"Loading existing results from: {result_path}")
        with open(result_path, "r", encoding="utf-8") as f:
            resume_data = json.load(f)

        for dataset_name, scenes_dict in resume_data.items():
            for scene_name, scene_responses in scenes_dict.items():
                for response in scene_responses:
                    responses[dataset_name][scene_name].append(response)
                    item_id = response.get("id")
                    question = response.get("question", "")
                    done_keys.add((dataset_name, scene_name, item_id, question))

        print(f"Loaded {len(done_keys)} completed items.")
    else:
        print("No existing result file found, starting from scratch.")

    all_data = {}
    total_number = 0

    for qa_file in qa_files:
        dataset = os.path.basename(os.path.dirname(os.path.dirname(qa_file)))
        scene = os.path.basename(os.path.dirname(qa_file))

        with open(qa_file, "r", encoding="utf-8") as f:
            entries = json.load(f)

        all_data.setdefault(dataset, {})[scene] = entries
        total_number += len(entries)

    progress_bar = tqdm(total=total_number, desc="Processing")
    if done_keys:
        progress_bar.update(len(done_keys))

    for dataset, scenes in all_data.items():
        for scene, entries in scenes.items():
            for idx, item in enumerate(entries):
                item_id = item.get("id")
                question = item.get("query_question", "")
                item_key = (dataset, scene, item_id, question)

                if item_key in done_keys:
                    continue

                file_name = item.get("image_name")
                qa_info = item.get("qa_info", {})
                conversation = item.get("conversation", [])

                raw_image_path = item.get("image_info", {}).get("image_path", "")
                image_name = os.path.basename(raw_image_path.replace("\\", "/"))
                image_path = os.path.join(data_dir, dataset, scene, "rgb", image_name)

                if not os.path.exists(image_path):
                    print("image not found:", image_path)
                    progress_bar.update(1)
                    done_keys.add(item_key)
                    continue

                try:
                    image = Image.open(image_path).convert("RGB")
                except Exception as e:
                    print(f"Failed to open image file: {image_path}. Error: {e}")
                    progress_bar.update(1)
                    done_keys.add(item_key)
                    continue

                answer = ""
                if conversation and len(conversation) > 1:
                    answer = conversation[1].get("value", "")

                system_prompt = (
                    "You are an assistant who perfectly answers questions in urban environments. "
                    "Only based on the image, directly answer height, width, volume, and distance questions with exact numbers. "
                    "Answer distance questions without intermediate reasoning. "
                    "Answer direction questions using the clock direction format, taking your front as 12 o'clock, "
                    "your left as 9 o'clock, and your right as 3 o'clock."
                )

                full_prompt = system_prompt + "\n\n" + question

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": full_prompt}
                        ]
                    }
                ]

                max_retries = 5
                retry_count = 0
                response_text = ""

                while retry_count < max_retries:
                    try:
                        inputs = processor.apply_chat_template(
                            messages,
                            tokenize=True,
                            add_generation_prompt=True,
                            return_dict=True,
                            return_tensors="pt"
                        )
                        inputs = inputs.to(model.device)

                        with torch.no_grad():
                            generated_ids = model.generate(
                                **inputs,
                                max_new_tokens=32
                            )

                        generated_ids_trimmed = [
                            out_ids[len(in_ids):]
                            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                        ]

                        response_text = processor.batch_decode(
                            generated_ids_trimmed,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False
                        )[0]

                        del inputs, generated_ids, generated_ids_trimmed
                        gc.collect()
                        torch.cuda.empty_cache()
                        break

                    except Exception as e:
                        print(f"Error during local inference: {e}")
                        retry_count += 1
                        gc.collect()
                        torch.cuda.empty_cache()

                if retry_count == max_retries:
                    print("max retries exceeded, failed local inference.")
                    response_text = ""

                response = {
                    "id": item_id,
                    "image_name": file_name,
                    "qa_info": qa_info,
                    "question": question,
                    "answer": answer,
                    "response": response_text
                }

                responses[dataset][scene].append(response)
                done_keys.add(item_key)
                progress_bar.update(1)

                if len(done_keys) % 5 == 0:
                    os.makedirs(os.path.dirname(result_path), exist_ok=True)
                    with open(result_path, "w", encoding="utf-8") as f:
                        json.dump({k: dict(v) for k, v in responses.items()}, f, ensure_ascii=False, indent=4)

            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump({k: dict(v) for k, v in responses.items()}, f, ensure_ascii=False, indent=4)

    print(f"Done. Results saved to: {result_path}")


if __name__ == "__main__":
    main()