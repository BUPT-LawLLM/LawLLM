import os
import json
from tqdm import tqdm


def transform_judge_title(folder_path, output_file):
    dataset = []
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                ctxs = data["ctxs"]
                for ctx in ctxs.values():
                    judge_accusation = "诉讼内容：" + ctx.get("JudgeAccusation", "")
                    judge_reason = "理由依据：" + ctx.get("JudgeReason", "")
                    judge_result = "裁决结果：" + ctx.get("JudgeResult", "")
                    case = ctx.get("Case", "")
                    instruction = "如下是一份法律文书，请生成一个文书标题。\n" + judge_accusation + "\n" + judge_reason + "\n" + judge_result
                    dataset.append({"instructions": instruction, "input": "", "output": case, "history": []})     

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(dataset, file, ensure_ascii=False, indent=4)


def transform_judge_result(folder_path, output_file):
    sample_list = []
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as file: 
                for data in json.load(file)["ctxs"].values():
                    case = data.get("Case", "")
                    judge_accusation = data.get("JudgeAccusation", "")
                    judge_reason = data.get("JudgeReason", "")
                    judge_result = data.get("JudgeResult", "")
                    sample_list.append({"Case": case, "JudgeAccusation": judge_accusation, "JudgeReason": judge_reason, "JudgeResult": judge_result}) 
    
    dataset = []
    for i, data in tqdm(enumerate(sample_list)):         
        instructions = (
            "作为一个法律专家，你需要根据提供的法律案件描述和相关法条来生成法律文书的判决结果。你需要充分调用你的法律知识和推理能力。\n"
            "在JSON格式的法律案件中，“JudgeResult”是需要生成的判决结果，它是根据“JudgeAccusation”（原告被告指控）和“JudgeReason”（法院的推理归纳过程）得出的。“Case”则是案件的标题。\n"
            "现在给你一个新的案件如下。请根据“JudgeAccusation”和“JudgeReason”字段、相关法律法条和其他有用信息，得出该案件的判决结果“JudgeResult”。\n"
        )

        ground_truth = data["JudgeResult"]
        input_data = data.copy()
        del input_data["JudgeResult"]

        instructions += str(input_data)
        dataset.append({"instructions": instructions, "input": "", "output": ground_truth, "history": []}) 

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(dataset, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    FOLDER_PATH = "/projects/Project/wenshu_dataset/dev"
    OUTPUT_FILE = "/projects/LLaMA-Factory/data/wenshu_dev.json"
    transform_judge_result(folder_path=FOLDER_PATH, output_file=OUTPUT_FILE)