import os
import json
import yaml
import time
import random
import numpy as np
from tqdm import tqdm
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModel
from typing import Dict, Tuple, Union, Optional

from metric import calculate_chinese_rouge_scores
from retriever import Retriever


class WenshuEvaluator:
    def __init__(self, k=3, top_p=0.9, temperature=0.1, relevant_k=5, finetuned=True, source=None, verbose=True):

        with open("config.yaml", 'r') as stream:
            config = yaml.safe_load(stream)
        self.dataset_dir = config["WENSHU_DATASET_DIR"]
        self.model_dir = config["WENSHU_MODEL_DIR"]
        self.checkpoint_path = config["WENSHU_FT_CHECKPOINT_PATH"]

        self.k = k
        self.top_p = top_p
        self.temperature = temperature
        self.finetuned = finetuned
        self.source = source
        self.verbose = verbose

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.tokenizer.padding_side = "right"  # Fix for fp16
        self.model = AutoModel.from_pretrained(self.model_dir, trust_remote_code=True, device='cuda')
        
        if self.finetuned:
            print("Using LoRA finetuned.")
            # self.peftconfig = PeftConfig.from_pretrained(self.checkpoint_path)
            self.model = PeftModel.from_pretrained(self.model, self.checkpoint_path) 

        self.model = self.model.eval()
        self.retriever = Retriever(relevant_k=relevant_k, source=self.source)

        self.dataset = []
        self.rouge_1_score = []
        self.rouge_2_score = []
        self.rouge_l_score = []

        self.law_prompt = "已知如下法律条文的具体内容：\n"

        if self.k == 0:
            self.pre_prompt = (
                "作为一个法律专家，你需要根据提供的法律案件描述和相关法条来生成法律文书的判决结果。你需要充分调用你的法律知识和推理能力。\n"
                "在JSON格式的法律案件中，“JudgeResult”是需要生成的判决结果，它是根据“JudgeAccusation”（原告被告指控）和“JudgeReason”（法院的推理归纳过程）得出的。“Case”则是案件的标题。\n"
            )
            self.post_prompt = (
                "\n现在给你一个新的案件如下。请根据“JudgeAccusation”和“JudgeReason”字段、相关法律法条和其他有用信息，得出该案件的判决结果“JudgeResult”。\n"
            )
        else:
            self.pre_prompt = (
                "作为一个法律专家，你需要根据提供的法律案件描述和相关法条来生成法律文书的判决结果。你需要充分调用你的法律知识和推理能力。\n"
                f"如下是{self.k}个法律文书生成的例子，其中“JudgeResult”就是需要生成的判决结果，它是根据“JudgeAccusation”和“JudgeReason”得出的。\n"
            )
            self.post_prompt = (
                "\n现在给你一个新的案件如下。请根据“JudgeAccusation”和“JudgeReason”字段、相关法律法条和其他有用信息，得出该案件的判决结果“JudgeResult”。\n"
            )

    def preprocess_dataset(self):
        for filename in tqdm(os.listdir(self.dataset_dir)):
            if filename.endswith(".json"):
                file_path = os.path.join(self.dataset_dir, filename)
                with open(file_path, "r", encoding="utf-8") as file: 
                    for data in json.load(file)["ctxs"].values():
                        case = data.get("Case", "")
                        judge_accusation = data.get("JudgeAccusation", "")
                        judge_reason = data.get("JudgeReason", "")
                        judge_result = data.get("JudgeResult", "")
                        self.dataset.append({"Case": case, "JudgeAccusation": judge_accusation, "JudgeReason": judge_reason, "JudgeResult": judge_result}) 

        print("DATASET_PATH:", self.dataset_dir, "DATASET_LENGTH:", len(self.dataset))

    def model_generate(self, prompt):
        if self.finetuned:
            input_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
            # with torch.inference_mode():
            outputs = self.model.generate(input_ids=input_ids, max_new_tokens=250, do_sample=True, top_p=self.top_p, temperature=self.temperature)
            output = self.tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
            output = output.replace("[gMASK]sop", "")
            output = output[len(prompt)+1:]
        else:
            output, _ = self.model.chat(self.tokenizer, prompt, history=[], top_p=self.top_p, temperature=self.temperature)
            if output.startswith("'JudgeResult': "):
                output = output[len("'JudgeResult': "):]
        
        return output
        
    def run(self, query: Dict[str, int]):
        prompt = self.pre_prompt

        if self.k > 0:
            fewshot_samples = random.sample(self.dataset, self.k)
            for fewshot_sample in fewshot_samples:
                prompt += str(fewshot_sample) + "\n" 

        if self.source:
            judge_reason = query["JudgeReason"]
            law_result = self.retriever.query(f"如下是一篇文书中的一段审判：\n{judge_reason}\n请按照顺序，逐一输出其中提到的相关法律条文的具体内容，并用中文总结。\n", verbose=self.verbose)
            prompt += (self.law_prompt + law_result + "\n")

        prompt += self.post_prompt
        prompt += str(query)
        output = self.model_generate(prompt)
        return output

    def run_dataset(self):
        for i, data in tqdm(enumerate(self.dataset)):  
            try:       
                prompt = self.pre_prompt
                if self.k > 0:
                    fewshot_samples = random.sample(self.dataset[:i] + self.dataset[i+1:], self.k)
                    for fewshot_sample in fewshot_samples:
                        prompt += str(fewshot_sample) + "\n" 

                if self.source:
                    judge_reason = data["JudgeReason"]
                    law_result = self.retriever.query(f"如下是一篇文书中的一段审判：\n{judge_reason}\n请按照顺序，逐一搜索其中提到的相关法律条文的具体内容，并用中文总结。\n", verbose=self.verbose)
                    prompt += (self.law_prompt + law_result + "\n")

                prompt += self.post_prompt

                ground_truth = data["JudgeResult"]
                input_data = data.copy()
                del input_data["JudgeResult"]

                prompt += str(input_data)
                output = self.model_generate(prompt)
                scores = calculate_chinese_rouge_scores(output, ground_truth)

                print(f"\n\n------DATA_ID_{i}------")
                print("INPUT:", prompt)
                print("OUTPUT:", output)
                print("GT:", ground_truth)
                
                self.rouge_1_score.append(scores['rouge-1']['f'])
                self.rouge_2_score.append(scores['rouge-2']['f'])
                self.rouge_l_score.append(scores['rouge-l']['f'])

                print("AVG ROUGE-1 f1:", np.mean(self.rouge_1_score))
                print("AVG ROUGE-2 f1:", np.mean(self.rouge_2_score))
                print("AVG ROUGE-l f1:", np.mean(self.rouge_l_score))

                time.sleep(1)

            except Exception as e:
                print(e)
                continue

        print(f"{self.k}-shot EVAL FINISHED.")


class LawQAEvaluator:
    def __init__(self, top_p=0.9, temperature=0.1, relevant_k=5, finetuned=False, source="local", verbose=True):

        with open("config.yaml", 'r') as stream:
            config = yaml.safe_load(stream)
        self.dataset_dir = config["QA_DATASET_DIR"] # TODO
        self.model_dir = config["QA_MODEL_DIR"]
        self.checkpoint_path = config["QA_FT_CHECKPOINT_PATH"]

        self.top_p = top_p
        self.temperature = temperature
        self.finetuned = finetuned
        self.source = source
        self.verbose = verbose

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.model_dir, trust_remote_code=True, device='cuda').to(f"cuda:{cuda}")
         
        if self.finetuned: # TODO
            # self.peftconfig = PeftConfig.from_pretrained(self.checkpoint_path)
            self.model = PeftModel.from_pretrained(self.model, self.checkpoint_path)

        self.model = self.model.eval()

        self.retriever = Retriever(relevant_k=relevant_k, source=self.source)
        self.qa_template = (
            "你是一个专业的法律顾问，你的回答要求逻辑完善，有理有据，不允许伪造事实。现在有一个法律咨询问题: {question}\n"
            "为了回答这个问题，我们检索到相关法条如下：\n"
            "{retrieved_laws}\n"
            "\n根据以上法条和你的推理，你的回答是：让我们一步一步思考。"
        )

    def preprocess_dataset(self):
        pass

    def model_generate(self, prompt):
        if self.finetuned:
            input_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
            # with torch.inference_mode():
            outputs = self.model.generate(input_ids=input_ids, max_new_tokens=250, do_sample=True, top_p=self.top_p, temperature=self.temperature)
            output = self.tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
            output = output.replace("[gMASK]sop", "")
            output = output[len(prompt):]
        else:
            output, _ = self.model.chat(self.tokenizer, prompt, history=[], top_p=self.top_p, temperature=self.temperature)
            if output.startswith("'JudgeResult': "):
                output = output[len("'JudgeResult': "):]
        
        return output
        
    def run(self, query: str):
        law_result = self.retriever.query(query, verbose=self.verbose)
        prompt = self.qa_template.format(question=query, retrieved_laws=law_result)
        output = self.model_generate(prompt)
        return output

    def run_dataset(self):
        pass
        

if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    # evaluator = WenshuEvaluator(k=0, temperature=0.1, finetuned=False, source=None, verbose=False)
    # evaluator.preprocess_dataset()
    # evaluator.run_dataset()
    # query = {
    #          "Case": "高金辉、绍兴市越城区蓝精灵教育咨询中心教育培训合同纠纷一审民事判决书", 
    #          "JudgeAccusation": "原告向本院提出诉讼请求：希望蓝精灵负责人黄可可择日偿还未上课时的剩余费用捌仟柒佰捌拾元整（8780元整）。事实与理由：原告于2017年11月29日在被告处给小孩买了69＋2节早教课，总费用9180元。之后因生活需要去了外地，上了三节课后就申请冻结了课程，后来被告因政府建设地铁原因停止了早教服务。得知消息后，原告联系了被告相关负责人，希望协调退还剩余费用，但其不愿解决，之后原告联系相关维权和监督部门依然无果，建议走司法途径。之后从监管局得知黄可可联系方式后与其进行了沟通，2019年8月22日黄可可承诺限期最迟十月底解决偿还剩余费用，但黄可可至今未有任何信息解决问题。\n被告未到庭答辩，亦未向本院提交书面答辩意见。\n经审理查明：2017年11月29日，原告为其小孩与被告签订课程销售协议，约定课程为69节，另赠送2节，费用为9180元。\n此后，被告因故停课。原告曾电话联系被告经营者协商退费事宜，但最终协商未果，故原告起诉要求退还学费。\n以上事实由原告提交的蓝精灵ＶＩＰ卡、售课协议照片打印件1份、光盘及录音文字整理材料1份予以证实。被告未向本院提交证据", 
    #          "JudgeReason": "本院认为，依法成立的合同，对当事人具有法律约束力。当事人应当按照约定履行自己的义务。经本院查明，被告因故停课，现合同已无法继续履行，原告有权解除合同。合同解除后，被告应当退还剩余学费。根据原告庭审陈述，其小孩已上课3节，被告未到庭答辩，亦未举证证明实际上课次数多于3次，故本院结合原告提交的售课协议及上述陈述，对原告主张要求被告退还剩余课时费用8780元的诉讼请求，予以支持。被告经本院公告传唤，无正当理由未到庭参加诉讼，视为放弃抗辩权利，本院依法可作缺席判决。综上，依照《中华人民共和国合同法》第八条、第九十四条、第九十七条，《最高人民法院关于适用的解释》第九十条，《中华人民共和国民事诉讼法》第一百四十四条之规定，判决如下"
    #         }
    # ans = evaluator.run(query)
    # print(ans)

    # evaluator = LawQAEvaluator(finetuned=False, source="local", verbose=False)
    # ans = evaluator.run("醉酒撞了车之后被人打了应该算谁的？")
    # print(ans)
    pass