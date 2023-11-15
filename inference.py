from evaluator import WenshuEvaluator, LawQAEvaluator
from retriever import Retriever
import argparse


def main():
    parser = argparse.ArgumentParser(description='BUPT-LawLLM')
    parser.add_argument('--fewshot_k', type=int, required=False, default=0, help='choose 0, 1 or 3 shot')
    parser.add_argument('--task', type=str, required=True, default="ws_result" , help='law_retrieve, law_qa or ws_result')
    parser.add_argument('--top_p', type=float, required=False, default=0.9, help='top_p value')
    parser.add_argument('--temperature', type=float, required=False, default=0.1, help='temperature value')
    parser.add_argument('--relevant_k', type=int, required=False, default=5, help='most relevant k retrieval')
    parser.add_argument('--finetuned', type=bool, required=False, default=False, help='using finetuned lora checkpoint')
    parser.add_argument('--source', type=str, required=False, default=None, help='local or web if needed')
    parser.add_argument('--verbose', type=bool, required=False, default=True, help='print out web search steps')

    args = parser.parse_args()

    if args.task == "ws_result":
        evaluator = WenshuEvaluator(k=args.fewshot_k, top_p=args.top_p, temperature=args.temperature, relevant_k=args.relevant_k, 
                                    finetuned=args.finetuned, source=args.source, verbose=args.verbose
                                )
        query = {
                    "Case": "高金辉、绍兴市越城区蓝精灵教育咨询中心教育培训合同纠纷一审民事判决书", 
                    "JudgeAccusation": "原告向本院提出诉讼请求：希望蓝精灵负责人黄可可择日偿还未上课时的剩余费用捌仟柒佰捌拾元整（8780元整）。事实与理由：原告于2017年11月29日在被告处给小孩买了69＋2节早教课，总费用9180元。之后因生活需要去了外地，上了三节课后就申请冻结了课程，后来被告因政府建设地铁原因停止了早教服务。得知消息后，原告联系了被告相关负责人，希望协调退还剩余费用，但其不愿解决，之后原告联系相关维权和监督部门依然无果，建议走司法途径。之后从监管局得知黄可可联系方式后与其进行了沟通，2019年8月22日黄可可承诺限期最迟十月底解决偿还剩余费用，但黄可可至今未有任何信息解决问题。\n被告未到庭答辩，亦未向本院提交书面答辩意见。\n经审理查明：2017年11月29日，原告为其小孩与被告签订课程销售协议，约定课程为69节，另赠送2节，费用为9180元。\n此后，被告因故停课。原告曾电话联系被告经营者协商退费事宜，但最终协商未果，故原告起诉要求退还学费。\n以上事实由原告提交的蓝精灵ＶＩＰ卡、售课协议照片打印件1份、光盘及录音文字整理材料1份予以证实。被告未向本院提交证据", 
                    "JudgeReason": "本院认为，依法成立的合同，对当事人具有法律约束力。当事人应当按照约定履行自己的义务。经本院查明，被告因故停课，现合同已无法继续履行，原告有权解除合同。合同解除后，被告应当退还剩余学费。根据原告庭审陈述，其小孩已上课3节，被告未到庭答辩，亦未举证证明实际上课次数多于3次，故本院结合原告提交的售课协议及上述陈述，对原告主张要求被告退还剩余课时费用8780元的诉讼请求，予以支持。被告经本院公告传唤，无正当理由未到庭参加诉讼，视为放弃抗辩权利，本院依法可作缺席判决。综上，依照《中华人民共和国合同法》第八条、第九十四条、第九十七条，《最高人民法院关于适用的解释》第九十条，《中华人民共和国民事诉讼法》第一百四十四条之规定，判决如下"
                }

        evaluator.preprocess_dataset()
        print(evaluator.run(query))

    elif args.task == "law_qa":
        evaluator = LawQAEvaluator(top_p=args.top_p, temperature=args.temperature, relevant_k=args.relevant_k, 
                                   finetuned=args.finetuned, source=args.source, verbose=args.verbose
                                )
        query = "醉酒撞了车之后被人打了应该算谁的？"
        print(evaluator.run(query))

    elif args.task == "law_retrieve":
        retriever = Retriever(relevant_k=args.relevant_k, source=args.source)
        query = "谁可以申请撤销监护人的监护资格?"
        print(retriever.query(query, verbose=args.verbose))

    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()