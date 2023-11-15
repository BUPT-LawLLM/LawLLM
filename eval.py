from evaluator import WenshuEvaluator, LawQAEvaluator
import argparse


def main():
    parser = argparse.ArgumentParser(description='BUPT-LawLLM')
    parser.add_argument('--fewshot_k', type=int, required=False, default=0, help='choose 0, 1 or 3 shot')
    parser.add_argument('--task', type=str, required=True, default="ws_result" , help='law_qa or ws_result')
    parser.add_argument('--top_p', type=float, required=False, default=0.9, help='top_p value')
    parser.add_argument('--temperature', type=float, required=False, default=0.1, help='temperature value')
    parser.add_argument('--relevant_k', type=int, required=False, default=5 , help='most relevant k retrieval')
    parser.add_argument('--finetuned', type=bool, required=False, default=False, help='using finetuned lora checkpoint')
    parser.add_argument('--source', type=str, required=False, default=None, help='local or web if needed')
    parser.add_argument('--verbose', type=bool, required=False, default=True, help='print out web search steps')
    
    args = parser.parse_args()

    if args.task == "ws_result":
        evaluator = WenshuEvaluator(k=args.fewshot_k, top_p=args.top_p, temperature=args.temperature, relevant_k=args.relevant_k, 
                                    finetuned=args.finetuned, source=args.source, verbose=args.verbose
                                )
    elif args.task == "law_qa":
        evaluator = LawQAEvaluator(top_p=args.top_p, temperature=args.temperature, relevant_k=args.relevant_k, 
                                   finetuned=args.finetuned, source=args.source, verbose=args.verbose
                                ) # TODO     
    else:
        raise NotImplementedError

    evaluator.preprocess_dataset()
    evaluator.run_dataset()


if __name__ == "__main__":
    main()