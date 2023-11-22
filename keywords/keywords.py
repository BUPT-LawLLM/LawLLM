from transformers import AutoTokenizer, AutoModel
from IPython.display import display, clear_output

def keywords_extract(prompt):
    
    model_path = "bupt/keywords_extract"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).quantize(4).half().cuda()
    model = model.eval()
    
    #response = model.stream_chat(tokenizer, prompt, history=[])

    for response, history in model.stream_chat(tokenizer, prompt, history=[]):
        clear_output(wait=True)
    
    return response

if __name__ == "__main__":
    prompt = "有人未经我同意用我的照片在网上传播违法言论，我该做什么，根据上述内容，给出相关法律关键词"
    output_text =  keywords_extract(prompt)
    print(output_text)
