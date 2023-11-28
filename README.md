# LawLLM ⚖️
该仓库是 BUPT 智能系统实验室的法律大模型项目。如果你对我们的工作感兴趣，请联系我或发 Issue 交流，十分感谢🙏！

### QuickStart
尝试 web demo 可以运行
```streamlit run web_demo.py --server.address 0.0.0.0```

如果要在评估数据集上测试分数
```python eval.py --task ws_result --finetuned True```

如果要测试法条向量库检索
```python inference.py --task law_retrieve --relevant_k 7 ```

如果要试试法条增强的法律问答
```python inference.py --task law_qa --finetuned False --source web```

其余参数可以依据脚本内容进行修改。


### Resources
全部数据集都已经开源到 BUPT 对应的 HuggingFace 中，如下是链接🔗。

https://huggingface.co/datasets/bupt/LawDataset-BUPT

使用到的针对文书生成任务的 ChatGLM3-32K 微调模型也已经开源到 HuggingFace，如下是链接🔗。

https://huggingface.co/bupt/chatglm3-6b-32k-wenshu-finetuned

法律法条数据向量库基于开源项目 tigerbot 制作，如下是链接🔗。

https://huggingface.co/datasets/Jinsns/tiger_laws

### Requirements

Run `pip install -r requirements.txt` in Llama-Factory for further finetuning.

Install `langchain`, `sentence-transformers`, `faiss-cpu`, `gradio` and `streamlit` using pip.


### Acknowledgements
我们由衷感谢这些开源项目
- ChatGLM3
- Baichuan2
- Llama-Chinese
- Llama-Factory
- DISC-LawLLM

<!-- readme: contributors -start -->
<!-- readme: contributors -end -->
