#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/23 17:03
# @File    : extraction.py
# @Description : 不含文本分类，直接手动输入各类文本。函数extraction是用来做法律关键信息抽取的,输入为input的两端文本或直接修改成读取两端文本，返回模型生成值responses
# !/usr/bin/env python3

import re
import json
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from rich import print
from rich.console import Console
from transformers import AutoTokenizer, AutoModel

def extraction():
    
    IE_PATTERN = "{}\n\n提取上述句子中能够描述{}的实体，实体属性应该包括:{}，并将按照如同ie_examples一样的JSON格式输出，上述句子中不存在的信息用['原文未提及']来表示，多个值之间用','分隔。"


    # 定义不同类型下具备的属性
    schema = {
        '案发过程': ['原告涉事企业', '原告涉事人员', '原告委托代理人', '被告涉事企业', '被告涉事人员', '被告委托代理人', '事发地点', '涉事金额', '案件规模', '原告主张','被告主张'],
        '处理结果': ['法院判决结果', '罪名判决', '处罚']
    }

    # 提供一些例子供模型参考
    ie_examples = {
            '案发过程': [
                    {
                        'content': '再审申请人（一审原告、二审上诉人）：山东隆坤房地产开发有限公司。住所地：山东省济南市经十路17079号汇文轩东座17层。法定代表人：于光远，该公司董事长。委托代理人：刘丕岐，山东舜天律师事务所律师。委托代理人：李红英，北京大成（青岛）律师事务所律师。被申请人（一审被告、二审被上诉人）：山东省机床维修站。住所地：山东省济南市工业北路285号。法定代表人：张作让，该站站长。委托代理人：王勇，山东诚信人律师事务所律师。被申请人（一审被告、二审被上诉人）：济南引力有限公司。住所地：山东省济南市工业北路285号。法定代表人：张作让，该公司总经理。委托代理人：王勇，山东诚信人律师事务所律师。被申请人（一审被告、二审被上诉人）：济南鲁联集团投资有限公司。住所地：山东省济南市长清区经十西路11889号。法定代表人：薛鹏雁，该公司总经理。委托代理人：郭士旺，该公司经理。被申请人（一审被告、二审被上诉人）：济南鲁联集团有限公司。住所地：山东省济南市长清经十西路11889号。法定代表人：刘津福，该公司董事长、总经理。委托代理人：苏志高，该公司职工。再审申请人孙卫良因与被申请人陶玲燕、周益民股权转让纠纷一案，不服浙江省高级人民法院（2014）浙商外终字第75号民事判决，向本院申请再审。本院依法组成合议庭对本案进行了审查，现已审查终结。孙卫良向本院申请再审称：案涉《股权转让和资产收购（框架）协议》（以下简称《框架协议》）签订时，该协议的当事人海盐新艺建筑装饰材料厂（以下简称新艺建材厂）已经被注销。新艺建材厂被注销后，其主体已经消亡，却仍然从事经营活动。新艺建材厂签订的《框架协议》应当认定无效，一、二审判决对此适用法律错误。依据《中华人民共和国民事诉讼法》第二百条第（六）项的规定，请求对本案进行再审。周益民、陶玲燕提交意见称：案涉《框架协议》合法有效。新艺建材厂作为个人独资企业被注销后，其投资人周益民可以进行股权转让。新艺建材厂在注销后的合理期限内对外处理投资事务不属于经营活动。《框架协议》的主体瑕疵不构成合同无效的事由。孙卫良的根本性违约行为导致案涉协议无法履行而解除。一、二审判决认定事实清楚，适用法律正确，请求驳回孙卫良的再审申请。本院认为，本案争议焦点为，盈吉建筑公司迟延给付的432742.5元货款是否应当支付违约金。2011年3月25日，伟业混凝土公司与盈吉建筑公司分别签订《商品混凝土购销合同》，该合同第四条约定，供方为需方供应商砼，每供应5000m3混凝土结算一次，需方需在十日内支付前面所供5000m3混凝土款的80%，供方再为需方供应商砼，以此类推，直至工程主体封顶，每年12月31日前需方需付到所供混凝土总款的90%，余款在混凝土工程结束后一个月内全部付清。根据上述约定，盈吉建筑公司付款的前提条件是结算，在未结算前，不具备付款条件。经查，双方于2014年6月20日、6月23日进行结算对账，对所有项目中供货量进行了确认，已挂帐未付货款为1473300元，后盈吉建筑公司支付70万元，尚欠773300元。而对于另一笔尚欠的432742.5元货款，双方未在上述对账过程结算确认。上述两笔欠款合计1206042.50元，伟业混凝土公司诉请盈吉建筑公司予以偿还。至起诉之时，432742.5元货款仍未进行结算处理。伟业混凝土公司再审申请主张，2014年6月20日、6月23日结算对账之前的2013年11月25日，伟业混凝土公司与盈吉建筑公司指定的签证人就432742.5元货款进行了结算。经查，涉案合同指定的签证人员为张玉邦，而非芦生英。故该份结算手续不能作为结算依据。伟业混凝土公司再审请求按照一审判决对所有欠付货款1206042.50元均计算违约金。根据本案合同第四条的约定，结算是付款的条件，未结算未付款不属于违约情形。对于已结算未支付的货款773300元计算违约金，二审法院判令盈吉建筑公司向伟业公司赔偿225900.26元并无不当。而对未结算的货款432742.5元，因不属于盈吉建筑公司的违约行为，不应计算违约金。因此，二审判决未对432742.5元货款计算违约金正确，应于维持。一审判决未加区分结算情况，对1206042.50元一并计收违约金不当。伟业混凝土公司再审请求撤销二审法院对违约金的判决、维持一审法院对违约金的判决，不能成立，本院不予支持。综上，原审判决认定事实清楚，适用法律正确。西宁伟业混凝土有限公司的再审申请不符合《中华人民共和国民事诉讼法》第二百条规定的情形。本院依照《中华人民共和国民事诉讼法》第二百零四条第一款之规定，裁定如下：驳回西宁伟业混凝土有限公司的再审申请。',
                        'answers': {
                                        '原告涉事企业':['山东隆坤房地产开发有限公司'],
                                        '原告涉事人员':['山东隆坤房地产开发有限公司于光远'],
                                        '原告委托代理人':['山东舜天律师事务所刘丕岐律师', '北京大成（青岛）律师事务所李红英律师'],
                                        '被告涉事企业':['山东省机床维修站','济南引力有限公司','济南鲁联集团投资有限公司'],
                                        '被告涉事人员':['山东省机床维修站站长、济南引力有限公司总经理张作让', '济南鲁联集团投资有限公司总经理薛鹏雁','济南鲁联集团有限公司董事长兼总经理刘津福'],
                                        '被告委托代理人':['山东诚信人律师事务所王勇律师','济南鲁联集团投资有限公司总经理郭士旺', '济南鲁联集团投资有限公司职工苏志高'],
                                        '涉事地点': ['山东省'],
                                        '涉事金额': ['迟延给付的432742.5元货款', '已挂帐未付货款为1473300元', '后支付70万元','尚欠773300元', '另一笔尚欠的432742.5元货款', '两笔欠款合计1206042.50元', '一审判决对所有欠付货款1206042.50元均计算违约金', '二审法院判令盈吉建筑公司向伟业公司赔偿225900.26元并无不当'],
                                        '案件规模':['个人独资企业'], 
                                        '原告主张': ['个人独资企业新艺建材厂已注销','个人独资企业主体消亡','从事经营活动','签订协议无效'],
                                        '被告主张': ['个人独资企业','投资人可进行股权转让','注销合理期限内','对外处理投资事务','不构成签订无效']
                                        }
                    },
                    {
                        'content': '本院认为，被告人卢小波故意非法剥夺他人生命，其行为已构成故意杀人罪；卢小波违反毒品管理法规，贩卖毒品的行为又构成贩卖毒品罪。卢小波吸食毒品后无端猜忌妻子有外遇并怀疑孩子不是亲生的而将其妻及二名亲生年幼子女杀死，犯罪情节特别恶劣，后果特别严重，罪行极其严重，应依法惩处',
                        'answers': {
                                        '原告涉事企业':['原文未提及'],
                                        '原告涉事人员':['原文未提及'],
                                        '原告委托代理人':['原文未提及'],
                                        '被告涉事企业':['原文未提及'],
                                        '被告涉事人员':['卢小波'],
                                        '被告委托代理人':['原文未提及'],
                                        '涉事地点': ['山东省'],
                                        '涉事金额': ['原文未提及'],
                                        '案件规模':['故意杀人罪','贩卖毒品罪'], 
                                        '原告主张': ['非法剥夺他人生命','违反毒品管理法规','贩卖毒品'],
                                        '被告主张': ['原文未提及']
                            }
                    }
            ],
            '处理结果': [
                        {
                            'content': '本院认为，本案争议焦点为，盈吉建筑公司迟延给付的432742.5元货款是否应当支付违约金。2011年3月25日，伟业混凝土公司与盈吉建筑公司分别签订《商品混凝土购销合同》，该合同第四条约定，供方为需方供应商砼，每供应5000m3混凝土结算一次，需方需在十日内支付前面所供5000m3混凝土款的80%，供方再为需方供应商砼，以此类推，直至工程主体封顶，每年12月31日前需方需付到所供混凝土总款的90%，余款在混凝土工程结束后一个月内全部付清。根据上述约定，盈吉建筑公司付款的前提条件是结算，在未结算前，不具备付款条件。经查，双方于2014年6月20日、6月23日进行结算对账，对所有项目中供货量进行了确认，已挂帐未付货款为1473300元，后盈吉建筑公司支付70万元，尚欠773300元。而对于另一笔尚欠的432742.5元货款，双方未在上述对账过程结算确认。上述两笔欠款合计1206042.50元，伟业混凝土公司诉请盈吉建筑公司予以偿还。至起诉之时，432742.5元货款仍未进行结算处理。伟业混凝土公司再审申请主张，2014年6月20日、6月23日结算对账之前的2013年11月25日，伟业混凝土公司与盈吉建筑公司指定的签证人就432742.5元货款进行了结算。经查，涉案合同指定的签证人员为张玉邦，而非芦生英。故该份结算手续不能作为结算依据。伟业混凝土公司再审请求按照一审判决对所有欠付货款1206042.50元均计算违约金。根据本案合同第四条的约定，结算是付款的条件，未结算未付款不属于违约情形。对于已结算未支付的货款773300元计算违约金，二审法院判令盈吉建筑公司向伟业公司赔偿225900.26元并无不当。而对未结算的货款432742.5元，因不属于盈吉建筑公司的违约行为，不应计算违约金。因此，二审判决未对432742.5元货款计算违约金正确，应于维持。一审判决未加区分结算情况，对1206042.50元一并计收违约金不当。伟业混凝土公司再审请求撤销二审法院对违约金的判决、维持一审法院对违约金的判决，不能成立，本院不予支持。综上，原审判决认定事实清楚，适用法律正确。西宁伟业混凝土有限公司的再审申请不符合《中华人民共和国民事诉讼法》第二百条规定的情形。本院依照《中华人民共和国民事诉讼法》第二百零四条第一款之规定，裁定如下：驳回西宁伟业混凝土有限公司的再审申请。',
                            'answers': {
                                            '法院判决结果': ['驳回再审申请'],
                                            '罪名判决': ['原文未提及'],
                                            '处罚': ['二审判决未对432742.5元货款计算违约金正确，应于维持。一审判决未加区分结算情况，对1206042.50元一并计收违约金不当。']
                                }
                        },
                        {
                            'content': '第一审判决、第二审裁定认定的事实清楚，证据确实、充分，定罪准确，量刑适当。审判程序合法。依照《中华人民共和国刑事诉讼法》第二百三十五条、第二百三十九条和《最高人民法院关于适用〈中华人民共和国刑事诉讼法〉的解释》第三百五十条第（一）项的规定，裁定如下：核准四川省高级人民法院（2013）川刑终字第1005号维持第一审对被告人卢小波以故意杀人罪判处死刑，剥夺政治权利终身；以贩卖毒品罪判处有期徒刑十年，并处罚金人民币一万元，决定执行死刑，剥夺政治权利终身，并处罚金人民币一万元的刑事裁定。本裁定自宣告之日起发生法律效力。',
                            'answers': {
                                            '法院判决结果': ['第一审判决、第二审裁定认定的事实清楚，证据确实、充分，定罪准确，量刑适当。审判程序合法。'],
                                            '罪名判决': ['故意杀人罪','贩卖毒品罪'],
                                            '处罚': ['判处死刑，剥夺政治权利终身','有期徒刑十年，并处罚金人民币一万元']
                                }
                        }
            ]
    }

    """
    初始化前置prompt，便于模型做 incontext learning。
    """
    ie_pre_history = [
        (
            "现在你需要帮助我完成信息抽取任务，当我给你一个句子时，你需要帮我抽取出句子中提及到有关先前属性的实体信息，并按照JSON的格式输出，上述句子中没有的信息用['原文中未提及']来表示，多个值之间用','分隔。",
            '好的，请输入您的句子。'
        )
    ]

    for _type, example_list in ie_examples.items():
        for example in example_list:
            sentence = example['content']
            text_class = _type
            text_class_schema = schema[_type]
            sentence_with_prompt = IE_PATTERN.format(sentence, text_class, text_class_schema)
            ie_pre_history.append((
                f'{sentence_with_prompt}',
                f"{json.dumps(example['answers'], ensure_ascii=False)}"
            ))

    """
    推理部分。

    Args:
        sentences (List[str]): 待抽取的句子。
        custom_settings (dict): 初始设定，包含人为给定的 few-shot example。
    """
    #用户输入文本‘案发过程’和‘处理结果’作为模型的输入
    user_inputs = {'案发过程': '', '处理结果': ''}

    for i in range(len(user_inputs.keys())):
        user_input = ''
        while True:
            line = input(f'请输入{list(user_inputs.keys())[i]}（按回车结束输入）: ')
            if not line:  # 如果用户输入为空（按下回车），结束循环
                break
            user_input += line + '\n'

        user_inputs[list(user_inputs.keys())[i]] = user_input.strip()  # 去除多余的换行符

    #不用input 直接给函数加两个参数text1和text2分别代表输入的案发过程和处理结果的文本时：
    #user_inputs = {'案发过程': text1, '处理结果': text2}

    responses = []
    for key, value in user_inputs.items():
        with console.status("[bold bright_green] Model Inference..."):
            sentence_with_ie_prompt = IE_PATTERN.format(value,key,schema[key])
            print(sentence_with_ie_prompt)

            ie_res, _ = model.chat(tokenizer, sentence_with_ie_prompt, history=ie_pre_history)
        print(f'>>> [bold bright_red]sentence: {value}')
        print(f'>>> [bold bright_green]inference answer: ')
        print(ie_res)
        responses.append(ie_res)
    
    return responses 


if __name__ == '__main__':
    console = Console()
    device = 'cuda:0'
    tokenizer = AutoTokenizer.from_pretrained('/mnt/workspace/ChatGLM2-6b/THUDM/chatglm2-6b', trust_remote_code=True)
    model = AutoModel.from_pretrained('/mnt/workspace/ChatGLM2-6b/THUDM/chatglm2-6b', trust_remote_code=True,torch_dtype=torch.float32)

    #tokenizer = AutoTokenizer.from_pretrained("/mnt/workspace/local_model/Baichuan2-7B-Chat-4bits", use_fast=False, trust_remote_code=True)
    #model = AutoModelForCausalLM.from_pretrained("/mnt/workspace/local_model/Baichuan2-7B-Chat-4bits", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    #model.generation_config = GenerationConfig.from_pretrained("/mnt/workspace/local_model/Baichuan2-7B-Chat-4bits")

    model = model.eval()

    extraction()