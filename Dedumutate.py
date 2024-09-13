import json
import random
import os
from tqdm import tqdm
import re
from fuzzywuzzy import fuzz
from APIcoverage_PYskeleton_DSL_multi import query_deepseek,query_silicon,query_openai,query_openai_Azure

# 给定一个路径? 不对, 应该是文件. 我们单个文件不允许有完全一样的, 但是不同文件里面我们不做限制
# 对于单个文件, 我们的目标是 (1) 找到里面重复的, 把不重复的 + 重复的但是改写的 变成新的文件, 以FIX_开头 => 保留原本的其他字段
# 接下来还是原来的步骤, 只需要改变rules里面对于文件名的规范即可

# 判断两个字符串是否相似

def fuzzy_similarity(str1, str2):
    return fuzz.ratio(str1, str2) / 100

def are_strings_similar(str1, str2, threshold=0.8, strict=True):
    if strict:
        return str1.strip() == str2.strip() , -1
    else:
        # similarity = SequenceMatcher(None, str1, str2).ratio()
        similarity = fuzz.ratio(str1, str2) / 100
        return similarity >= threshold, similarity

def strinlist(str1, targetlist):
    # if 'instruction' not in targetlist[0]:
        # 需要自己
    for item in targetlist:
        str2 = item['instruction'] # 通过instruction来看
        if are_strings_similar(str1, str2)[0]:
            return True
    return False

def remove_comments(code, delete_one_line_comment=True):
    # 去掉单行注释
    if delete_one_line_comment:
        code = re.sub(r'#.*', '', code)
    code = code.strip()
    # 如果遇到 "Example usage", 则丢掉其后面所有内容
    if 'Example usage' in code:
        code = code.split('Example usage')[0]
    # 提取所有在```python 和 ```之间的内容
    if '```python' in code:
        codes_list = re.findall(r'```python(.*?)```', code, re.DOTALL)
        code = '\n'.join(codes_list)
    return code

'''
{"output":xxx
"instruction": xxx
}

'''

def simplify_dict(content,delete_one_line_comment=True):
    if content is None:
        print('content is None')
        return '', ''
    nocometcode = ''
    if '[Solution]' not in content:
        return '', ''
    problem = content.split('[Solution]')[0].replace('[Problem Description]','').strip()
    solution = content.split('[Solution]')[1]
    # 第二步是提取出solution 里面的code 和explain
    # 首先看 ``` 有几个, 如果有两个, 则第一个是code, 第二个是explain
    if solution.count('```') > 1: # 至少2个
        code = solution.split('```')[1].strip()
        explain = solution.split('```')[2].strip()
        # assert code.startswith('python\n') , code
        if code.startswith('python\n'):
            code = code[7:]
            if code.startswith('\n'):
                code = code[1:]
        nocometcode = remove_comments(code,delete_one_line_comment=delete_one_line_comment).strip()
    else:
        nocometcode = remove_comments(solution,delete_one_line_comment=delete_one_line_comment).strip()


    return nocometcode, problem 


def reask(visible=False, model_name='gpt-35-turbo',prompt_template_dir=None,template_num="0",problem="",code=""):
    system_prompt = 'You are a helpful assistant.'
    
    if prompt_template_dir and os.path.exists(prompt_template_dir) and prompt_template_dir.endswith('.json'):
        with open(prompt_template_dir, 'r') as f:
            prompt_template = json.load(f)
            for prompt in prompt_template:
                if prompt["stage"]=="reask" and prompt["template"]==template_num:
                    question_prompt = prompt["content"]
                    break
    if template_num == '0': # 感觉这个一般
        new_question_prompt = question_prompt.format(code=code)
    elif template_num == '1':  # 感觉这个似乎还行
        new_question_prompt = question_prompt.format(problem=problem, code=code)
    elif template_num == '2':  # 直接就单纯让它更改problem的提问方法, 以及润色code
        new_question_prompt = question_prompt.format(problem=problem, code=code)
    question_prompt = new_question_prompt

    if model_name in ['gpt-3.5-turbo', 'gpt-4o', 'gpt-4', 'gpt-4-turbo']:
        response, used_token = query_openai(input=question_prompt, system_promt=system_prompt, visible=visible, model=model_name)
    elif model_name in ['gpt-35-turbo']:
        response, used_token = query_openai_Azure(input=question_prompt, system_promt=system_prompt, visible=visible, model=model_name)
    elif model_name in ['deepseek' ]:
        response, used_token = query_deepseek(input=question_prompt, system_promt=system_prompt, visible=visible)
    elif model_name in ['deepseek-silicon']:
        response, used_token = query_silicon(input=question_prompt, system_promt=system_prompt, visible=visible)
    return question_prompt, response, used_token

def readfile(filefullname, delete_one_line_comment=True,deduplicate=False,template_num="2"):
    usedmodel = 'deepseek'
    prompt_template_dir = 'prompt_template_multi.json'
    All_content_simple, All_content = [], []   # 简要信息和完整信息
    filedir, filename = os.path.split(filefullname)
    print(filename)
    print(filedir)
    
    outputjson_name = 'FIX_'+ filename
    # 检测是否存在, 如果存在, 统计已经有了多少个
    if os.path.exists(os.path.join(filedir, outputjson_name)):
        already_count = len(open(os.path.join(filedir, outputjson_name), 'r').readlines())
    else:
        already_count = 0

    print('already_count', already_count)
    with open(filefullname, 'r') as f:
        lines = f.readlines()
        number_examples = len(lines)
        if already_count >= number_examples:
            print('all examples have been queryed')
            return 0
        if already_count!=0: # 如果已经有了,则需要把已经query到的全部读取到All_content_simple 和 All_content
            with open(os.path.join(filedir, outputjson_name), 'r') as f:
                alreadylines = f.readlines()
                for line in alreadylines:
                    line = json.loads(line)
                    content = line['result']
                    nocometcode, problem = simplify_dict(content,delete_one_line_comment=delete_one_line_comment)
                    obj = {"output": nocometcode, "instruction": problem}
                    All_content_simple.append(obj)
                    All_content.append(line)

        print('finished loading the already queryed content')
        # for line in tqdm(lines):
        # 从already_count 开始
        for line in tqdm(lines[already_count:]):
            successfully_change = True
            line = json.loads(line)
            content = line['result']
            nocometcode, problem = simplify_dict(content,delete_one_line_comment=delete_one_line_comment)

            if not strinlist(problem, All_content_simple):  # 没有重复, 直接添加
                print('no repeat, add it -----------------')
                obj = {"output": nocometcode, "instruction": problem}
                line['reask'] = False
                All_content_simple.append(obj)
                All_content.append(line)

            else:  # 否则重新query 得到新的问题
                print('start reask the problem ---------------------------')
                question_prompt, response, used_token = reask(visible=False, model_name=usedmodel,prompt_template_dir=prompt_template_dir,template_num=template_num,problem=problem,code=nocometcode)

                nocometcode_round2, problem_round2 = simplify_dict(response, delete_one_line_comment=delete_one_line_comment)
                print('end reask the problem ---------------------------')

                if not strinlist(problem_round2, All_content_simple):  # 没有重复, 直接添加
                    obj = {"output": nocometcode_round2, "instruction": problem_round2}
                    line['result'] = response
                    line['used_token'] = used_token
                    line['reask'] = True
                    All_content_simple.append(obj)
                    All_content.append(line)
                else:
                    print('after deduplicate, still has the same problem:', problem_round2)
                    successfully_change = False

            if successfully_change:
                # 写入jsonl文件
                with open(os.path.join(filedir, outputjson_name), 'a') as f:
                    # for obj in All_content:
                    f.write(json.dumps(line) + '\n')


# 指定一系列规则以判断是否要选取
def selected_rules(file):
    need_return = True
    need_specify_model = False
    need_specify_type = True # 是否要选择不同的来源组合
    need_specify_difficulty = True # 是否要选择不同的难度
    modelname = 'deepseek'
    if need_specify_model:
        if modelname not in file:
            need_return = False
    if need_specify_type:
        if 'matplotlib' in file or 'pandas' in file or 'numpy' in file or 'MIX_3' in file:
            need_return = False
    
    return need_return



            
def main():
    basedir = '/export/d3/zjli/API_coverage_LLM/Advanced/QA_split/data/PYSkeletonDSL/generalpython/deepseek'  # 只有这个里面才有简单的代码
    print('start')
    for root, dirs, files in os.walk(basedir):
        for file in files:
            if file.endswith('.jsonl') and selected_rules(root):  # 读取符合规则的jsonl
                if 'level2' in file or 'SIMPLETrue' not in file or  'FIX_' in file:
                    continue
                filepath = os.path.join(root, file)
                # 读取该jsonl文件
                print(filepath)
                readfile(filepath)
                # break

if __name__ == '__main__':
    main()