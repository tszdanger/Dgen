import json
import random
import os
from tqdm import tqdm
import re
from fuzzywuzzy import fuzz
from APIcoverage_PYskeleton_DSL_multi import query_deepseek, query_silicon, query_openai, query_openai_Azure

# For a single file, our goal is to:
# (1) Find duplicates within it
# (2) Create a new file with non-duplicates + rewritten duplicates, prefixed with 'FIX_'
# (3) Retain other original fields

def fuzzy_similarity(str1, str2):
    return fuzz.ratio(str1, str2) / 100

def are_strings_similar(str1, str2, threshold=0.8, strict=True):
    if strict:
        return str1.strip() == str2.strip(), -1
    else:
        similarity = fuzz.ratio(str1, str2) / 100
        return similarity >= threshold, similarity

def strinlist(str1, targetlist):
    for item in targetlist:
        str2 = item['instruction']
        if are_strings_similar(str1, str2)[0]:
            return True
    return False

def remove_comments(code, delete_one_line_comment=True):
    if delete_one_line_comment:
        code = re.sub(r'#.*', '', code)
    code = code.strip()
    if 'Example usage' in code:
        code = code.split('Example usage')[0]
    if '```python' in code:
        codes_list = re.findall(r'```python(.*?)```', code, re.DOTALL)
        code = '\n'.join(codes_list)
    return code

def simplify_dict(content, delete_one_line_comment=True):
    if content is None:
        print('content is None')
        return '', ''
    nocometcode = ''
    if '[Solution]' not in content:
        return '', ''
    problem = content.split('[Solution]')[0].replace('[Problem Description]', '').strip()
    solution = content.split('[Solution]')[1]
    
    if solution.count('```') > 1:
        code = solution.split('```')[1].strip()
        explain = solution.split('```')[2].strip()
        if code.startswith('python\n'):
            code = code[7:]
            if code.startswith('\n'):
                code = code[1:]
        nocometcode = remove_comments(code, delete_one_line_comment=delete_one_line_comment).strip()
    else:
        nocometcode = remove_comments(solution, delete_one_line_comment=delete_one_line_comment).strip()

    return nocometcode, problem 

def reask(visible=False, model_name='gpt-35-turbo', prompt_template_dir=None, template_num="0", problem="", code=""):
    system_prompt = 'You are a helpful assistant.'
    
    if prompt_template_dir and os.path.exists(prompt_template_dir) and prompt_template_dir.endswith('.json'):
        with open(prompt_template_dir, 'r') as f:
            prompt_template = json.load(f)
            for prompt in prompt_template:
                if prompt["stage"] == "reask" and prompt["template"] == template_num:
                    question_prompt = prompt["content"]
                    break
    if template_num == '0':
        new_question_prompt = question_prompt.format(code=code)
    elif template_num == '1':
        new_question_prompt = question_prompt.format(problem=problem, code=code)
    elif template_num == '2':
        new_question_prompt = question_prompt.format(problem=problem, code=code)
    question_prompt = new_question_prompt

    if model_name in ['gpt-3.5-turbo', 'gpt-4o', 'gpt-4', 'gpt-4-turbo']:
        response, used_token = query_openai(input=question_prompt, system_promt=system_prompt, visible=visible, model=model_name)
    elif model_name in ['gpt-35-turbo']:
        response, used_token = query_openai_Azure(input=question_prompt, system_promt=system_prompt, visible=visible, model=model_name)
    elif model_name in ['deepseek']:
        response, used_token = query_deepseek(input=question_prompt, system_promt=system_prompt, visible=visible)
    elif model_name in ['deepseek-silicon']:
        response, used_token = query_silicon(input=question_prompt, system_promt=system_prompt, visible=visible)
    return question_prompt, response, used_token

def readfile(filefullname, delete_one_line_comment=True, deduplicate=False, template_num="2"):
    usedmodel = 'deepseek'
    prompt_template_dir = 'prompt_template_multi.json'
    All_content_simple, All_content = [], []
    filedir, filename = os.path.split(filefullname)
    print(filename)
    print(filedir)
    
    outputjson_name = 'FIX_' + filename
    if os.path.exists(os.path.join(filedir, outputjson_name)):
        already_count = len(open(os.path.join(filedir, outputjson_name), 'r').readlines())
    else:
        already_count = 0

    print('already_count', already_count)
    with open(filefullname, 'r') as f:
        lines = f.readlines()
        number_examples = len(lines)
        if already_count >= number_examples:
            print('all examples have been queried')
            return 0
        if already_count != 0:
            with open(os.path.join(filedir, outputjson_name), 'r') as f:
                alreadylines = f.readlines()
                for line in alreadylines:
                    line = json.loads(line)
                    content = line['result']
                    nocometcode, problem = simplify_dict(content, delete_one_line_comment=delete_one_line_comment)
                    obj = {"output": nocometcode, "instruction": problem}
                    All_content_simple.append(obj)
                    All_content.append(line)

        print('finished loading the already queried content')
        for line in tqdm(lines[already_count:]):
            successfully_change = True
            line = json.loads(line)
            content = line['result']
            nocometcode, problem = simplify_dict(content, delete_one_line_comment=delete_one_line_comment)

            if not strinlist(problem, All_content_simple):
                print('no repeat, add it -----------------')
                obj = {"output": nocometcode, "instruction": problem}
                line['reask'] = False
                All_content_simple.append(obj)
                All_content.append(line)
            else:
                print('start reask the problem ---------------------------')
                question_prompt, response, used_token = reask(visible=False, model_name=usedmodel, prompt_template_dir=prompt_template_dir, template_num=template_num, problem=problem, code=nocometcode)

                nocometcode_round2, problem_round2 = simplify_dict(response, delete_one_line_comment=delete_one_line_comment)
                print('end reask the problem ---------------------------')

                if not strinlist(problem_round2, All_content_simple):
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
                with open(os.path.join(filedir, outputjson_name), 'a') as f:
                    f.write(json.dumps(line) + '\n')

# Define rules for file selection
def selected_rules(file):
    need_return = True
    need_specify_model = False
    need_specify_type = True
    need_specify_difficulty = True
    modelname = 'deepseek'
    if need_specify_model:
        if modelname not in file:
            need_return = False
    if need_specify_type:
        if 'matplotlib' in file or 'pandas' in file or 'numpy' in file or 'MIX_3' in file:
            need_return = False
    
    return need_return

def main():
    basedir = '/path/to/your/data/directory'
    print('start')
    for root, dirs, files in os.walk(basedir):
        for file in files:
            if file.endswith('.jsonl') and selected_rules(root):
                if 'level2' in file or 'SIMPLETrue' not in file or 'FIX_' in file:
                    continue
                filepath = os.path.join(root, file)
                print(filepath)
                readfile(filepath)

if __name__ == '__main__':
    main()
