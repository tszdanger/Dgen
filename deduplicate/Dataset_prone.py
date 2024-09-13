import os
import json
import re
from tqdm import tqdm
from fuzzywuzzy import fuzz

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
    # Remove single-line comments
    if delete_one_line_comment:
        code = re.sub(r'#.*', '', code)
    code = code.strip()
    # If "Example usage" is encountered, discard all content after it
    if 'Example usage' in code:
        code = code.split('Example usage')[0]
    # Extract all content between ```python and ```
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
    problem = content.split('[Solution]')[0].replace('[Problem Description]','').strip()
    solution = content.split('[Solution]')[1]
    # Extract code and explanation from the solution
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

def selected_rules(root, file=''):
    need_return = True
    need_specify_model = False
    need_specify_type = False
    need_select_fix = True # choose from deduplicate examples or not
    modelname = 'deepseek'
    if need_specify_model:
        if modelname not in root:
            need_return = False
    if need_specify_type:
        if 'matplotlib' in root or 'pandas' in root or 'numpy' in root or 'MIX_3' in root:
            need_return = False
    if need_select_fix:
        if 'generalpython' in root and 'SIMPLETrue' in file and 'level1' in file and 'FIX' not in file:
            need_return = False
    return need_return

def readfile(basedir, delete_one_line_comment=False, deduplicate=True):
    dataset_info = {}
    All_content_simple, All_content = [], []
    outputdir = '/path/to/output/directory'
    outputjson_name = 'generalpython_22k_DEDUFIX.json'
    if not delete_one_line_comment:
        outputjson_name = outputjson_name.replace('.json','_wcoment.json')
    outputjson_name = os.path.join(outputdir, outputjson_name)
    outputjson_name_simple = outputjson_name.replace('.json','_simple.json')
    txt_name = outputjson_name.replace('.json','.txt')
    
    for root, dirs, files in os.walk(basedir):
        for file in files:
            if file.endswith('.jsonl') and selected_rules(root, file=file):  
                filepath = os.path.join(root, file)
                print(filepath)
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                    number_examples = len(lines)
                    short_filepath = filepath.replace(basedir, '')
                    dataset_info[short_filepath] = number_examples
                    for line in tqdm(lines):
                        line = json.loads(line)
                        content = line['result']
                        nocometcode, problem = simplify_dict(content, delete_one_line_comment=delete_one_line_comment)
                        if deduplicate:
                            if not strinlist(problem, All_content_simple):
                                obj = {"output": nocometcode, "instruction": problem}
                                All_content_simple.append(obj)
                                All_content.append(line)
                        else:
                            obj = {"output": nocometcode, "instruction": problem}
                            All_content_simple.append(obj)
                            All_content.append(line)
    
    print(f'In total we have {len(All_content_simple)} items in new dataset.')
    print('Output file is ', outputjson_name)
    print('Here is its info: ')
    for k, v in dataset_info.items():
        print(k, v)

def main():
    basedir = '/path/to/input/directory'
    readfile(basedir)

if __name__ == '__main__':
    main()