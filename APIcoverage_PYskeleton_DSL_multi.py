import json
import random
from openai import OpenAI
import os
from tqdm import tqdm
from DSL import generate_valid_code_skeleton
import re
from NumberGener import generate_list
import time
import argparse

def remove_comments(code):
    """
    Removes comments from a given code string.

    Args:
        code (str): The input code string.

    Returns:
        str: The code string with comments removed.
    """
    
    code = re.sub(r'#.*', '', code)
    code = code.strip()
    if '```python' in code:
        codes_list = re.findall(r'```python(.*?)```', code, re.DOTALL)
        code = '\n'.join(codes_list)
    return code

# "sk-221ce4c301a14ddcb88b69e527f30837"

def query_deepseek(input, system_promt='You are a helpful assistant.',key="sk-xxxxx", visible=True):
    '''
    Sends a query to the DeepSeek API and returns the response.

	Parameters:
		input (str): The user's input to be sent to the DeepSeek API.
		system_promt (str): The system prompt to be sent to the DeepSeek API. Defaults to 'You are a helpful assistant.'.
		deepseek_key (str): The DeepSeek API key. Defaults to "sk-xxxxx".
		visible (bool): Whether to print the response. Defaults to True.

	Returns:
		tuple: A tuple containing the response content and the total tokens used.
	'''

    client = OpenAI(api_key=key, base_url="https://api.deepseek.com")

    completion = client.chat.completions.create(
        model="deepseek-chat",
        temperature=0.8,
        max_tokens=2048,
        messages=[
            {"role": "system", "content": system_promt},
            {"role": "user", "content": input},
        ]
    )

    if visible:
        print(completion.choices[0].message.content)
    used_token = completion.usage.total_tokens
    return completion.choices[0].message.content, used_token

def query_silicon(input, system_promt='You are a helpful assistant.', key="sk-xxxx", visible=True):
    """
    Sends a query to the Silicon API and returns the response.

    Parameters:
        input (str): The user's input to be sent to the Silicon API.
        system_promt (str): The system prompt to be sent to the Silicon API. Defaults to 'You are a helpful assistant.'.
        key (str): The Silicon API key. Defaults to "sk-xxxx".
        visible (bool): Whether to print the response. Defaults to True.

    Returns:
        tuple: A tuple containing the response content and the total tokens used.
    """
    client = OpenAI(api_key=key, base_url="https://api.siliconflow.cn/v1")

    completion = client.chat.completions.create(
        model="deepseek-ai/deepseek-v2-chat",
        temperature=0,
        max_tokens=2048,
        messages=[
            {"role": "system", "content": system_promt},
            {"role": "user", "content": input},
        ]
    )

    if visible:
        print(completion.choices[0].message.content)
    used_token = completion.usage.total_tokens
    return completion.choices[0].message.content, used_token


def query_openai(input, system_promt='You are a helpful assistant.', model="gpt-35-turbo", visible=True, key="", temperature=0.8, max_tokens=2048, top_p=1.0):
    """
    Sends a query to the OpenAI API and returns the response.

    Parameters:
        input (str): The user's input to be sent to the OpenAI API.
        system_promt (str): The system prompt to be sent to the OpenAI API. Defaults to 'You are a helpful assistant.'.
        model (str): The model to be used for the query. Defaults to "gpt-35-turbo".
        visible (bool): Whether to print the response. Defaults to True.
        key (str): The OpenAI API key. Defaults to an empty string.
        temperature (float): The temperature for the query. Defaults to 0.8.
        max_tokens (int): The maximum number of tokens for the query. Defaults to 2048.
        top_p (float): The top p value for the query. Defaults to 1.0.

    Returns:
        tuple: A tuple containing the response content and the total tokens used.
    """

    client = OpenAI(api_key=key)
    messages=[
            {"role": "system", "content": system_promt},
            {"role": "user", "content": input},
        ]
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    if visible:
        print(completion.choices[0].message.content)
    used_token = completion.usage.total_tokens
    return completion.choices[0].message.content, used_token


def read_apis_from_json(file_path):
    """
    Reads a JSON file and returns its contents.

    Args:
        file_path (str): The path to the JSON file to be read.

    Returns:
        dict: The contents of the JSON file.
    """
    with open(file_path, 'r') as f:
        content = json.load(f)
    return content

def get_n_apis(inputapis, n):
    """
    Returns a list of n randomly selected APIs from the input list.

    Parameters:
        inputapis (list): A list of APIs to select from.
        n (int): The number of APIs to select.

    Returns:
        list: A list of tuples containing the name and description of the selected APIs.
    """
    apis = inputapis.copy()
    selected_apis = []
    for i in range(n):
        api_idx = random.randint(0, len(apis) - 1)
        api = apis[api_idx]
        selected_apis.append((api['name'], api['describe']))
        apis.pop(api_idx)
    return selected_apis


def remove_empty_lines(text):
    """
    Removes consecutive empty lines from a given text.

    Args:
        text (str): The input text to process.

    Returns:
        str: The text with consecutive empty lines removed.
    """
    while '\n\n' in text:
        text = text.replace('\n\n', '\n')
    return text


def step0(selected_apis, visible=False, model_name='gpt-35-turbo',prompt_template_dir=None,template_num="1",Onerepo=True,use_builtin=True):
    """
    This function generates a prompt for the first step of a process, simple way
    formats it with a list of selected APIs, and queries a model for a response.

    Parameters:
        selected_apis (list): A list of tuples containing the name and description of the selected APIs.
        visible (bool): A flag indicating whether the response should be printed. Defaults to False.
        model_name (str): The name of the model to be used for the query. Defaults to 'gpt-35-turbo'.
        prompt_template_dir (str): The path to a JSON file containing prompt templates. Defaults to None.
        template_num (str): The number of the template to be used. Defaults to "1".
        Onerepo (bool): A flag indicating whether to use a single repository. Defaults to True.
        use_builtin (bool): A flag indicating whether to use built-in APIs. Defaults to True.

    Returns:
        tuple: A tuple containing the formatted prompt and the response from the model.
    """
    system_prompt = 'You are a helpful assistant.'
    API_list_str= '\n'
    for idx in range(len(selected_apis)):
        API_list_str += f'{idx}: {selected_apis[idx][0]}: {selected_apis[idx][1]}\n'
    if prompt_template_dir and os.path.exists(prompt_template_dir) and prompt_template_dir.endswith('.json'):
        with open(prompt_template_dir, 'r') as f:
            prompt_template = json.load(f)
            for prompt in prompt_template:
                if prompt["stage"]=="step0" and prompt["template"]==template_num:
                    question_prompt = prompt["content"]
                    break
    new_question_prompt = question_prompt.format(API_list_str=API_list_str)
    question_prompt = new_question_prompt

    if model_name in ['gpt-3.5-turbo', 'gpt-4o', 'gpt-4', 'gpt-4-turbo']:
        response, used_token = query_openai(input=question_prompt, system_promt=system_prompt, visible=visible, model=model_name)
    elif model_name in ['deepseek' ]:
        response, used_token = query_deepseek(input=question_prompt, system_promt=system_prompt, visible=visible)
    elif model_name in ['deepseek-silicon']:
        response, used_token = query_silicon(input=question_prompt, system_promt=system_prompt, visible=visible)
    return question_prompt, response, used_token


def step1(N=5, random_stmt = '<Random Stmt>', need_grammar_check = True):
    """
    Generates a valid code skeleton based on the given parameters.

    Parameters:
        N (int): The number of code statements to generate. Defaults to 5.
        random_stmt (str): A random statement to include in the code skeleton. Defaults to '<Random Stmt>'.
        need_grammar_check (bool): A flag indicating whether to perform grammar checks on the generated code. Defaults to True.

    Returns:
        tuple: A tuple containing the generated code skeleton and the corresponding Treesitter code.
    """
    code_skeleton, treesitter_code = generate_valid_code_skeleton(N, random_stmt = random_stmt, need_grammar_check = need_grammar_check)
    return code_skeleton, treesitter_code


def step2(repo, selected_apis, code_skeleton, visible=False, model_name='gpt-35-turbo',prompt_template_dir=None,template_num="1",Onerepo=True,use_builtin=False,difficulty_level=1):
    """
    This function generates a prompt for step 2 of the API coverage process. Generating the pseudo code.
    
    Parameters:
    repo (str): The repository name.
    selected_apis (list): A list of selected APIs.
    code_skeleton (str): The code skeleton.
    visible (bool): Whether the prompt is visible. Defaults to False.
    model_name (str): The name of the model. Defaults to 'gpt-35-turbo'.
    prompt_template_dir (str): The directory of the prompt template. Defaults to None.
    template_num (str): The template number. Defaults to "1".
    Onerepo (bool): Whether it's a single repository. Defaults to True.
    use_builtin (bool): Whether to use built-in templates. Defaults to False.
    difficulty_level (int): The difficulty level. Defaults to 1.
    
    Returns:
    tuple: A tuple containing the generated prompt, the response, and the used token.
    """
    system_prompt = 'You are a helpful assistant.'
    API_list_str= '\n'
    for idx in range(len(selected_apis)):
        API_list_str += f'{idx}: {selected_apis[idx][0]}: {selected_apis[idx][1]}\n'
    if prompt_template_dir and os.path.exists(prompt_template_dir) and prompt_template_dir.endswith('.json'):
        with open(prompt_template_dir, 'r') as f:
            prompt_template = json.load(f)
            for prompt in prompt_template:
                if use_builtin:
                    # if difficulty_level == 1:# 只用基础的api or 两者都用
                    if prompt["stage"]=="step2" and prompt["template"]==template_num and "type" in prompt and prompt['type']=='builtin' and prompt['difficulty_level'] == str(difficulty_level):
                        question_prompt = prompt["content"]
                        break
                else:
                    if prompt["stage"]=="step2" and prompt["template"]==template_num:
                        question_prompt = prompt["content"]
                        break
    if Onerepo:
        new_question_prompt = question_prompt.format(repo=repo,API_list_str=API_list_str,code_skeleton=code_skeleton)  
    else:
        new_question_prompt = question_prompt.format(API_list_str=API_list_str)
    question_prompt = new_question_prompt

    if model_name in ['gpt-3.5-turbo', 'gpt-4o', 'gpt-4', 'gpt-4-turbo']:
        response, used_token = query_openai(input=question_prompt, system_promt=system_prompt, visible=visible, model=model_name)
    elif model_name in ['deepseek' ]:
        response, used_token = query_deepseek(input=question_prompt, system_promt=system_prompt, visible=visible)
    elif model_name in ['deepseek-silicon']:
        response, used_token = query_silicon(input=question_prompt, system_promt=system_prompt, visible=visible)
    return question_prompt, response, used_token



def step3(repo, code_skeleton, psudo_code=None, visible=False, model_name='gpt-35-turbo',selected_apis=[],prompt_template_dir=None,template_num="2",Onerepo=False,use_builtin=False):
    """
    This function generates a prompt for step 3 of the API coverage process. Generating the Question and Answer.
    
    Parameters:
    repo (str): The repository name.
    code_skeleton (str): The code skeleton.
    psudo_code (str, optional): The pseudo code. Defaults to None.
    visible (bool, optional): Whether the process is visible. Defaults to False.
    model_name (str, optional): The model name. Defaults to 'gpt-35-turbo'.
    selected_apis (list, optional): The selected APIs. Defaults to [].
    prompt_template_dir (str, optional): The prompt template directory. Defaults to None.
    template_num (str, optional): The template number. Defaults to "2".
    Onerepo (bool, optional): Whether it's a one-repo process. Defaults to False.
    use_builtin (bool, optional): Whether to use built-in templates. Defaults to False.

    Returns:
    tuple: A tuple containing the generated prompt, the response, and the used token.
    """
    system_prompt = 'You are a helpful assistant.'
    API_list_str= '\n'
    for idx in range(len(selected_apis)):
        API_list_str += f'{idx}: {selected_apis[idx][0]}: {selected_apis[idx][1]}\n'

    if prompt_template_dir and os.path.exists(prompt_template_dir) and prompt_template_dir.endswith('.json'):
        with open(prompt_template_dir, 'r') as f:
            prompt_template = json.load(f)
            for prompt in prompt_template:
                if use_builtin:
                    if prompt["stage"]=="step3" and prompt["template"]==template_num and 'type' in prompt and prompt['type']=='builtin':
                        question_prompt = prompt["content"]
                        break
                else:
                    if prompt["stage"]=="step3" and prompt["template"]==template_num:
                        question_prompt = prompt["content"]
                        break
    if Onerepo and template_num == "2":
        new_question_prompt = question_prompt.format(repo=repo,code=code_skeleton)
    elif not Onerepo and template_num == "2":
        question_prompt.format(repo='', scode=code_skeleton)
    elif template_num == "5" or template_num == "6":
        new_question_prompt = question_prompt.format(code_skeleton=code_skeleton,API_list_str=API_list_str, psudo_code=psudo_code)

    question_prompt = new_question_prompt
    if model_name in ['gpt-3.5-turbo', 'gpt-4o', 'gpt-4', 'gpt-4-turbo']:
        response, used_token = query_openai(input=question_prompt, system_promt=system_prompt, visible=visible, model=model_name)
    elif model_name in ['deepseek']:
        response, used_token = query_deepseek(input=question_prompt, system_promt=system_prompt, visible=visible)
    elif model_name in ['deepseek-silicon']:
        response, used_token = query_silicon(input=question_prompt, system_promt=system_prompt, visible=visible)
    return question_prompt, response, used_token


def step4(repo, example_code, visible=False, model_name='gpt-35-turbo',selected_apis=[],prompt_template_dir=None,template_num="1"):
    """
    This function generates a prompt for step 4 of the API coverage process. Get inspiration from the Question and Answer.
    
    Parameters:
    repo (str): The repository name.
    example_code (str): The example code.
    visible (bool): Whether the process is visible. Defaults to False.
    model_name (str): The model name. Defaults to 'gpt-35-turbo'.
    selected_apis (list): The selected APIs. Defaults to [].
    prompt_template_dir (str): The prompt template directory. Defaults to None.
    template_num (str): The template number. Defaults to "1".
    
    Returns:
    tuple: A tuple containing the question prompt, response, and used token.
    """
    system_prompt = 'You are a helpful assistant.'
    API_list_str= '\n'
    for idx in range(len(selected_apis)):
        API_list_str += f'{idx}: {selected_apis[idx][0]}: {selected_apis[idx][1]}\n'

    if prompt_template_dir and os.path.exists(prompt_template_dir) and prompt_template_dir.endswith('.json'):
        with open(prompt_template_dir, 'r') as f:
            prompt_template = json.load(f)
            for prompt in prompt_template:
                if prompt["stage"]=="step4" and prompt["template"]==template_num:
                    question_prompt = prompt["content"]
                    break
    
    if template_num == "1":
        new_question_prompt = question_prompt.format(code=example_code,API_list_str=API_list_str)

    question_prompt = new_question_prompt
    if model_name in ['gpt-3.5-turbo', 'gpt-4o', 'gpt-4', 'gpt-4-turbo']:
        response, used_token = query_openai(input=question_prompt, system_promt=system_prompt, visible=visible, model=model_name)
    elif model_name in ['deepseek']:
        response, used_token = query_deepseek(input=question_prompt, system_promt=system_prompt, visible=visible)
    elif model_name in ['deepseek-silicon']:
        response, used_token = query_silicon(input=question_prompt, system_promt=system_prompt, visible=visible)
    return question_prompt, response, used_token


def run_use_DSL_multirepo(args):
    repo = args.repo
    model_name = args.model_name
    Use_full_coverage = args.use_full_coverage
    SKE_mean_num = args.ske_mean_num
    difficulty_level = args.difficulty_level
    API_knowledge_dir = args.api_knowledge_dir
    prompt_template_dir = args.prompt_template_dir
    TOTAL_RUNTIMES = args.total_runtimes
    library_names = args.library_names
    generation_identifier = args.generation_identifier
    USE_SIMPLE = args.use_simple
    step0_template_num = args.step0_template_num

    # 是否涉及到了用built-in APIs => 如果既有builtin又有其他的, 可能还需要额外考虑
    use_builtin = True if 'Python' in library_names else False
    print('use_builtin is ', use_builtin)

    def APIchoosen(library_names = ['numpy',  'pandas' , 'matplotlib'] ):
        apis, apis2 = [], []
        for library_name in library_names:
            if library_name == 'numpy':
                numpybasic_paths = os.path.join(API_knowledge_dir, 'numpy/Numpy_basic.json')   
                apis.extend(read_apis_from_json(numpybasic_paths))  
                numpyadvanced_paths = os.path.join(API_knowledge_dir, 'numpy/Numpy_advanced.json')
                apis2.extend(read_apis_from_json(numpyadvanced_paths))  
            elif library_name == 'pandas':
                pandasbasic_paths = os.path.join(API_knowledge_dir, 'pandas/Pandas_basic.json')
                apis.extend(read_apis_from_json(pandasbasic_paths))
                pandasadvanced_paths = os.path.join(API_knowledge_dir, 'pandas/Pandas_advanced.json')
                apis2.extend(read_apis_from_json(pandasadvanced_paths))
            elif library_name == 'matplotlib':
                numpybasic_paths = os.path.join(API_knowledge_dir, 'matplotlib/matplotlib_basic.json')
                apis.extend(read_apis_from_json(numpybasic_paths))
                numpyadvanced_paths = os.path.join(API_knowledge_dir, 'matplotlib/matplotlib_advanced.json')
                apis2.extend(read_apis_from_json(numpyadvanced_paths))
            elif library_name == 'Python':
                numpybasic_paths = os.path.join(API_knowledge_dir, 'generalpython/Python_basic.json')
                apis.extend(read_apis_from_json(numpybasic_paths))
                numpyadvanced_paths = os.path.join(API_knowledge_dir, 'generalpython/Python_advanced.json')
                apis2.extend(read_apis_from_json(numpyadvanced_paths))
        API_list = [apis, apis2]
        return API_list


    for difficulty_level in [2]:
        if not use_builtin: 
            N_list = [5, 0, 0, 0] if difficulty_level==1 else [3,2,1,1]  

        else: 
            N_list = [5, 0, 0, 0] if difficulty_level==1 else [2,3,1,1] 
            N_list = [5, 0, 0, 0] if difficulty_level==1 else [3,2,1,1]  

        Length = 'normal' 
        APInumbers = sum(N_list[:difficulty_level]) 
        
        SKEleton_numbers = generate_list(mean = SKE_mean_num, std_dev = 1, size = TOTAL_RUNTIMES , lower_bound = SKE_mean_num-2, upper_bound = SKE_mean_num+2, mid_lower_bound = SKE_mean_num-1, mid_upper_bound = SKE_mean_num+1)
        API_list = APIchoosen(library_names = library_names)
        final_res = []


        basedir = f'./data/{repo}/{generation_identifier}/{model_name}' 
        if not os.path.exists(basedir):
            os.makedirs(basedir)


        question_benchmark_namefile = f'{repo}_level{difficulty_level}_SIMPLE{USE_SIMPLE}_{APInumbers}items_{Length}_{TOTAL_RUNTIMES}.json'
        question_benchmark_name = os.path.join(basedir, question_benchmark_namefile)
        question_benchmark_jsonl_namefile= f'{repo}_level{difficulty_level}_SIMPLE{USE_SIMPLE}_{APInumbers}items_{Length}_{TOTAL_RUNTIMES}.jsonl'
        question_benchmark_jsonl_name = os.path.join(basedir, question_benchmark_jsonl_namefile)

        

        startidx = 0
        if os.path.exists(question_benchmark_jsonl_name):
            with open(question_benchmark_jsonl_name, 'r') as f:
                lines = f.readlines()
                startidx = len(lines)

        for idx,i in tqdm(enumerate(range(startidx, TOTAL_RUNTIMES))):
            selected_apis = []
            code_skeleton,treesitter_code = '',''

            if not Use_full_coverage: 
                for i in range(difficulty_level):
                    selected_apis.extend(get_n_apis(API_list[i], N_list[i]))
            
            
            try:
                if not USE_SIMPLE:  # complex ones
                    print('SKEleton_numbers is ', SKEleton_numbers)
                    code_skeleton, treesitter_code = step1(N=SKEleton_numbers[i], random_stmt = '<Random Stmt>', need_grammar_check = True)
                    print('start step2--------------------------------\n\n')
                    question_prompt, response_step2, used_token = step2(generation_identifier, selected_apis, code_skeleton, model_name=model_name,prompt_template_dir=prompt_template_dir, template_num="2", Onerepo=False,use_builtin=use_builtin, difficulty_level=difficulty_level)
                    print('question_prompt is \n', question_prompt)
                    print('response_step2 is \n', response_step2)
                    print('start step3--------------------------------\n\n')
                    question_prompt, response_step3, used_token = step3(generation_identifier, code_skeleton= code_skeleton, psudo_code=response_step2, model_name=model_name,prompt_template_dir=prompt_template_dir,template_num="6",use_builtin=use_builtin)
                    print('question_prompt is \n', question_prompt)
                    print('response_step3 is \n', response_step3)
                    response_step3_nocomment = remove_comments(response_step3)
                    print('start step4--------------------------------\n\n')
                    question_prompt, response_step4, used_token = step4(generation_identifier, example_code = response_step3_nocomment, model_name=model_name, prompt_template_dir=prompt_template_dir, template_num="1")
                    response_step4 = response_step4.replace('\n\n', '\n')
                    print('question_prompt is \n', question_prompt)
                    print('\n\nresponse_step4 is \n', response_step4)
                elif USE_SIMPLE:  # simple ones only rely on API
                    print('start step0--------------------------------\n\n')
                    question_prompt, response_step4, used_token = step0(selected_apis, visible=False, model_name=model_name,prompt_template_dir=prompt_template_dir,template_num=step0_template_num)
            except Exception as e:
                print('error is ', e)
                time.sleep(10)
                continue 
            
            
            obj = {
                'skeleton_keywords_num':SKEleton_numbers[i],
                'apis': selected_apis,
                'question_prompt': question_prompt,
                'result': response_step4,
                'used_token': used_token,
                'difficulty_level': difficulty_level,
                'treesitter_code': treesitter_code,
                'code_skeleton': code_skeleton
            }
            with open(question_benchmark_jsonl_name, 'a', encoding='utf-8') as f:
                f.write(json.dumps(obj)+'\n')
        

        with open(question_benchmark_jsonl_name, 'r') as f:
            lines = f.readlines()
            final_res = [json.loads(line) for line in lines]
            print('writing to json file ', question_benchmark_name)
            with open(question_benchmark_name, 'w') as f:
                json.dump(final_res, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run DSL multi-repo script')
    parser.add_argument('--repo', type=str, default='PYSkeletonDSL', help='Repository name')
    parser.add_argument('--model_name', type=str, default='deepseek', help='Model name')
    parser.add_argument('--use_full_coverage', action='store_true', help='Use full coverage')
    parser.add_argument('--ske_mean_num', type=int, default=4, help='Skeleton mean number')
    parser.add_argument('--difficulty_level', type=int, default=2, choices=[1, 2], help='Difficulty level now only support 1,2')
    parser.add_argument('--api_knowledge_dir', type=str, default='./API_knowledge', help='API knowledge directory')
    parser.add_argument('--prompt_template_dir', type=str, default='./prompt_template_multi.json', help='Prompt template directory')
    parser.add_argument('--total_runtimes', type=int, default=3000, help='Total number of questions to generate')
    parser.add_argument('--library_names', nargs='+', default=['Python'], help='Library names')
    parser.add_argument('--generation_identifier', type=str, default='generalpython', help='Generation identifier')
    parser.add_argument('--use_simple', action='store_true', help='Use simple mode')
    parser.add_argument('--step0_template_num', type=str, default='3', help='Step 0 template number')

    args = parser.parse_args()
    run_use_DSL_multirepo(args)