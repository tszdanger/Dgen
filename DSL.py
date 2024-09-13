import random

# keywords: def if while for try except break continue elif else return

def init_indent_map():
    indent_map = {
        'if': [0],
        'elif': [None],
        'else': [None],
        'while': [0],
        'for': [0],
        'try': [0],
        'def': [0],
        'except': [None],
        'break': [None],
        'continue': [None],
        'return': [None],
    }
    return indent_map

def clear_other_higher_indent(indent_map, indent_level, keyword):
    for kw in ['if', 'while', 'for', 'try', 'def', 'except', 'else', 'elif', 'break', 'continue','return']:
        if keyword == kw:
            indent_map[kw].remove(indent_level)
        else:
            indent_map[kw] = [indent for indent in indent_map[kw] if indent is None or indent < indent_level]
    return indent_map


def update_indent_map(indent_map, keyword, indent_level):
    if keyword == 'def':
        for kw in ['if', 'while', 'for', 'try', 'def', 'return']:
            indent_map[kw] = [indent_level + 1]
    elif keyword == 'if':
        indent_map['elif'].append(indent_level)
        indent_map['else'].append(indent_level)     
        for kw in ['while', 'for', 'try', 'return']:
            if indent_level + 1 not in indent_map[kw]:
                indent_map[kw].append(indent_level + 1)
    elif keyword == 'elif': 
        pass
    elif keyword == 'else': 
        indent_map['else'].remove(indent_level)
    elif keyword in ['while', 'for']:
        for kw in ['if', 'while', 'for', 'try', 'def', 'return']:
            indent_map[kw].append(indent_level + 1)
        indent_map['continue'].append(indent_level+1)
        indent_map['break'].append(indent_level+1)
    elif keyword == 'try':
        indent_map['except'].append(indent_level)
        indent_map['else'].append(indent_level)
        for kw in ['if', 'while', 'for', 'try', 'def']:
            indent_map[kw].append(indent_level + 1)
    elif keyword == 'except': 
        indent_map['except'].remove(indent_level) 
    elif keyword in ['break', 'continue']: 
        indent_map = clear_other_higher_indent(indent_map, indent_level, keyword)
    elif keyword == 'return':
        indent_map = clear_other_higher_indent(indent_map, indent_level, keyword)
    
    for kw in indent_map.keys():
        indent_map[kw] = list(set(indent_map[kw]))
    
    

def generate_code_skeleton(keywords, stmt_list, random_stmt = 'a = 1',random_stmt_check='a = 1', need_grammar_check = False):
    template_code = [] # for code generation
    treesitter_code = []  # for grammar checking
    indent_level = 0
    indent_map = init_indent_map()
    
    for keyword in keywords:
        if keyword in indent_map:
            indent_choices = [indent for indent in indent_map[keyword] if indent is None or indent + indent_level >= 0]
            if indent_choices != [None] or indent_choices != []:
                try:
                    indent = random.choice(indent_choices)
                except IndexError:
                    raise ValueError(f"Invalid indentation level for keyword '{keyword}'")
                if indent is not None:
                    indent_level = indent
            else:
                raise ValueError(f"Invalid indentation level for keyword '{keyword}'")
            
            update_indent_map(indent_map, keyword, indent_level)
        else:
            indent = 0
        
        template_code.append(' ' * 4 * indent_level + keyword)
        treesitter_code.append(' ' * 4 * indent_level + keyword)

        if keyword == 'def':
            template_code[-1] += ' function_name():'
            treesitter_code[-1] += ' function_name():'
        elif keyword in ['if', 'elif']:
            template_code[-1] += ' <IF CONDITION>:'
            treesitter_code[-1] += ' True:'
        elif keyword in ['while']:
            template_code[-1] += ' <WHILE CONDITION>:' 
            treesitter_code[-1] += ' True:'
        elif keyword == 'for':
            template_code[-1] += ' <FOR ITERATION>:'
            treesitter_code[-1] += ' i in range(5):'
        elif keyword in ['try', 'else']:
            template_code[-1] += ':'
        elif keyword == 'except':
            template_code[-1] += ' <Exception Type>:'
            treesitter_code[-1] += ' Exception as e:'
        elif keyword == 'return':
            template_code[-1] += ' <RETURN VALUE>'
            treesitter_code[-1] += ' 1'


        stmt_num = stmt_list.pop(0)
        for _ in range(stmt_num):
            template_code.append(' ' * 4 * (indent_level + 1) + random_stmt)
            treesitter_code.append(' ' * 4 * (indent_level + 1) + random_stmt_check)

    template_code_str = '\n'.join(template_code)
    treesitter_code_str = '\n'.join(treesitter_code)
    if need_grammar_check:
        return template_code_str, treesitter_code_str
    else:
        return template_code_str, ''
    

class InvalidKeywordsError(Exception):
    pass

def generate_random_keywords(N):
    keywords = ['def']
    stmt_list = [random.randint(1, 2)]
    
    keyword_weights = {
        'if': 10,
        'while': 4,
        'for': 8,
        'elif': 6,
        'else': 4,
        'try': 2,
        'except': 2,
        'break': 2,
        'continue': 2,
        'return': 2
    }
    
    total_stmts = stmt_list[0]
    has_try = False
    
    for _ in range(N - 1):
        keyword = random.choices(list(keyword_weights.keys()), weights=list(keyword_weights.values()))[0]
        if keyword == 'try':
            has_try = True
        keywords.append(keyword)
        if keyword in ['break', 'continue', 'return']:
            stmt_num = 0
        else:
            stmt_num = random.randint(1, 2)
        stmt_list.append(stmt_num)
        
    
    if has_try and 'except' not in keywords:
        raise InvalidKeywordsError("'try' keyword present without 'except'")
    
    maxnumber_keywords = max([keywords.count(kw) for kw in keyword_weights.keys()])
    if maxnumber_keywords > N * 0.75:
        raise InvalidKeywordsError("Single keyword statements exceed 0.75 of total statements")
    
    return keywords, stmt_list

def generate_valid_code_skeleton(N, random_stmt = 'a = 1', need_grammar_check = False):
    random_stmt_check = 'a = 1' if need_grammar_check else random_stmt
    while True:
        try:
            keywords, stmt_list = generate_random_keywords(N)
            print('keywords:', keywords)
            print('stmt_list:', stmt_list)
            code_skeleton, treesitter_code = generate_code_skeleton(keywords, stmt_list, random_stmt = random_stmt, random_stmt_check = random_stmt_check, need_grammar_check = need_grammar_check)
            return code_skeleton, treesitter_code
        except ValueError:
            print('Invalid code skeleton, retrying...')
            continue
        except InvalidKeywordsError as e:
            print(e)
            print('Invalid code skeleton, retrying...')
            continue


def test1():
    keywords = ['def', 'elif', 'try']
    stmt_list = [2, 2, 1]
    code_skeleton, _ = generate_code_skeleton(keywords, stmt_list)
    print(code_skeleton)
    return code_skeleton

if __name__ == '__main__':
    code_skeleton, treesitter_code = generate_valid_code_skeleton(5, random_stmt = '<Random Statement>', need_grammar_check = True)
    print(code_skeleton)
    print('-'*20)
    print(treesitter_code)
