import numpy as np
from tqdm import tqdm
from tot.tasks.game24 import Game24Task, get_current_numbers
from tot.tasks.knights_and_knaves import Knights_and_Knaves_Task
from tot.models import Model
import re, sympy, random, math, itertools
from collections import Counter, OrderedDict
from typing import List


def solve_24_game(numbers: List[float], target: float = 24) -> List[str]:
    # "oracle generator"
    if not numbers:
        return []
    solutions = []
    epsilon = 1e-6  # Tolerance for floating point comparisons

    def dfs(nums: List[float], steps: List[str]):
        if len(nums) == 1:
            if math.isclose(nums[0], target, abs_tol=epsilon):
                solutions.append(steps[::])
            return

        for i in range(len(nums)):
            for j in range(len(nums)):
                if i == j:
                    continue
                a, b = nums[i], nums[j]
                next_nums = [nums[k] for k in range(len(nums)) if k != i and k != j]
                
                operations = [
                    (a + b, f"{a} + {b}"),
                    (a - b, f"{a} - {b}"),
                    (b - a, f"{b} - {a}"),
                    (a * b, f"{a} * {b}")
                ]

                if abs(b) > epsilon:
                    operations.append((a / b, f"{a} / {b}"))
                if abs(a) > epsilon:
                    operations.append((b / a, f"{b} / {a}"))

                for result, expr in operations:
                    next_steps = steps + [expr]
                    dfs(next_nums + [result], next_steps)

    dfs(numbers, [])
    return solutions

def split_at_nth_newline(s, n):
    parts = s.split('\n')
    # Join the parts before and after the nth newline
    return '\n'.join(parts[:n]), '\n'.join(parts[n:])

def get_value(task, x, y, n_evaluate_sample, cache_value=True):
    value_prompt, max_tokens = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt], {}
    value_outputs = model.inference(value_prompt, max_tokens=max_tokens, num_return_sequences=n_evaluate_sample, stop=None)
    value = task.value_outputs_unwrap(x, y, value_outputs)
    p = {'value prompt': value_prompt, 
         'value output': value_outputs,
         'value': value,
         }
    if cache_value:
        task.value_cache[value_prompt] = value
    return value, p

def get_values(task, x, ys, n_evaluate_sample, cache_value=True, current_step=None):
    values, prompt_log = [], []
    local_value_cache = {}
    if isinstance(task, Game24Task):
        if current_step < 3:
            for y in ys:
                try:
                    if ('left: ' not in y.split('\n')[-2]) or (y in local_value_cache):
                        value = 0
                    else:    
                        value, p = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
                        local_value_cache[y] = value
                        prompt_log.append(p)
                except Exception as e:
                    print(e)
                values.append(value)
                
        else:
            for y in ys:
                if ('Answer:' not in y) or (y in local_value_cache):
                    value = 0
                else:    
                    value, p = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
                    local_value_cache[y] = value
                    prompt_log.append(p)
                values.append(value)
    else:
        for y in ys:
            if y in local_value_cache:
                value = 0
            else:    
                value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
                local_value_cache[y] = value
            values.append(value)
    return values, prompt_log

def get_votes(task, x, ys, n_evaluate_sample):
    vote_prompt = task.vote_prompt_wrap(x, ys)
    vote_outputs = model.inference(vote_prompt, num_return_sequences=n_evaluate_sample, stop=None)
    values = task.vote_outputs_unwrap(vote_outputs, len(ys))
    prompt_log = {'vote prompt': vote_prompt, 
                'vote outputs': vote_outputs,
                'votes': values,
                }
    return values, prompt_log

def test_output(x: str, output: str)-> int:
    # return 1 if output is wrong else 0
    expression = output.strip().split('\n')[-1].lower().replace('answer: ', '').split('=')[0]
    numbers = re.findall(r'\d+', expression)
    problem_numbers = re.findall(r'\d+', x)
    if sorted(numbers) != sorted(problem_numbers):
        return 1
    try:
        return int(sympy.simplify(expression) != 24)
    except Exception as e:
        return 1
    
def get_proposals(task, x, y, current_step=None)->list: 
    global propose_prompt
    propose_prompt = task.propose_prompt_wrap(x, y)
    generations = []
    while not generations:
        if current_step == 3:  # last step
            proposals = model.inference(propose_prompt, num_return_sequences=10, stop=None)
            p = list(itertools.chain(*[a.split('\n') for a in proposals]))
            generations = [y + _ + '\n' for _ in p if _ and 'answer' in _.lower()]
        # drop bad proposals
        elif current_step < 2:
            proposals = model.inference(propose_prompt, num_return_sequences=1, stop=None)[0].split('\n')
            for _ in proposals:
                if _ and 'left: ' in _:
                    current_numbers = get_current_numbers(_)
                    numbers = [float(num) if '.' in num else int(num) for num in re.findall(r'\b\d+\.\d+|\b\d+\b', current_numbers)]
                    if len(numbers) + current_step == 3:
                        generations.append(y + _ + '\n')
        else: #current_step == 2: left 1 number
            proposals = model.inference(propose_prompt, num_return_sequences=1, stop=None)[0].split('\n')
            for _ in proposals:
                if _ and 'left: ' in _:
                    current_numbers = get_current_numbers(_)
                    if current_numbers == '24':
                        generations.append(y + _ + '\n')
            return generations    
    return generations

def get_samples(task, x, y, n_generate_sample, prompt_sample, stop):
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    samples = model.inference(prompt, num_return_sequences=n_generate_sample, stop=stop)
    return [y + _ for _ in samples],prompt

def string_to_list(s)->list:
    lines = s.strip().split('\n')
    expressions = []

    for line in lines:
        expr = re.findall(r"(?:\d+\.\s|\*\s)?(.+?)\s=", line)[-1].replace('×','*').replace('÷','/')
        expressions.append(expr)

    return expressions
    
def list_to_string(x:list, expressions:list)->str:
    output_lines = []
    
    for expr in expressions:
        try:
            result = eval(expr)
            cnt_remove = 2
            x.remove(int(expr[0]))
            cnt_remove -= 1
            x.remove(int(expr[-1]))
            cnt_remove -= 1
        except ValueError:
            while cnt_remove > 0:
                x.remove(random.choice(x))
                cnt_remove -= 1
        except TypeError:
            try:
                result = eval(expr[0])
                cnt_remove = 2
                x.remove(int(expr[0][0]))
                cnt_remove -= 1
                x.remove(int(expr[0][-1]))
                cnt_remove -= 1
            except Exception as e:
                while cnt_remove > 0:
                    x.remove(random.choice(x))
                    cnt_remove -= 1
        except SyntaxError as s:
            print(s)

        x.append(result)

        left_str = " ".join(map(str, x))
        output_line = f"{expr} = {result} (left: {left_str})"
        output_lines.append(output_line)

    return "\n".join(output_lines)

def find_elements_after_sublist(main_list, sublist)->set:
    results = set()
    for item in main_list:
        if item[:len(sublist)] == sublist and len(item) > len(sublist):
            results.add(item[len(sublist)])
    return results

def split_generation(x, ys, current_step):
    '''
    split the generation into romising and unpomising
    return 2 sets
    '''
    positive, negative = set(), set()
    solutions = solve_24_game(list(map(int, x.split())))

    if current_step == 3:  # last step
        for g in ys: 
            if g and 'answer' in g.lower() and not test_output(x, g):
                positive.add(g)
            else:
                negative.add(g)
    else: # current_step < 3:
        for y in ys:
            y_l, g = split_at_nth_newline(y, -2)
            if g and 'left: ' in g:                
                try:
                    # extract expression left of = in new generation 
                    expression = re.findall(r"(?:\d+\.\s|\*\s|-\s)?(.+?)\s=", g)[-1].replace('×','*').replace('÷','/')
                    if current_step == 0:
                        state = [float(num) if '.' in num else int(num) for num in re.findall(r"(?:\d+\.\s)?(-?\d+\.?\d*)", x)]
                    else: 
                        state = [float(num) if '.' in num else int(num) for num in re.findall(r"(?:\d+\.\s)?(-?\d+\.?\d*)", get_current_numbers(y_l))]
                    numbers_in_g = [float(num) if '.' in num else int(num) for num in re.findall(r"(?:\d+\.\s)?(-?\d+\.?\d*)", g)]
                    state.remove(numbers_in_g[0])
                    state.remove(numbers_in_g[1])
                    state.append(numbers_in_g[2])
                    # valid generation
                    if Counter(state) == Counter(numbers_in_g[3:]) and \
                        abs(numbers_in_g[2] - sympy.simplify(expression)) <= 0.01 and \
                        any(solution[current_step] == expression for solution in solutions):
                        if current_step == 2:
                            current_numbers = get_current_numbers(g)
                            if current_numbers == '24':
                                positive.add(y)
                            else:
                                negative.add(y)
                        else:
                            positive.add(y)
                    else:
                        negative.add(y)
                except Exception as e:
                    continue
    return positive, negative

def get_expr_sample()->str:
    ops = '+-*/' 
    return f"{random.randint(0,13)} {ops[random.randint(0,3)]} {random.randint(1,13)}"

def merge_expressions(original_str):
    # Split the original string by lines
    lines = original_str.strip().split('\n')

    # Parse equations and store the mappings of results to expressions
    expressions = []
    for line in lines:
        lhs, rhs = line.split('=')[:2]
        lhs = lhs.strip()
        rhs_value = rhs.strip().split()[0]  # Only take the numerical value on the right-hand side
        expressions.append((lhs, rhs_value))

    # Start with the last equation and work backwards
    final_lhs, final_rhs = expressions[-1]  # Take the last equation
    current_expr = final_lhs

    # Replace numbers in the final expression with corresponding expressions from previous steps
    for lhs, rhs_value in reversed(expressions[:-1]):  # Skip the last equation
        if rhs_value in current_expr:
            # Replace the right-hand value with the corresponding left-hand expression
            current_expr = current_expr.replace(rhs_value, f"({lhs})")

    # The final expression will be reconstructed from the third equation with substitutions
    result = f"{current_expr} = {final_rhs}\n"
    return  original_str + "Answer: " + result

def get_oracle_generation(task, x:str, ys:list, n_generate_sample:int, oracle_prob:float, current_step=None)->list:
    '''
    ys: existing steps
    gs: saving intermediate expressions
    '''
    binary_list, generations, gs = [],[],[]
    solutions = solve_24_game(list(map(int, x.split())))

    if current_step > 0:
        positive = []
        if current_step == 3:
            for y in ys:
                last_number_str = re.findall(r'-?\d+\.?\d*', y)[-1]
                last_number = float(last_number_str) if '.' in last_number_str else int(last_number_str)
                if last_number == 24:
                   positive.append(merge_expressions(y))
        else:
            for y in ys:
                y_l = string_to_list(y)
                ps = find_elements_after_sublist(solutions, y_l)
                positive.extend([*y_l, elem] for elem in ps)
        if not positive:
            return generations

    for i in range(n_generate_sample):
        if random.random() <= oracle_prob:
            binary_list.append(1) # positiv sample
        else:
            binary_list.append(0) # negative sample

    for j in binary_list:
        if j: # positiv sample
            if current_step == 0:
                gs.append([random.choice(solutions)[current_step]])
            elif current_step < 3:
                try:
                    gs.append(random.choice(positive))
                except Exception as e:
                    print(e)
            else: # last step
                generations.append(random.choice(positive))

        else: # negative sample
            n = get_expr_sample()
            if current_step == 0:
                while n in [s[0] for s in solutions]:
                    n = get_expr_sample()
                gs.append([n])
            elif current_step < 3:
                while n in [l[-1] for l in positive]:
                    n = get_expr_sample()
                gs.append([*string_to_list(random.choice(ys)), n])
            else: # last step
                generations.append(random.choice(ys)+'Answer: \n')
                
    for g in gs:
        generations.append(list_to_string(x=list(map(int, x.split())), expressions=g)+'\n')

    return generations

def get_oracle_discrimination(x, ys, n_select_sample, oracle_prob=1.0, current_step=None)->list:
    positive, negative = split_generation(x, ys, current_step)
    select_new_ys, binary_list = [], []
    for i in range(n_select_sample):
        if random.random() <= oracle_prob:
            binary_list.append(1) # positiv sample
        else:
            binary_list.append(0) # negative sample

    for j in binary_list:
        if j and positive:
            select_new_ys.append(positive.pop())
        elif not j and negative:
            select_new_ys.append(negative.pop())

    if len(select_new_ys) == 0:
        select_new_ys = "System Information: No promising thoughts available."

    return select_new_ys


def solve(global_model, args, task, idx, to_print=True):
    global model
    model = global_model
    x = task.get_input(idx)
    print(x)
    ys = ['']  # current output candidates
    infos = []
    for step in tqdm(range(task.steps), desc='steps'):
        # generation
        if args.method_generate == 'sample':
            new_ys = [get_samples(task, x, y, args.n_generate_sample, prompt_sample=args.prompt_sample, stop=task.stops[step]) for y in ys]
            new_ys = list(itertools.chain(*new_ys))
        elif args.method_generate == 'propose':
            new_ys = [get_proposals(task, x, y, current_step=step) for y in ys]
            new_ys = list(itertools.chain(*new_ys))
        elif args.method_generate == 'oracle':
            new_ys = get_oracle_generation(task, x, ys, args.n_generate_sample, oracle_prob=args.oracle_prob, current_step=step)

        if not new_ys:
            infos.append({'step': step, 
                      'x': x, 
                      'ys': ys, 
                      'prompt':propose_prompt if args.method_generate != 'oracle' else None, 
                      'new generations': 'System Information: No valid new generation.',})
            break
        random.shuffle(new_ys)
        ids = list(range(len(new_ys)))

        # evaluation
        if args.method_evaluate == 'vote':
            values, vprompt = get_votes(task, x, new_ys, args.n_evaluate_sample)
        elif args.method_evaluate == 'value':
            values, vprompt = get_values(task, x, new_ys, args.n_evaluate_sample, current_step=step)
        elif args.method_evaluate == 'oracle':  # oracle discriminator
            select_new_ys = get_oracle_discrimination(x, new_ys, 
                                                      n_select_sample=args.n_select_sample, 
                                                      oracle_prob=args.oracle_prob, 
                                                      current_step=step)
        elif args.method_evaluate == 'random':  # random oracle discriminator
            select_new_ys = random.sample(new_ys, args.n_select_sample) if len(new_ys) >= args.n_select_sample else new_ys

        # selection
        if args.method_select == 'sample':
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
            select_new_ys = [new_ys[select_id] for select_id in select_ids]
        elif args.method_select == 'greedy':
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:args.n_select_sample]
            select_new_ys = [new_ys[select_id] for select_id in select_ids]
        elif args.method_select == 'oracle' or args.method_select == 'random':
            pass

        # log
        if to_print and args.method_select != 'oracle' and args.method_select != 'random': 
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')
        elif to_print:
            print(f'-- choices --: {select_new_ys}\n')
        
        infos.append({'step': step, 
                      'x': x, 
                      'ys': ys, 
                      'prompt':propose_prompt if args.method_generate != 'oracle' else None,
                      'new generations': new_ys, 
                      'evaluation': vprompt if args.method_evaluate != 'oracle' and args.method_evaluate != 'random' else None,
                      'values': values if args.method_evaluate != 'oracle' and args.method_select != 'random' else None, 
                      'select_new_ys': select_new_ys})
        if isinstance(select_new_ys, str):
            break
        ys = select_new_ys
    
    if to_print: 
        print(ys)
    return ys, OrderedDict({'steps': infos})

def naive_solve(global_model, args, task, idx, to_print=True):
    global model
    model = global_model
    x = task.get_input(idx)  # input
    ys, prompt = get_samples(task, x, '', args.n_generate_sample, args.prompt_sample, stop=None)
    return ys, OrderedDict({ 
        'x': x, 
        'solution': task.get_answer(idx) if isinstance(task, Knights_and_Knaves_Task) else None,
        'prompt':prompt,
        })