import numpy as np
from tqdm import tqdm
from tot.tasks.knights_and_knaves import ENTITIES
from tot.models import Model
import re, random, itertools
from collections import OrderedDict
from typing import List

def get_votes(task, x, ys, known_identities, n_evaluate_sample):
    vote_prompt = task.vote_prompt_wrap(x, ys, known_identities)
    vote_outputs = model.inference(vote_prompt, num_return_sequences=n_evaluate_sample, stop=None)
    values = task.vote_outputs_unwrap(vote_outputs, len(ys))
    prompt_log = {
        'vote prompt': vote_prompt, 
        'vote outputs': vote_outputs,
        'votes': values,
    }
    return values, prompt_log

def get_proposals(task, x, y, n_generate_sample, step)->list:
    '''
    y: known identities
    '''
    global propose_prompt
    propose_prompt = task.propose_prompt_wrap(x, y, step)
    proposals = model.inference(propose_prompt, num_return_sequences=n_generate_sample, stop=None)
    return proposals

def get_samples(task, x, y, n_generate_sample, prompt_sample, stop):
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    samples = model.inference(prompt, num_return_sequences=n_generate_sample, stop=stop)
    return [y + _ for _ in samples],prompt

def get_oracle_generation(task, idx:int, step:int, n_generate_sample:int, oracle_prob:float)->list:
    '''
    ys: existing steps
    gs: saving intermediate expressions
    '''
    binary_list, generations, = [],[]
    solution = task.get_answer(idx)
    smap = {True: "truth-teller", False: "liar"}

    for i in range(n_generate_sample):
        if random.random() <= oracle_prob:
            binary_list.append(1) # positiv sample
        else:
            binary_list.append(0) # negative sample

    for j in binary_list:
        if j: # positiv sample
            generations.append(f"{ENTITIES[step]}: {smap[solution[ENTITIES[step]]]}")
        else: # negative sample
            generations.append(f"{ENTITIES[step]}: {smap[not solution[ENTITIES[step]]]}")

    return generations
            
def get_oracle_discrimination(task, idx:int, step:int, ys:list, n_select_sample, oracle_prob:float)->list:
    select_new_ys, binary_list = [], []
    solution = task.get_answer(idx)
    smap = {True: "truth-teller", False: "liar"}
    try:
        ys_check = [
            extract_identity(y, step).split()[-1] == smap[solution[ENTITIES[step]]] 
            if y and extract_identity(y, step) and len(extract_identity(y, step).split()) > 0 
            else False for y in ys
        ]
    except IndexError as i:
        print(i)
        print(ys)
        print(solution)

    for _ in range(n_select_sample):
        if random.random() <= oracle_prob:
            binary_list.append(1) # positiv sample
        else:
            binary_list.append(0) # negative sample

    for j in binary_list:
        if j and True in ys_check: # positiv sample
            select_new_ys.append(random.choice([val for condition, val in zip(ys_check, ys) if condition]))
        elif not j and False in ys_check: # negative sample
            select_new_ys.append(random.choice([val for condition, val in zip(ys_check, ys) if not condition]))
            
    if len(select_new_ys) == 0:
        select_new_ys = "System Information: No promising thoughts available."

    return select_new_ys

def extract_identity(text: str, step:int)-> str:
    pattern = r"\**[A-F]\**:\** ?\**(truth-teller|liar)\**(?:/[truth\-tellerliar]+)?(?: *= *(truth-teller|liar))?"
    matches = re.findall(pattern, text)
    result = set()
    for match in matches:
        result.add(f"{ENTITIES[step]}: {match[-1] if match[-1] else match[0]}\n")
    return result.pop() if result else text

def solve(global_model, args, task, idx, to_print=True):
    global model
    model = global_model
    x = task.get_input(idx)  # input
    print(x)
    ys = ['']  # current output candidates
    infos = []
    for step in tqdm(range(task.steps), desc='steps'):
        # generation
        if args.method_generate == 'sample':
            new_ys = [get_samples(task, x, y, args.n_generate_sample, prompt_sample=args.prompt_sample, stop=task.stops[step]) for y in ys]
            new_ys = list(itertools.chain(*new_ys))
        elif args.method_generate == 'propose':
            new_ys = [get_proposals(task, x, y, args.n_generate_sample, step) for y in ys]
            new_ys = list(itertools.chain(*new_ys))
        elif args.method_generate == 'oracle':
            new_ys = get_oracle_generation(task, idx, step, args.n_generate_sample, oracle_prob=args.oracle_prob)

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
            values, vprompt = get_votes(task, x, new_ys, ys[0], args.n_evaluate_sample)
        elif args.method_evaluate == 'oracle':  # oracle discriminator
            select_new_ys = get_oracle_discrimination(task, idx, step, new_ys, args.n_select_sample, args.oracle_prob)
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
            select_new_ys = [extract_identity(y, step) for y in select_new_ys]
            select_new_ys = [ys[0] + select_new_ys[0]]
        elif args.method_select == 'oracle' or args.method_select == 'random':
            if isinstance(select_new_ys, list):
                select_new_ys = [extract_identity(y, step) for y in select_new_ys]
                select_new_ys = [ys[0] + select_new_ys[0]]
            

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
    x = task.get_input(idx)
    ys, prompt = get_samples(task, x, '', args.n_generate_sample, args.prompt_sample, stop=None)
    return ys, OrderedDict({
        'x': x, 
        'prompt':prompt,
        })