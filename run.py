import os, json, argparse
from tqdm import tqdm

from tot.tasks import get_task
from tot.tasks.knights_and_knaves import Knights_and_Knaves_Task
from tot.models import Model

def run(args):
    task = get_task(args.task, args)
    if isinstance(task, Knights_and_Knaves_Task):
        from tot.methods.bfs_kk import solve, naive_solve
    else:
        from tot.methods.bfs_24 import solve, naive_solve
    logs, successful_task = [], []
    cnt_avg, cnt_any = 0, 0
    if args.naive_run:
        file = (
            f'./logs/{args.task}_{args.n_char}/{args.backend}_{args.temperature}_naive_{args.prompt_sample}_sample_{args.n_generate_sample}_'
            f'start{args.task_start_index}_end{args.task_end_index}.json' 
            if args.task == 'knights_and_knaves' else
            f'./logs/{args.task}/{args.backend}_{args.temperature}_naive_{args.prompt_sample}_sample_{args.n_generate_sample}_'
            f'start{args.task_start_index}_end{args.task_end_index}.json'
        )
    else:
        file = (
            f'./logs/{args.task}_{args.n_char}/{args.backend}_{args.temperature}_{args.method_generate}{args.n_generate_sample}_'
            f'{args.method_evaluate}{args.n_evaluate_sample}_{args.method_select}{args.n_select_sample}_prob{args.oracle_prob}_'
            f'start{args.task_start_index}_end{args.task_end_index}.json'
            if args.task == 'knights_and_knaves' else
            f'./logs/{args.task}/{args.backend}_{args.temperature}_{args.method_generate}{args.n_generate_sample}_'
            f'{args.method_evaluate}{args.n_evaluate_sample}_{args.method_select}{args.n_select_sample}_prob{args.oracle_prob}_'
            f'start{args.task_start_index}_end{args.task_end_index}.json'
        )
    os.makedirs(os.path.dirname(file), exist_ok=True)
    # global model
    model = Model(model_id=args.backend, temperature=args.temperature)

    for i in tqdm(range(args.task_start_index, args.task_end_index), desc='task index'):
        # solve
        if args.naive_run:
            ys, info = naive_solve(model, args, task, i) 
        else:
            ys, info = solve(model, args, task, i)

        # log
        scores = [task.test_output(i, y) for y in ys]

        # log main metric
        accs = [score['r'] for score in scores]
        cnt_avg += sum(accs) / len(accs)
        cnt_any += any(accs)
        if any(accs):
            successful_task.append(i)
        print(f"task index: {i}, sum(accs)={sum(accs)}, cnt_avg={cnt_avg}, cnt_any={cnt_any}\n")
        info.update({ 
            'task idx': i,
            'outputs': ys,
            'solution': task.get_answer(i) if isinstance(task, Knights_and_Knaves_Task) else None,
            'scores': scores,
            'sum(accs)': sum(accs), 
            'cnt_avg': cnt_avg, 
            'cnt_any': cnt_any,})
        info.move_to_end('task idx', last=False)
        logs.append(info)
        with open(file, 'w') as f:
            json.dump(logs, f, indent=4)
    
    n = args.task_end_index - args.task_start_index
    print(cnt_avg / n, cnt_any / n)
    logs.append({
        'cnt_avg / n': round(cnt_avg / n, 4), 
        'cnt_any / n': round(cnt_any / n, 4),
        'successful task': successful_task,})
    with open(file, 'w') as f:
        json.dump(logs, f, indent=4)


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--backend', type=str, choices=['google/gemma-2-27b-it',
                                                      'meta-llama/Meta-Llama-3.1-70B-Instruct',
                                                      'meta-llama/Meta-Llama-3.1-8B-Instruct'], 
                                             default='meta-llama/Meta-Llama-3.1-8B-Instruct')
    args.add_argument('--temperature', type=float, default=1.0)

    args.add_argument('--task', type=str, required=True, choices=['game24','knights_and_knaves'],default='game24')
    args.add_argument('--task_start_index', type=int, default=0)
    args.add_argument('--task_end_index', type=int, default=1362)

    args.add_argument('--naive_run', action='store_true')
    args.add_argument('--prompt_sample', type=str, choices=['standard', 'cot'])  # only used when method_generate = sample, or naive_run

    args.add_argument('--method_generate', type=str, choices=['sample', 'propose', 'oracle'],default='propose')
    args.add_argument('--method_evaluate', type=str, choices=['value', 'vote', 'oracle', 'random'], default='value')
    args.add_argument('--method_select', type=str, choices=['sample', 'greedy', 'oracle', 'random'], default='greedy')
    args.add_argument('--n_generate_sample', type=int, default=1)  # only thing needed if naive_run
    args.add_argument('--n_evaluate_sample', type=int, default=1)
    args.add_argument('--n_select_sample', type=int, default=1)
    args.add_argument('--oracle_prob', type=float, default=1.0)
    args.add_argument('--n_char', type=int, default=3) # only used when task is 'knights_and_knaves'

    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    run(args)