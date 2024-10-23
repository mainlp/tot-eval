# Understanding When Tree of Thoughts Succeeds: Larger Models Excel in Generation, Not Discrimination
Official implementation for paper [Understanding When Tree of Thoughts Succeeds: Larger Models Excel in Generation, Not Discrimination]() with code, prompts and datasets.

## Setup
The `enviroment.yaml` file contains the required conda environment.

## Code
The dataset, prompts and task implementstations can  be found under `tot/data/`, `tot/prompts/` and `tot/tasks/` respectively.<br>  
`tot/models.py` provides LLMs as generator/discriminator modules for ToT.<br>  
The code to execute ToT and the oracle implementation is under `tot/methods/` directory. Since the oracle implementation relies on standard answers for different tasks, we tailor an implementation for different tasks.<br>
To experiment, run the `run.py` file with the required arguments.

## Citation
