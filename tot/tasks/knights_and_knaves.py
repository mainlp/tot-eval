import re, logging, json, logging
from tot.tasks.base import Task, DATA_PATH
from tot.prompts.knights_and_knaves import *
from datasets import Dataset
from typing import Any, Dict, List

# Get the existing logger
logger = logging.getLogger(__name__)
ENTITIES = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
STANDARD_PROMPT = standard_prompt
COT_PROMPT = cot_prompt
PROPOSE_PROMPT = propose_prompt
VOTE_PROMPT = vote_prompt

# @staticmethod
def rule_based_evaluation(
    model_reponse: str, 
    num_characters: int = 3, 
    context: dict[str, str] = {"truth-teller": "truth-teller", "liar": "liar"}, 
) -> dict[str, bool | None]:
    """
    Extract final conclusion from model's response.

    Args:
        model_reponse (str): The response of the model.
        context (dict[str, str]): The context, i.e. descriptions for truthteller and liars.
        num_characters (int): The number of characters considered in the problem statetement.

    Returns:
        dict[str, bool]: The extract final conclusion. Could be empty if no unique conclusion could be found.
    """
    entities = f": ({context['truth-teller']}|{context['liar']})\n".join(
        ENTITIES[:num_characters]
    )
    pattern = f"^{entities.lower()}: ({context['truth-teller']}|{context['liar']})\s*$"  # noqa: W605
    matches = re.findall(pattern, model_reponse.lower(), re.MULTILINE)

    if not matches:
        return {}

    # If more than one match is found, check if they are all the same
    if len(matches) > 1:
        if len(set(matches)) != 1:
            return {}

    context_mapping = {context["truth-teller"]: True, context["liar"]: False}

    extraction: dict[str, bool | None] = {
        ENTITIES[character_id]: context_mapping[identity]
        for character_id, identity in enumerate(matches[0])
    }
    return extraction

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Load a JSON Lines (JSONL) file and return a list of dictionaries.

    This function includes error handling to log messages if the file doesn't exist or if a line contains invalid JSON.

    Args:
        file_path (str): The path to the JSONL file to be loaded.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary
        represents a JSON object from a single line of the JSONL file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        json.JSONDecodeError: If a line in the file is not valid JSON.
    """
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                try:
                    json_line = json.loads(line.strip())
                    data.append(json_line)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON at line {line_number} in {file_path}")
                    raise
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise

    return data

def prepare_dataset_from_disk(num_characters: int) -> Any:
    """
    Loads the dataset from disk (jsonl file), and converts it into a HF Dataset.
    Each character contains 600 tasks.

    Args:
        file_path (str): Path to dataset.

    Returns:
        Any: The prepared dataset.
    """
    file_path = f'tot/data/knights_and_knaves/characters_{num_characters}/puzzles.jsonl'
    # Load data from disk and preprocess dataset
    if file_path.endswith(".jsonl"):
        dataset = load_jsonl(file_path)
        transformed_data = {key: [dic[key] for dic in dataset] for key in dataset[0]}
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    dataset = Dataset.from_dict(transformed_data)
    return dataset

class Knights_and_Knaves_Task(Task):
    """
    Input (x)   : 
    Output (y)  :
    Reward (r)  : 
    Input Example: 
    Output Example: 
    """
    def __init__(self, num_characters=3):
        super().__init__()
        self.data = prepare_dataset_from_disk(num_characters)
        self.steps = num_characters
        self.num_characters = num_characters

    def __len__(self) -> int:
        return len(self.data)

    def get_input(self, idx: int) -> str:
        return self.data[idx]["problem"]

    def get_answer(self, idx: int) -> str:
        return self.data[idx]['solutions'][0]

    def test_output(self, idx: int, output: str) -> dict:
        pred = rule_based_evaluation(output, self.num_characters)
        ground_truth = self.get_answer(idx)
        if all(
            ground_truth.get(character) == pred.get(character)
            for character in ground_truth
        ):
            return {'r': 1}
        else:
            return {'r': 0}


    def standard_prompt_wrap(self, x: str, y) -> str:
        standard_prompt = STANDARD_PROMPT
        substitution_dict: dict[str, str] = {
            "<num-characters>": str(self.num_characters),
            "<statements>": "<statements>",
            "truth-teller": "truth-teller",
            "liar": "liar",
        }
        substitution_dict["<statements>"] = (
                "\n".join(x) if isinstance(x, list) else x
            )
        for special_token, sub_txt in substitution_dict.items():
            standard_prompt = standard_prompt.replace(special_token, sub_txt)

        return system_message + '\n\n' + standard_prompt


    def cot_prompt_wrap(self, x: str, y) -> str:
        prompt = STANDARD_PROMPT + "\n" + COT_PROMPT
        substitution_dict: dict[str, str] = {
            "<num-characters>": str(self.num_characters),
            "<statements>": "<statements>",
            "truth-teller": "truth-teller",
            "liar": "liar",
        }
        substitution_dict["<statements>"] = (
                "\n".join(x) if isinstance(x, list) else x
            )
        for special_token, sub_txt in substitution_dict.items():
            prompt = prompt.replace(special_token, sub_txt)

        return system_message + '\n\n' + prompt
    

    def propose_prompt_wrap(self, x: str, known_identities: str = '', step: int = None) -> str:
        propose_prompt = PROPOSE_PROMPT.format(character=ENTITIES[step])
        substitution_dict: dict[str, str] = {
            "<num-characters>": str(self.num_characters),
            "<statements>": "<statements>",
            "truth-teller": "truth-teller",
            "liar": "liar",
            "<known_identities>": known_identities,
        }
        substitution_dict["<statements>"] = (
                "\n".join(x) if isinstance(x, list) else x
            )
        for special_token, sub_txt in substitution_dict.items():
            propose_prompt = propose_prompt.replace(special_token, sub_txt)

        return propose_prompt


    def vote_prompt_wrap(self, x: str, ys: List[str], known_identities: str = '') -> str:
        '''
        ys: the proposal of generator, should contain the reason here
        '''
        vote_prompt = VOTE_PROMPT
        substitution_dict: dict[str, str] = {
            "<num-characters>": str(self.num_characters),
            "<statements>": "<statements>",  #?
            "truth-teller": "truth-teller",
            "liar": "liar",
            "<known_identities>": known_identities,
        }
        substitution_dict["<statements>"] = (
                "\n".join(x) if isinstance(x, list) else x
            )
        for special_token, sub_txt in substitution_dict.items():
            vote_prompt = vote_prompt.replace(special_token, sub_txt)

        for i, y in enumerate(ys, 1):
            vote_prompt += f'Choice {i}:\n{y}\n'
        return vote_prompt
    
    @staticmethod
    def vote_outputs_unwrap(vote_outputs: list, n_candidates: int) -> list:
        vote_results = [0] * n_candidates
        for vote_output in vote_outputs:
            pattern = r".*best choice is .*(\d+).*"
            match = re.match(pattern, vote_output, re.DOTALL)
            if match:
                vote = int(match.groups()[0]) - 1
                if vote in range(n_candidates):
                    vote_results[vote] += 1
            else:
                print(f'vote no match: {[vote_output]}')
        return vote_results
