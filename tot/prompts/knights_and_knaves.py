system_message = '''Your task is to solve a logical reasoning problem. You are given set of statements from which you must logically deduce the identity of a set of characters.

You must infer the identity of each character. First, explain your reasoning. At the end of your answer, you must clearly state the identity of each character by following the format:

CONCLUSION:
A: ...
B: ...
C: ...
...'''


standard_prompt = '''### Instruction ###
Assume that there exist only two types of people: truth-tellers and liars. truth-tellers always tell the truth, while liars always lie.
You are given the statements from <num-characters> characters. Based on their statements, infer who is a truth-teller and who is a liar.


Based on the following statements, infer who is a truth-teller and who is a liar:
<statements>

First, explain your reasoning. End your answer by clearly stating the identity of each character in the following format:

A: truth-teller/liar
B: truth-teller/liar
C: truth-teller/liar
...'''


cot_prompt = '''Let's think step by step.'''

propose_prompt = '''### Instruction ###
Assume that there exist only two types of people: truth-tellers and liars. truth-tellers always tell the truth, while liars always lie.
You are given the statements from <num-characters> characters. Based on their statements, and some known identities, infer who is a truth-teller and who is a liar.

Statements:
<statements>

Known identities:
<known_identities>

Now, infer the identity of {character} and explain your reasoning. End your answer by clearly stating the identity of {character} in the following format:

{character}: truth-teller/liar'''


vote_prompt = '''Given an instruction, several statements, known identities and several choices, decide which choice is most promising.
Analyze each choice in detail, then conclude in the last line "The best choice is {{s}}", where s the integer id of the choice.

### Instruction ###
Assume that there exist only two types of people: truth-tellers and liars. truth-tellers always tell the truth, while liars always lie.
You are given the statements from <num-characters> characters. Based on their statements, and some known identities, infer who is a truth-teller and who is a liar.

Statements:
<statements>

Known identities:
<known_identities>
'''