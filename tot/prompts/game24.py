# 5-shot
standard_prompt = '''Use numbers and basic arithmetic operations (+ - * /) to obtain 24.
Input: 2 9 10 12
Answer: 2 * 12 * (10 - 9) = 24
Input: 4 9 10 13
Answer: (13 - 9) * (10 - 4) = 24
Input: 1 4 8 8
Answer: (8 / 4 + 1) * 8 = 24
Input: 5 5 5 9
Answer: 5 + 5 + 5 + 9 = 24
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) = 24
Input: {input}
'''

# 5-shot
cot_prompt = '''Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Each step, you are only allowed to choose two of the remaining numbers to obtain a new number.
Input: 4 4 6 8
Steps:
4 + 8 = 12 (left: 4 6 12)
6 - 4 = 2 (left: 2 12)
2 * 12 = 24 (left: 24)
Answer: (6 - 4) * (4 + 8) = 24
Input: 2 9 10 12
Steps:
12 * 2 = 24 (left: 9 10 24)
10 - 9 = 1 (left: 1 24)
24 * 1 = 24 (left: 24)
Answer: (12 * 2) * (10 - 9) = 24
Input: 4 9 10 13
Steps:
13 - 10 = 3 (left: 3 4 9)
9 - 3 = 6 (left: 4 6)
4 * 6 = 24 (left: 24)
Answer: 4 * (9 - (13 - 10)) = 24
Input: 1 4 8 8
Steps:
8 / 4 = 2 (left: 1 2 8)
1 + 2 = 3 (left: 3 8)
3 * 8 = 24 (left: 24)
Answer: (1 + 8 / 4) * 8 = 24
Input: 5 5 5 9
Steps:
5 + 5 = 10 (left: 5 9 10)
10 + 5 = 15 (left: 9 15)
15 + 9 = 24 (left: 24)
Answer: ((5 + 5) + 5) + 9 = 24
Input: {input}
'''

merge_prompt = '''Given three calculation steps. Follow the examples and combine the three calculation steps into one equation, but do not simplify. Your output should be in this format "Answer: {combined one equation}"
Examples:
Steps:
8 / 4 = 2 (left: 1 2 8)
1 + 2 = 3 (left: 3 8)
3 * 8 = 24 (left: 24)
Answer: (1 + 8 / 4) * 8 = 24
Steps:
12 * 2 = 24 (left: 9 10 24)
10 - 9 = 1 (left: 1 24)
24 * 1 = 24 (left: 24)
Answer: (12 * 2) * (10 - 9) = 24
Steps:
4 + 8 = 12 (left: 4 6 12)
6 - 4 = 2 (left: 2 12)
2 * 12 = 24 (left: 24)
Answer: (6 - 4) * (4 + 8) = 24
Steps:
5 + 5 = 10 (left: 5 9 10)
10 + 5 = 15 (left: 9 15)
15 + 9 = 24 (left: 24)
Answer: ((5 + 5) + 5) + 9 = 24
Steps:
13 - 10 = 3 (left: 3 4 9)
9 - 3 = 6 (left: 4 6)
4 * 6 = 24 (left: 24)
Answer: 4 * (9 - (13 - 10)) = 24
It's your turn:
Steps:
'''

# 3-shot
propose_prompt = '''Input: 2 8 8 14
Possible next steps:
2 + 8 = 10 (left: 8 10 14)
8 / 2 = 4 (left: 4 8 14)
14 + 2 = 16 (left: 8 8 16)
2 * 8 = 16 (left: 8 14 16)
8 - 2 = 6 (left: 6 8 14)
14 - 8 = 6 (left: 2 6 8)
14 / 2 = 7 (left: 7 8 8)
14 - 2 = 12 (left: 8 8 12)
Input: 4 4 10
Possible next steps:
4 + 4 = 8 (left: 8 10)
4 * 10 = 40 (left: 4 40)
10 - 4 = 6 (left: 4 6)
4 / 4 = 1 (left: 1 10)
4 - 4 = 0 (left: 0 10)
4 + 10 = 14 (left: 4 14)
4 * 4 = 16 (left: 10 16)
Input: 10 14
Possible next steps:
10 + 14 = 24 (left: 24)
14 - 10 = 4 (left: 4)
10 * 14 = 140 (left: 140)
14 / 10 = 1.4 (left: 1.4)
Generate possible next steps for the following inputs, following the example above. Note that the number of leftover digits should be one less than the number of input digits.
Input: {input}
Possible next steps:
'''

vote_prompt = '''You are playing Game of 24. 
Given an instruction, numbers and several choices of reasoning processes, decide which choice is most promising to complete the game.
Analyse in detail the correctness of each choice and the chance of completing the game, then conclude in the last line "The best choice is {{s}}", 
where s the integer id of the choice.
For example, Numbers: 4 5 6 10
10 - 4 = 6 (left: 5 6 6)
5 * 6 = 30 (left: 6 30)
30 - 6 = 24 (left: 24)
Answer: (5 * (10 - 4)) - 6 = 24
The first step should left 3 numbers, second step should left 2 numbers, third step should left one number and if success, it is 24.
Instruction: Use numbers and basic arithmetic operations (+ - * /) to obtain 24.
Numbers: {input}
'''

value_prompt = '''Evaluate if given numbers can reach 24. Conclude in the last line "confident", "likely" or "impossible".
Example 1:
4 4 10
4 + 4 + 10 = 8 + 10 = 18
4 * 10 - 4 = 40 - 4 = 36
(10 - 4) * 4 = 6 * 4 = 24
confident
Example 2:
11 12
11 + 12 = 23
12 - 11 = 1
11 * 12 = 132
11 / 12 = 0.91
impossible
Example 3:
5 7 8
5 + 7 + 8 = 12 + 8 = 20
(8 - 5) * 7 = 3 * 7 = 21
I cannot obtain 24 now, but numbers are within a reasonable range
likely
Now you should evaluate:
{input}
'''

value_last_step_prompt = '''Given an input and an answer, evaluate if the answer is correct, i.e. it uses each input exactly once and no other numbers, calculation is correct and reaches 24.
Give your judgement in the last line: "confident" or "impossible".
Input: {input}
Answer: {answer}
'''

