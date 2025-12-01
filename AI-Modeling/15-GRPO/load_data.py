import re

def extract_grid_from_answer(text):

    if text is None:
        return None

    try:
        lines = []
        for line in text.strip().split('\n'):
            if re.search(r'[1-4]', line):
                numbers = [int(n) for n in re.findall(r'[1-4]', line)]  # [4,2,1,3]
                if len(numbers) == 4:
                    lines.append(numbers)

        if len(lines) == 4 and all(len(line) == 4 for line in lines):
            return lines

        return None

    except Exception as e:
        return None

def is_valid_sudoku_solution(grid):
    # Validate grid shape
    if grid is None or len(grid) != 4 or any(len(row) != 4 for row in grid):
        return False

    # Check rows
    for row in grid:
        if sorted(row) != [1, 2, 3, 4]:
            return False

    # Check columns
    for col in range(4):
        column = [grid[row][col] for row in range(4)]
        if sorted(column) != [1, 2, 3, 4]:
            return False

    # Check 2x2 sub-grids
    for box_row in range(0, 4, 2):
        for box_col in range(0, 4, 2):
            box = []
            for r in range(box_row, box_row + 2):
                for c in range(box_col, box_col + 2):
                    box.append(grid[r][c])
            if sorted(box) != [1, 2, 3, 4]:
                return False

    return True

def parse_sudoku_question(question):
    question = question.replace("_", ' ')
    return question

def extract_

def correctness_reward_func(completions, answer):
    response = [completion[0]['content'] for completion in completions]

grid = extract_grid_from_answer("aas4 2 1 3\n1 3 4 2\n2 1 3 4\n3 4 2 1sadas")
print(grid)
print(is_valid_sudoku_solution(grid))