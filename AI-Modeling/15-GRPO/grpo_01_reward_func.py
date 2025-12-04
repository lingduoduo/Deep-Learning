import re

# Create a regex format to match the reasoning and solution sections
reasoning_start = "<start_working_out>"
reasoning_end   = "<end_working_out>"

solution_start  = "<SOLUTION>"
solution_end    = "</SOLUTION>"

def extract_grid_from_answer(text):
    if text is None:
        return None

    try:
        lines = []
        for line in text.strip().split('\n'):
            if re.search(r'[1-4]', line):
                numbers = [int(n) for n in re.findall(r'[1-4]', line)]  # [4, 2, 1, 3]
                if len(numbers) == 4:
                    lines.append(numbers)

        if len(lines) == 4 and all(len(line) == 4 for line in lines):
            return lines

        return None
    except Exception:
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
    question = question.replace("_", " ")
    return question


def extract_solution(text):
    # Remove any extra markers like end_of_turn
    text = text.replace("<end_of_turn>", "")

    # Try extracting content inside <SOLUTION>...</SOLUTION>
    match_format = re.compile(
        rf"{solution_start}(.*?){solution_end}",
        flags=re.MULTILINE | re.DOTALL
    )

    matchs = match_format.search(text)
    if matchs:
        return matchs.group(1).strip()

    # Fallback: detect last 4 valid grid lines
    lines = text.strip().split("\n")
    potential_grid_lines = []

    for line in reversed(lines):
        if re.match(r'\s*[1-4](\s+[1-4]){3}\s*$', line.strip()):
            potential_grid_lines.insert(0, line.strip())

        if len(potential_grid_lines) == 4:
            return "\n".join(potential_grid_lines)

        # If we started collecting grid lines but hit a non-matching line,
        # the grid is broken; stop early
        elif potential_grid_lines:
            break

    return None


def correctness_reward_func(completions, answers):
    responses = [completion[0]["content"] for completion in completions]
    rewards = []

    for response, correction_answer in zip(responses, answers):
        solution_text = extract_solution(response)
        if solution_text is None:
            rewards.append(0.0)
            continue

        predict_grid = extract_grid_from_answer(solution_text)
        correct_grid = extract_grid_from_answer(correction_answer)

        if predict_grid is None or correct_grid is None:
            rewards.append(0.0)
            continue

        if predict_grid == correct_grid and is_valid_sudoku_solution(predict_grid):
            rewards.append(5.0)
        else:
            rewards.append(0.0)

    return rewards


def in_reward_func(completions):
    responses = [completion[0]["content"] for completion in completions]
    rewards = []

    for response in responses:
        solution_text = extract_solution(response)
        if solution_text is None:
            rewards.append(0.0)
            continue

        grid = extract_grid_from_answer(solution_text)
        if grid is None:
            rewards.append(0.0)
            continue

        try:
            if all(all(num in [1, 2, 3, 4] for num in row) for row in grid):
                rewards.append(0.5)
            else:
                rewards.append(0.0)
        except Exception:
            rewards.append(0.0)

    return rewards


def grid_format_reward_func(completions):
    responses = [completion[0]["content"] for completion in completions]
    rewards = []

    for response in responses:
        solution_text = extract_solution(response)
        if solution_text is None:
            rewards.append(0.0)
            continue

        lines = solution_text.strip().split("\n")
        valid_lines = 0

        for line in lines:
            if re.match(r"\s*[1-4](\s+[1-4]){3}", line):
                valid_lines += 1

        if valid_lines == 4:
            rewards.append(1.0)
        elif valid_lines > 4:
            rewards.append(valid_lines / 8.0)
        else:
            rewards.append(0.0)

    return rewards


def reward_match_format_func(completions):
    scores = []

    for completion in completions:
        score = 0.0
        response = completion[0]["content"]

        match_format = re.compile(
            rf"[\s]{{0, }}"
            rf"{reasoning_start}.+?{reasoning_end}.*?"
            rf"{solution_start}(.+?){solution_end}"
            rf"[\s]*"
            rf"(?:<end_of_turn>)?"
            rf"[\s]*$",
            flags = re.MULTILINE | re.DOTALL
        )

        if match_format.search(response) is not None:
            score += 2.0

        scores.append(score)

    return scores
