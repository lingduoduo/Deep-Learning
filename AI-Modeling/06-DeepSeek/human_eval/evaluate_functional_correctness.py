import fire
import sys

from evaluation import evaluate_functional_correctness


def entry_point(
    sample_file: str,
    k: str = "1,10,100",
    n_workers: int = 4,
    # timeout: float = 3.0,
    timeout: float = 30,
    # problem_file: str = HUMAN_EVAL,
    problem_file: str = None,

):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    print(k)
    k = list(map(int, k.split(",")))
    results = evaluate_functional_correctness(sample_file, k, n_workers, timeout, problem_file)
    print(results)
    return results

# def entry_point(
#     sample_file,
#     k= "1,10,100",
#     n_workers = 4,
#     # timeout: float = 3.0,
#     timeout = 30,
#     # problem_file: str = HUMAN_EVAL,
#     problem_file = None,

# ):
#     """
#     Evaluates the functional correctness of generated samples, and writes
#     results to f"{sample_file}_results.jsonl.gz"
#     """
#     print(k)
#     k = list(map(int, k.split(",")))
#     results = evaluate_functional_correctness(sample_file, k, n_workers, timeout, problem_file)
#     print(results)
#     return results


def main():
    # freeze_support()
    fire.Fire(entry_point)


sys.exit(main())

"""
/mnt/human-eval/human_eval/evaluate_functional_correctness.py /mnt/human-eval/data/example_samples.jsonl
"""
