import time
import utils
from src.evaluator import *


def test_lean_codes(header_list, formal_statement_list):
    for index, header in enumerate(header_list):
        header_list[index] = re.sub(r"\n+", "\n", header).replace("import Mathlib\n", "").replace("import Mathlib", "")

    reorganized_formal_statement_list = utils.syntax_standardization(header_list, formal_statement_list)
    utils.build_ast(
        header_list,
        reorganized_formal_statement_list,
        extract_path=f"test_files/extract/test_code_index.jsonl",
        processed_path=f"test_files/processed/test_code_index.jsonl",
        ast_path=f"test_files/ast/test_code_index.json",
        png_path=f"test_files/figures/test_code_index.png",
    )


def benchmark_evaluation(benchmark):
    IdentityMatcher(benchmark).identity_matcher()
    Typechecker(benchmark).typechecker()
    BLEUer(benchmark).bleuer()
    MajorityVoter(benchmark).majority_voter()
    DefinitionalEqualityer(benchmark).definitional_equalityer()
    BEqer(benchmark).beqer()
    utils.test_benchmark(benchmark)
    TreeSimilarer(benchmark).treesimilarer()


if __name__ == "__main__":
    # test lean codes
    start_time = time.time()
    header = "import Mathlib"
    formal_statement = "theorem th_name (p : Prop) : let q := ¬¬p; p = q := by sorry"
    header_list = [header] * 3
    formal_statement_list = [formal_statement] * 3
    test_lean_codes(header_list, formal_statement_list)
    end_time = time.time()
    print(f"Time taken for one lean code: {end_time - start_time:.2f} seconds")

    # test benchmark
    benchmark_evaluation("minif2f")
    benchmark_evaluation("proofnet")