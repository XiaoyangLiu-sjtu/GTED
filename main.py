import time
import utils
from src.opter import OPTSimilarer
from src.evaluator import *


def tree_lean_codes(header_list, formal_statement_list):
    """
    Input header and formal statement to build the corresponding operator tree.
    """
    for index, header in enumerate(header_list):
        header_list[index] = re.sub(r"\n+", "\n", header).replace("import Mathlib\n", "").replace("import Mathlib", "")

    reorganized_formal_statement_list = utils.syntax_standardization(header_list, formal_statement_list)
    tree_result_list = utils.build_opt(
        header_list,
        reorganized_formal_statement_list,
        extract_path=f"test_files/temp/extract/test_code_index.jsonl",
        processed_path=f"test_files/temp/processed/test_code_index.jsonl",
        opt_path=f"test_files/temp/opt/test_code_index.json",
        png_path=f"test_files/temp/figures/test_code_index.png",
    )
    return tree_result_list


def ted_lean_codes(label_header_list, label_formal_statement_list, predict_header_list, predict_formal_statement_list):
    """
    Input a pair of header and formal statements, build the corresponding operator tree and calculate the TED similarity.
    """
    label_tree_result_list = tree_lean_codes(label_header_list, label_formal_statement_list)
    predict_tree_result_list = tree_lean_codes(predict_header_list, predict_formal_statement_list)
    
    ted_similarity_list = []
    for index, (label_tree_result, predict_tree_result) in enumerate(zip(label_tree_result_list, predict_tree_result_list)):
        result = OPTSimilarer().similarer(data_a=label_tree_result, data_b=predict_tree_result)
        ted_similarity_list.append({f"pair {index+1}": result["ted_similarity"]})
    return ted_similarity_list


def evaluation_benchmark(benchmark):
    """
    Input a benchmark and evaluate it using various methods.
    """
    # IdentityMatcher(benchmark).identity_matcher()
    # Typechecker(benchmark).typechecker()
    # BLEUer(benchmark).bleuer()
    # MajorityVoter(benchmark).majority_voter()
    # DefinitionalEqualityer(benchmark).definitional_equalityer()
    # BEqer(benchmark).beqer()
    utils.test_benchmark(benchmark)
    # TreeSimilarer(benchmark).treesimilarer()


if __name__ == "__main__":
    # Function1: tree_lean_codes
    start_time = time.time()
    header_list = ["import Mathlib\n"] * 2500
    data = utils.read_json("/nfs/my/lxy/TreeAutoformalization/dataset/atlas_dataset_5000.json")
    formal_statement_list = [item["formal_statement"] for item in data]
    tree_lean_codes(header_list, formal_statement_list)
    end_time = time.time()
    print(f"Time taken for one lean code: {end_time - start_time:.2f} seconds")

    # # Function2: ted_lean_codes
    # label_header_list = ["import Mathlib"] * 3
    # label_formal_statement_list = ["theorem th_name (p : Prop) : let q := ¬¬p; p = q := by sorry"] * 100
    # predict_header_list = ["import Mathlib"] * 3
    # predict_formal_statement_list = ["theorem th_name (p : Prop) : let q := ¬¬p; p = q := by sorry"] * 100
    # ted_lean_codes(label_header_list, label_formal_statement_list, predict_header_list, predict_formal_statement_list)

    # # Function3: evaluation_benchmark
    evaluation_benchmark("minif2f")
    evaluation_benchmark("proofnet")