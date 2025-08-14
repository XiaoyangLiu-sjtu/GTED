import time
import utils
from src.opter import OPTSimilarer
from src.evaluator import *


def tree_lean_codes(header_list, formal_statement_list, informal_statement_list=None, save_folder_path="test_files", tag="test", extract_save=True, processed_save=True, opt_save=True, png_save=True):
    """
    Input header and formal statement to build the corresponding operator tree.
    """
    for index, header in enumerate(header_list):
        header_list[index] = re.sub(r"\n+", "\n", header).replace("import Mathlib\n", "").replace("import Mathlib", "")

    reorganized_formal_statement_list = utils.syntax_standardization(header_list, formal_statement_list)
    tree_result_list = utils.build_opt(
        header_list,
        informal_statement_list if informal_statement_list else [""] * len(header_list),
        formal_statement_list,
        reorganized_formal_statement_list,
        extract_path=f"{save_folder_path}/{tag}/extract/{tag}_code_index.jsonl" if extract_save else None,
        processed_path=f"{save_folder_path}/{tag}/processed/{tag}_code_index.jsonl" if processed_save else None,
        opt_path=f"{save_folder_path}/{tag}/opt/{tag}_code_index.json" if opt_save else None,
        png_path=f"{save_folder_path}/{tag}/figures/{tag}_code_index.png" if png_save else None
    )
    return tree_result_list


def ted_lean_codes(label_header_list, label_formal_statement_list, predict_header_list, predict_formal_statement_list, extract_save=True, processed_save=True, opt_save=True, png_save=True):
    """
    Input a pair of header and formal statements, build the corresponding operator tree and calculate the TED similarity.
    """
    label_tree_result_list = tree_lean_codes(label_header_list, label_formal_statement_list, tag="label", extract_save=extract_save, processed_save=processed_save, opt_save=opt_save, png_save=png_save)
    predict_tree_result_list = tree_lean_codes(predict_header_list, predict_formal_statement_list, tag="predict", extract_save=extract_save, processed_save=processed_save, opt_save=opt_save, png_save=png_save)

    ted_result_list = []
    for label_tree_result, predict_tree_result in zip(label_tree_result_list, predict_tree_result_list):
        result = OPTSimilarer().similarer(data_a=label_tree_result, data_b=predict_tree_result)
        ted_result_list.append(result)
    return ted_result_list


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
    # # Function1: tree_lean_codes
    # start_time = time.time()
    # header_list = ["import Mathlib\n"] * 100
    # formal_statement_list = ["theorem th_name (p : Prop) : let q := ¬¬p; p = q := by sorry"] * 100
    # tree_lean_codes(header_list, formal_statement_list, extract_save=False, processed_save=False, png_save=False)
    # end_time = time.time()
    # print(f"Time taken for one lean code: {end_time - start_time:.2f} seconds")

    # Function2: ted_lean_codes
    label_header_list = ["import Mathlib"] * 3
    label_formal_statement_list = ["theorem th_name (p : Prop) : let q := ¬¬p; p = q := by sorry"] * 3
    predict_header_list = ["import Mathlib"] * 3
    predict_formal_statement_list = ["theorem th_name (p : Prop) : let q := ¬¬p; p = q := by sorry"] * 3
    print(ted_lean_codes(label_header_list, label_formal_statement_list, predict_header_list, predict_formal_statement_list))

    # Function3: evaluation_benchmark
    evaluation_benchmark("minif2f")
    evaluation_benchmark("proofnet")