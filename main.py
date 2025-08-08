import os
import json
from src.hover import HoverExtractor, HoverProcessor, HoverRewriter
from src.parser import Reorganizer, NameExtractor
from src.aster import ASTBuilder, ASTVisualizer, ASTSimilarer
from src.evaluator import *


def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def write_json(file_path, data):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def mark_freecost_nodes(metadata, tree):
    variables = set(metadata.get("variables_name", []))
    hypotheses = set(metadata.get("hypotheses_name", []))

    def process_node(node):
        content = node.get("content", "")
        if content in variables:
            node["variable_freecost"] = True
        if content in hypotheses:
            node["hypothesis_freecost"] = True
        for child in node.get("children", []):
            process_node(child)
    process_node(tree)


def build_ast(header, formal_statement, extract_path=None, processed_path=None, ast_path=None, png_path=None):
    extractor = HoverExtractor()
    processor = HoverProcessor()
    builder = ASTBuilder()
    visualizer = ASTVisualizer()

    for path in [extract_path, processed_path, ast_path, png_path]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    extract_results = extractor.extract(header + formal_statement, extract_path)
    process_results = processor.process(extract_results, processed_path)
    tree_results = builder.build(process_results, ast_path)
    visualizer.visualize(tree_results, png_path)

    metadata = NameExtractor().name_extractor(header, formal_statement)
    mark_freecost_nodes(metadata, tree_results)
    write_json(ast_path, tree_results)
    return tree_results


def rewrite_lean_codes(input_path, output_path):
    extractor = HoverExtractor()
    rewriter = HoverRewriter()
    reorganizer = Reorganizer()

    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for index, line in enumerate(infile):
            data = json.loads(line.strip())
            lean_code = data["header"] + data["formal_statement"]
            extract_results = extractor.extract(lean_code)
            hover_results = rewriter.rewrite(extract_results)
            data["hover_formal_statement"] = hover_results
            reorganized_formal_statement = reorganizer.reorganize_statement(hover_results)
            data["reorganized_hover_formal_statement"] = reorganized_formal_statement
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
            print(f"Process: {index+1}")


def test_one_lean_code(header, formal_statement, index=1):
    extractor = HoverExtractor()
    rewriter = HoverRewriter()
    reorganizer = Reorganizer()

    extract_results = extractor.extract(header + formal_statement)
    hover_results = rewriter.rewrite(extract_results)
    reorganized_formal_statement = reorganizer.reorganize_statement(hover_results)

    build_ast(
        header=header,
        formal_statement=reorganized_formal_statement,
        extract_path=f"test_files/extract/test_code_{index}.jsonl",
        processed_path=f"test_files/processed/test_code_{index}.jsonl",
        ast_path=f"test_files/ast/test_code_{index}.json",
        png_path=f"test_files/figures/test_code_{index}.png",
    )


def test_benchmark_lean_code(benchmark):
    data = read_json(f"experiment/{benchmark}/human_evaluation.json")
    for index, item in enumerate(data):
        for subitem in item["sub_questions"]:
            label_tree = build_ast(
                header=item["header"],
                formal_statement=item["reorganized_hover_formal_statement"],
                extract_path=f"experiment/{benchmark}/gted/label/extract/label_code_{index+1}.jsonl",
                processed_path=f"experiment/{benchmark}/gted/label/processed/label_code_{index+1}.jsonl",
                ast_path=f"experiment/{benchmark}/gted/label/ast/label_code_{index+1}.json",
                png_path=f"experiment/{benchmark}/gted/label/figures/label_code_{index+1}.png",
                )
            predict_tree = build_ast(
                header=subitem["header"],
                formal_statement=subitem["reorganized_hover_formal_statement"],
                extract_path=f"experiment/{benchmark}/gted/predict/extract/predict_code_{index+1}.jsonl",
                processed_path=f"experiment/{benchmark}/gted/predict/processed/predict_code_{index+1}.jsonl",
                ast_path=f"experiment/{benchmark}/gted/predict/ast/predict_code_{index+1}.json",
                png_path=f"experiment/{benchmark}/gted/predict/figures/predict_code_{index+1}.png",
                )
            result = ASTSimilarer().similarer(data_a=label_tree, data_b=predict_tree)
            subitem["ted_similarity"] = result["ted_similarity"]
        print(f"Process: {index+1} --> {len(data)}")
    write_json(f"experiment/{benchmark}/gted/result.json", data)


def evaluation(benchmark):
    IdentityMatcher(benchmark).identity_matcher()
    Typechecker(benchmark).typechecker()
    BLEUer(benchmark).bleuer()
    MajorityVoter(benchmark).majority_voter()
    DefinitionalEqualityer(benchmark).definitional_equalityer()
    BEqer(benchmark).beqer()
    test_benchmark_lean_code(benchmark)
    TreeSimilarer(benchmark).treesimilarer()


if __name__ == "__main__":
    # test one lean code
    header = "import Mathlib\n\nnamespace EZAnalysis\n\ndef lim (a_ : ℕ → ℝ) (a : ℝ) : Prop := ∀ ε > 0, ∃ N : ℕ, ∀ n > N, -ε < a_ n - a ∧ a_ n - a < ε\n\n"
    formal_statement = "theorem lim_uniq {a_ : ℕ → ℝ} {a b : ℝ} (ha : lim a_ a) (hb : lim a_ b) : a = b := by sorry\n\nend EZAnalysis\n\n"
    test_one_lean_code(header, formal_statement, index=1)

    # test benchmark lean code
    # evaluation("minif2f")
    # evaluation("proofnet")