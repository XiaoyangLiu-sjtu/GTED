import os
import re
import json
from tqdm.contrib import tzip, tenumerate
from src.hover import HoverExtractor, HoverProcessor, HoverRewriter
from src.parser import Reorganizer, NameExtractor
from src.opter import OPTBuilder, OPTVisualizer, OPTSimilarer


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


def syntax_standardization(header_list, formal_statement_list):
    rewriter = HoverRewriter()
    reorganizer = Reorganizer()
    reorganized_formal_statement_list = []

    with HoverExtractor() as extractor:
        for header, formal_statement in tzip(header_list, formal_statement_list, desc="Syntax Standardization"):
            extract_results = extractor.extract(header+"\n"+formal_statement)
            hover_results = rewriter.rewrite(extract_results)
            reorganized_formal_statement = reorganizer.reorganize_statement(hover_results)
            reorganized_formal_statement_list.append(reorganized_formal_statement)
    return reorganized_formal_statement_list


def build_opt(header_list, reorganized_formal_statement_list, extract_path=None, processed_path=None, opt_path=None, png_path=None):
    processor = HoverProcessor()
    builder = OPTBuilder()
    visualizer = OPTVisualizer()
    path_templates = [extract_path, processed_path, opt_path, png_path]
    tree_result_list = []

    with HoverExtractor() as extractor:
        for index, reorganized_formal_statement in tenumerate(reorganized_formal_statement_list, desc="OPT Construction"):
            path_used = []
            for template in path_templates:
                path = template.replace("index", str(index+1))
                path_used.append(path)
                os.makedirs(os.path.dirname(path), exist_ok=True)

            extract_result = extractor.extract(header_list[index]+"\n"+reorganized_formal_statement, path_used[0])
            process_result = processor.process(extract_result, path_used[1])
            tree_result = builder.build(process_result, path_used[2])
            visualizer.visualize(tree_result[0], path_used[3])

            # Uncomment the following lines to mark freecost nodes for variables and hypotheses
            # metadata = NameExtractor().name_extractor(header_list[index], reorganized_formal_statement)
            # mark_freecost_nodes(metadata, tree_result)
            
            tree_result_list.append(tree_result[0])
    return tree_result_list


def test_benchmark(benchmark):
    data = read_json(f"experiment/{benchmark}/human_evaluation.json")

    label_header_list, label_formal_statement_list = [], []
    predict_header_list, predict_formal_statement_list = [], []
    for item in data:
        for subitem in item["sub_questions"]:
            label_header = re.sub(r"\n+", "\n", item["header"]).replace("import Mathlib\n", "")
            label_header_list.append(label_header)
            label_formal_statement_list.append(item["FL (Label)"])
            predict_header = re.sub(r"\n+", "\n", subitem["header"]).replace("import Mathlib\n", "")
            predict_header_list.append(predict_header)
            predict_formal_statement_list.append(subitem["FL (Prediction)"])
    reorganized_label_formal_statement_list = syntax_standardization(label_header_list, label_formal_statement_list)
    reorganized_predict_formal_statement_list = syntax_standardization(predict_header_list, predict_formal_statement_list)

    label_tree_result_list = build_opt(
        label_header_list,
        reorganized_label_formal_statement_list,
        extract_path=f"test_files/experiment/{benchmark}/ted/label/extract/label_code_index.jsonl",
        processed_path=f"test_files/experiment/{benchmark}/ted/label/processed/label_code_index.jsonl",
        opt_path=f"test_files/experiment/{benchmark}/ted/label/opt/label_code_index.json",
        png_path=f"test_files/experiment/{benchmark}/ted/label/figures/label_code_index.png",
    )
    predict_tree_result_list = build_opt(
        predict_header_list,
        reorganized_predict_formal_statement_list,
        extract_path=f"test_files/experiment/{benchmark}/ted/predict/extract/predict_code_index.jsonl",
        processed_path=f"test_files/experiment/{benchmark}/ted/predict/processed/predict_code_index.jsonl",
        opt_path=f"test_files/experiment/{benchmark}/ted/predict/opt/predict_code_index.json",
        png_path=f"test_files/experiment/{benchmark}/ted/predict/figures/predict_code_index.png",
    )

    for index, (label_tree_result, predict_tree_result) in enumerate(zip(label_tree_result_list, predict_tree_result_list)):
        result = OPTSimilarer().similarer(data_a=label_tree_result, data_b=predict_tree_result)
        data[index]["ted"] = result["ted_similarity"]
    write_json(f"test_files/experiment/{benchmark}/ted/result.json", data)