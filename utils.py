import os
import re
import json
from src.hover import ExtractorPool, HoverProcessor, HoverRewriter
from src.parser import Reorganizer, NameExtractor
from src.opter import OPTBuilder, OPTVisualizer, OPTSimilarer


NUM_SERVERS = 10
HOME_DIR = os.path.expanduser("~")
LEAN_PATH = f"{HOME_DIR}/.elan/bin/lean"
MATHLIB_PATH = "../ATLAS/src/repl"


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


def _traverse_node(node):
    formal_statement = node.get("formal_content", "")
    children = node.get("children", [])

    if "sorry" in formal_statement.lower():
        return f"sorry_error | {formal_statement}"
    if formal_statement.count("_") != len(children) and len(children) > 0 and (not formal_statement.startswith("f_")):
        return f"mismatch_error | {formal_statement}"
    for child in children:
        error_in_child = _traverse_node(child)
        if error_in_child:
            return error_in_child
    return None


def analyze_opt_files(folder_path, tag):
    error_log, sorry_count, mismatch_count, name_count, json_count = [], 0, 0, 0, 0
    all_filenames = os.listdir(folder_path)
    target_files = [f for f in all_filenames if f.startswith(f"{tag}_code_") and f.endswith(".json")]
    sorted_files = sorted(target_files, key=lambda f: int(f.replace(f"{tag}_code_", "").replace(".json", "")))

    for filename in sorted_files:
        filepath = os.path.join(folder_path, filename)
        data = read_json(filepath)
        error_type = None

        fs, rfs = data[0].get("formal_statement"), data[0].get("reorganized_formal_statement")
        fsl, rfsl = fs.split(), rfs.split()
        fs_f2, rfs_f2 = "".join(fsl[:2]), "".join(rfsl[:2]).split('.', 1)[0]
        if fs_f2 != rfs_f2:
            error_type = "name_error"
        else:
            try:
                start_nodes = data[1]["children"][0]["children"]
                for node in start_nodes:
                    file_error = _traverse_node(node)
                    if file_error:
                        error_type = file_error
                        break 
            except:
                error_type = "json_error"

        if error_type:
            if error_type.startswith("sorry_error"):
                sorry_count += 1
            elif error_type.startswith("mismatch_error"):
                mismatch_count += 1
            elif error_type == "name_error":
                name_count += 1
            elif error_type == "json_error":
                json_count += 1
            error_message = error_type
            error_log.append({"file": filename, "error": error_message})

    report_path = os.path.join(folder_path, "analysis_report.log")
    total_files, total_errors = len(sorted_files), len(error_log)
    total_errors_percentage = (total_errors /  total_files * 100) if total_files > 0 else 0
    sorry_percentage = (sorry_count / total_files * 100) if total_files > 0 else 0
    mismatch_percentage = (mismatch_count / total_files * 100) if total_files > 0 else 0
    name_percentage = (name_count / total_files * 100) if total_files > 0 else 0
    json_percentage = (json_count / total_files * 100) if total_files > 0 else 0

    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write("--- OPT Analysis Report ---\n")
        report_file.write(f"Processed Folder: {os.path.abspath(folder_path)}\n")
        report_file.write(f"File Tag: {tag}\n\n")
        report_file.write("--- Summary Statistics ---\n")
        report_file.write(f"Total Files Analyzed: {total_files}\n")
        report_file.write(f"Total Files with Errors: {total_errors} ({total_errors_percentage:.2f}%)\n")
        report_file.write(f"Files with sorry_error: {sorry_count} ({sorry_percentage:.2f}%)\n")
        report_file.write(f"Files with mismatch_error: {mismatch_count} ({mismatch_percentage:.2f}%)\n")
        report_file.write(f"Files with name_error: {name_count} ({name_percentage:.2f}%)\n")
        report_file.write(f"Files with json_error: {json_count} ({json_percentage:.2f}%)\n\n")
        report_file.write("--- Detailed Error Log ---\n")
        if not error_log:
            report_file.write("No errors found.\n")
        else:
            for entry in error_log:
                report_file.write(f"{entry['file']}: {entry['error']}\n")


def syntax_standardization(header_list, formal_statement_list):
    rewriter = HoverRewriter()
    reorganizer = Reorganizer()
    reorganized_formal_statement_list = []

    snippets = [header + "\n" + formal_statement for header, formal_statement in zip(header_list, formal_statement_list)]
    with ExtractorPool(num_workers=NUM_SERVERS, lean_bin=LEAN_PATH, mathlib_path=MATHLIB_PATH, desc="Syntax Standardization") as pool:
        extract_result_list = pool.process_all(code_snippets=snippets, out_paths=[None] * len(snippets))

    for extract_result in extract_result_list:
        hover_result = rewriter.rewrite(extract_result)
        reorganized_formal_statement = reorganizer.reorganize_statement(hover_result)
        reorganized_formal_statement_list.append(reorganized_formal_statement)
    return reorganized_formal_statement_list


def build_opt(header_list, informal_statement_list, formal_statement_list, reorganized_formal_statement_list, extract_path=None, processed_path=None, opt_path=None, png_path=None):
    processor = HoverProcessor()
    builder = OPTBuilder()
    visualizer = OPTVisualizer()
    path_templates = [extract_path, processed_path, opt_path, png_path]
    tree_result_list = []

    snippets = [header + "\n" + reorganized_formal_statement for header, reorganized_formal_statement in zip(header_list, reorganized_formal_statement_list)]
    output_files = [extract_path.replace("index", str(index + 1)) if extract_path else None for index in range(len(snippets))]
    os.makedirs(os.path.dirname(extract_path), exist_ok=True) if extract_path else None
    with ExtractorPool(num_workers=NUM_SERVERS, lean_bin=LEAN_PATH, mathlib_path=MATHLIB_PATH, desc="OPT Construction") as pool:
        extract_result_list = pool.process_all(code_snippets=snippets, out_paths=output_files)

    for index, extract_result in enumerate(extract_result_list):
        path_used = []
        for template in path_templates:
            if template:
                path = template.replace("index", str(index+1))
                path_used.append(path)
                os.makedirs(os.path.dirname(path), exist_ok=True)
            else:
                path_used.append(None)
                    
        process_result = processor.process(extract_result, path_used[1])
        tree_result = builder.build(process_result, path_used[2], informal_statement_list[index], formal_statement_list[index], reorganized_formal_statement_list[index])
        visualizer.visualize(tree_result, path_used[3])

        # Uncomment the following lines to mark freecost nodes for variables and hypotheses
        # metadata = NameExtractor().name_extractor(header_list[index], reorganized_formal_statement)
        # mark_freecost_nodes(metadata, tree_result)
            
        tree_result_list.append(tree_result)

    analyze_opt_files(os.path.dirname(opt_path), os.path.basename(opt_path).split("_")[0])
    return tree_result_list


def test_benchmark(benchmark):
    data = read_json(f"experiment/{benchmark}/human_evaluation.json")

    label_header_list, label_formal_statement_list, label_informal_statement_list = [], [], []
    predict_header_list, predict_formal_statement_list, predict_informal_statement_list = [], [], []
    for item in data:
        for subitem in item["sub_questions"]:
            label_header = re.sub(r"\n+", "\n", item["header"]).replace("import Mathlib\n", "")
            label_header_list.append(label_header)
            label_formal_statement_list.append(item["FL (Label)"])
            label_informal_statement_list.append(item.get("IL (Label)"))
            predict_header = re.sub(r"\n+", "\n", subitem["header"]).replace("import Mathlib\n", "")
            predict_header_list.append(predict_header)
            predict_formal_statement_list.append(subitem["FL (Prediction)"])
            predict_informal_statement_list.append(subitem.get("IL (Prediction)"))
    reorganized_label_formal_statement_list = syntax_standardization(label_header_list, label_formal_statement_list)
    reorganized_predict_formal_statement_list = syntax_standardization(predict_header_list, predict_formal_statement_list)

    label_tree_result_list = build_opt(
        label_header_list,
        label_informal_statement_list,
        label_formal_statement_list,
        reorganized_label_formal_statement_list,
        extract_path=f"test_files/{benchmark}/ted/label/extract/label_code_index.jsonl",
        processed_path=f"test_files/{benchmark}/ted/label/processed/label_code_index.jsonl",
        opt_path=f"test_files/{benchmark}/ted/label/opt/label_code_index.json",
        png_path=f"test_files/{benchmark}/ted/label/figures/label_code_index.png",
    )
    predict_tree_result_list = build_opt(
        predict_header_list,
        predict_informal_statement_list,
        predict_formal_statement_list,
        reorganized_predict_formal_statement_list,
        extract_path=f"test_files/{benchmark}/ted/predict/extract/predict_code_index.jsonl",
        processed_path=f"test_files/{benchmark}/ted/predict/processed/predict_code_index.jsonl",
        opt_path=f"test_files/{benchmark}/ted/predict/opt/predict_code_index.json",
        png_path=f"test_files/{benchmark}/ted/predict/figures/predict_code_index.png",
    )

    for index, (label_tree_result, predict_tree_result) in enumerate(zip(label_tree_result_list, predict_tree_result_list)):
        result = OPTSimilarer().similarer(data_a=label_tree_result, data_b=predict_tree_result)
        data[index]["ted_similarity"] = result["ted_similarity"]
    write_json(f"test_files/{benchmark}/ted/result.json", data)