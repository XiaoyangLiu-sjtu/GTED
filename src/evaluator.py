import os
import re
import gc
import time
import json
import torch
from tqdm import tqdm
from openai import OpenAI
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from src.verifier import FLVerifier


class Evaluator:
    def __init__(self, benchmark, metric):
        self.benchmark = benchmark
        self.ori_file_path = f"experiment/{benchmark}/human_evaluation.json"
        self.file_path = f"experiment/{benchmark}/{metric}/result.json"
        self.acc_path = f"experiment/{benchmark}/{metric}/accuracy.json"

    def read_json(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def write_json(self, file_path, data):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def find_best_element(self, metric):
        data_list = self.read_json(self.acc_path)
        max_kappa = -float("inf")
        kappa_filtered_elements = []
        for item in data_list:
            current_kappa = item.get(metric, {}).get("kappa", -float("inf"))
            if current_kappa > max_kappa:
                max_kappa = current_kappa
                kappa_filtered_elements = [item]
            elif current_kappa == max_kappa:
                kappa_filtered_elements.append(item)

        max_accuracy = -float("inf")
        accuracy_filtered_elements = []
        for item in kappa_filtered_elements:
            current_accuracy = item.get(metric, {}).get("accuracy", -float("inf"))
            if current_accuracy > max_accuracy:
                max_accuracy = current_accuracy
                accuracy_filtered_elements = [item]
            elif current_accuracy == max_accuracy:
                accuracy_filtered_elements.append(item)

        max_precision = -float("inf")
        precision_filtered_elements = []
        for item in accuracy_filtered_elements:
            current_precision = item.get(metric, {}).get("precision", -float("inf"))
            if current_precision > max_precision:
                max_precision = current_precision
                precision_filtered_elements = [item]
            elif current_precision == max_precision:
                precision_filtered_elements.append(item)

        max_recall = -float("inf")
        final_best_element = None
        for item in precision_filtered_elements:
            current_recall = item.get(metric, {}).get("recall", -float("inf"))
            if current_recall > max_recall:
                max_recall = current_recall
                final_best_element = item
        print(final_best_element)

    def merge_json_files(self, file_paths, output_path):
        all_lists = []
        for path in file_paths:
            with open(path, "r", encoding="utf-8") as f:
                all_lists.append(json.load(f))
        
        min_len = min(len(lst) for lst in all_lists)
        merged_list = []
        for idx in range(min_len):
            for lst in all_lists:
                merged_list.append(lst[idx])
                
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(merged_list, f, ensure_ascii=False, indent=4)

    def calculate(self, stats):
        precision_denom = stats["true_positive"] + stats["false_positive"]
        precision = stats["true_positive"] / precision_denom if precision_denom != 0 else 0.0
        recall_denom = stats["true_positive"] + stats["false_negative"]
        recall = stats["true_positive"] / recall_denom if recall_denom != 0 else 0.0
        accuracy = (stats["true_positive"] + stats["true_negative"]) / stats["total"] if stats["total"] != 0 else 0.0
        if stats["total"] == 0:
            pe = 0.0
        else:
            pe = ((stats["true_positive"] + stats["false_negative"]) *
                  (stats["true_positive"] + stats["false_positive"]) +
                  (stats["true_negative"] + stats["false_positive"]) *
                  (stats["true_negative"] + stats["false_negative"])) / (
                      stats["total"]**2)
        kappa_denom = 1 - pe
        kappa = (accuracy - pe) / kappa_denom if kappa_denom != 0 else 0.0
        stats["precision"], stats["recall"], stats["accuracy"], stats["kappa"] = precision, recall, accuracy, kappa
        return stats

    def summary(self, json_data, metric):
        stats = {
            "true_positive": 0,
            "true_negative": 0,
            "false_positive": 0,
            "false_negative": 0,
            "total": 0
        }
        for question in json_data:
            sub_questions = question.get("sub_questions", [])
            for sub_question in sub_questions:
                human_eval = "same" if sub_question.get(
                    "Human_Evaluation") is True else "different"
                for key, prediction in sub_question.items():
                    if key == metric:
                        if prediction == human_eval:
                            if human_eval == "same":
                                stats["true_positive"] += 1
                            else:
                                stats["true_negative"] += 1
                        else:
                            if human_eval == "same":
                                stats["false_negative"] += 1
                            else:
                                stats["false_positive"] += 1
                        stats["total"] += 1
        return self.calculate(stats)
    

class IdentityMatcher(Evaluator):
    def __init__(self, benchmark, metric="identity_match"):
        super().__init__(benchmark, metric)

    def identity_matcher(self):
        data = self.read_json(self.ori_file_path)
        for item in data:
            for subitem in item["sub_questions"]:
                label = re.sub(r"\s", "", item["FL (Label)"]).strip().replace(item["name"], "tm_name")
                predict = re.sub(r"\s", "", subitem["FL (Prediction)"]).strip().replace(subitem["name"], "tm_name")
                if label == predict:
                    subitem["identity_match"] = "same"
                else:
                    subitem["identity_match"] = "different"
        self.write_json(self.file_path, data)
        identity_match_result = self.summary(data, "identity_match")
        results = {"identity_match": identity_match_result}
        self.write_json(self.acc_path, results)


class Typechecker(Evaluator):
    def __init__(self, benchmark, metric="typecheck"):
        super().__init__(benchmark, metric)

    def typechecker(self):
        data = self.read_json(self.ori_file_path)
        for item in data:
            for subitem in item["sub_questions"]:
                subitem["typecheck"] = "same"
        self.write_json(self.file_path, data)
        typecheck_result = self.summary(data, "typecheck")
        results = {"typecheck": typecheck_result}
        self.write_json(self.acc_path, results)


class BLEUer(Evaluator):
    def __init__(self, benchmark, metric="bleu"):
        super().__init__(benchmark, metric)
        self.evaluate()

    def evaluate(self):
        data = self.read_json(self.ori_file_path)
        for item in data:
            for subitem in item["sub_questions"]:
                subitem["bleu"] = sentence_bleu(
                    [item["FL (Label)"].replace(item["name"], "tm_name").split()],
                    subitem["FL (Prediction)"].replace(subitem["name"], "tm_name").split(),
                    smoothing_function=SmoothingFunction().method4)
        self.write_json(self.file_path, data)

    def rewrite(self, threshold):
        data = self.read_json(self.file_path)
        for item in data:
            for subitem in item["sub_questions"]:
                subitem["bleu"] = "same" if subitem["bleu"] >= threshold else "different"
        return self.summary(data, "bleu")

    def bleuer(self):
        results, step = [], 0.001
        for i in range(0, 1001):
            threshold = i * step
            bleu_result = self.rewrite(threshold)
            results.append({"threshold": threshold, "bleu": bleu_result})
        self.write_json(self.acc_path, results)
        self.find_best_element("bleu")


class APIModel:
    def __init__(self, name_or_path, base_url, api_key, sampling_params):
        self.name_or_path = name_or_path
        self.base_url = base_url
        self.api_key = api_key
        self.sampling_params = sampling_params
        self.prompt = """Please check following two math problems is same or different? Please consider each statement in two problems, they are different if any statement is different. Please point out any differences you found. Please reply **same** or **different** in the final sentence with bold format.

Problem 1: {THM_1}

Problem 2: {THM_2}
"""
        self.client = self._init_model()

    def _init_model(self):
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=10000,
            max_retries=10,
        )

    def get_query(self, task):
        query = self.prompt.format(
            THM_1 = task["informal_statement"],
            THM_2 = task["back_translation"]
        )
        return [
                {"role": "user", "content": query}
            ]

    def generate(self, task):
        try:
            response = self.client.chat.completions.create(
                model=self.name_or_path,
                messages=self.get_query(task),
                **self.sampling_params,
            )
            return response.choices[0].message.content
        except:
            return "null"
        
    def generate_batch(self, task_list, desc, max_workers=100):
        remain_task_list, responses, resolve_times = task_list, [], 0
        while remain_task_list:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self.generate, task): task for task in remain_task_list}
                for future in tqdm(futures, total=len(remain_task_list), desc=desc): 
                    task = futures[future]
                    result = future.result()
                    if result != "null":
                        final_sentence = result.strip().split("\n")[-1]
                        same_cnt = 1 if "same" in final_sentence else 0
                        different_cnt = 1 if "different" in final_sentence else 0
                        if same_cnt + different_cnt != 1:
                            continue
                        elif same_cnt == 1:
                            result = "same"
                            responses.append((task, result))
                        elif different_cnt == 1:
                            result = "different"
                            responses.append((task, result))
            remain_task_list = [item for item in remain_task_list if item not in [task for task, _ in responses]]
            print(f"Remaining tasks: {len(remain_task_list)}")

            resolve_times += 1
            if resolve_times >= 20:
                print("Too many retries, stopping...")
                break
        return responses
    
    def __del__(self):
        del self.client
        gc.collect()


class MajorityVoter(Evaluator):
    def __init__(self, benchmark, metric="majority_vote"):
        super().__init__(benchmark, metric)

    def evaluate(self, index):
        DeepSeekV3 = APIModel(
            name_or_path="deepseek-chat",
            base_url="https://api.deepseek.com",
            api_key="sk-xxx",
            sampling_params={
                "temperature": 0.7,
            }
        )

        data = self.read_json(self.ori_file_path)
        task_list = [
            {
                "informal_statement": item.get("NL"),
                "back_translation": item["sub_questions"][0].get("back_translation")
            }
            for item in data
        ]
        responses = DeepSeekV3.generate_batch(task_list, desc="NLI Check")
        del DeepSeekV3

        results = []
        for task, output in responses:
            results.append({"informal_statement": task["informal_statement"],
                            "back_translation": task["back_translation"],
                            "vote": output})
        file_path = self.file_path.replace("result.json", f"vote_{index}.json")
        self.write_json(file_path, results)

    def majority_voter(self):
        for i in range(1, 17):
            self.evaluate(i)
        self.merge_json_files([
            f"experiment/{self.benchmark}/majority_vote/vote_{i}.json" for i in range(1, 17)
        ], f"experiment/{self.benchmark}/majority_vote/majority_vote.json")
        
        data, majority_vote_data = self.read_json(self.ori_file_path), self.read_json(f"experiment/{self.benchmark}/majority_vote/majority_vote.json")
        total_groups, current_group = 0, []
        for item in majority_vote_data:
            current_group.append(item)
            if len(current_group) == 16:
                vote_counts = {"same": 0, "different": 0}
                for subitem in current_group:
                    vote_counts[subitem["vote"]] += 1
                majority_vote = "same" if vote_counts["same"] > vote_counts["different"] else "different"
                data[total_groups]["sub_questions"][0]["majority_vote"] = majority_vote
                current_group = []
                total_groups += 1
        self.write_json(self.file_path, data)
        majority_vote_result = self.summary(data, "majority_vote")
        results = {"majority_vote": majority_vote_result}
        self.write_json(self.acc_path, results)

        
class DefinitionalEqualityer(Evaluator):
    def __init__(self, benchmark, metric="definitional_equality"):
        super().__init__(benchmark, metric)

    def merge_headers(self, a, b):
        imports = set()
        other_lines = []

        def process_string(s: str):
            for line in s.splitlines():
                stripped_line = line.strip()
                if stripped_line.startswith("import "):
                    imports.add(stripped_line)
                else:
                    other_lines.append(line)
        process_string(a)
        process_string(b)

        sorted_imports = sorted(list(imports))
        merged_other_lines = []
        last_line_was_empty = False
        for line in other_lines:
            if line.strip() == "":
                if not last_line_was_empty:
                    merged_other_lines.append("")
                    last_line_was_empty = True
            else:
                merged_other_lines.append(line)
                last_line_was_empty = False
        
        result_parts = []
        if sorted_imports:
            result_parts.append("\n".join(sorted_imports))
        if sorted_imports and any(line.strip() for line in merged_other_lines):
            result_parts.append("")
        if merged_other_lines:
            result_parts.append("\n".join(merged_other_lines).rstrip())
        return "\n".join(result_parts)
    
    def verify_statement(self, formal_statements_list):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        start_time = time.time()
        lean4_scheduler = FLVerifier()
        request_id_list = lean4_scheduler.submit_all_request(formal_statements_list)
        outputs_list = lean4_scheduler.get_all_request_outputs(request_id_list)
        lean4_scheduler.close()
        end_time = time.time()
        print(f"Verification time: {end_time - start_time:.2f} seconds")
        return outputs_list
    
    def definitional_equalityer(self):
        formal_statements_list = []
        data = self.read_json(self.ori_file_path)
        for item in data:
            for subitem in item["sub_questions"]:
                header = self.merge_headers(item["header"], subitem["header"])
                label_statement = item["FL (Label)"].replace(item["name"], "tm_name1")
                predict_statement = subitem["FL (Prediction)"].replace(subitem["name"], "tm_name2")
                formal_statement = "example : tm_name1 = tm_name2 := by rfl"
                formal_statements_list.append(header + "\n" + label_statement + "\n" + predict_statement + "\n" + formal_statement)

        outputs_list = self.verify_statement(formal_statements_list)
        for index, outputs in enumerate(outputs_list):
            if outputs.get("pass") is True:
                data[index]["sub_questions"][0]["definitional_equality"] = "same"
            else:
                data[index]["sub_questions"][0]["definitional_equality"] = "different"
        self.write_json(self.file_path, data)
        definitional_equality_result = self.summary(data, "definitional_equality")
        results = {"definitional_equality": definitional_equality_result}
        self.write_json(self.acc_path, results)


CODEBLOCK_PATTERN = re.compile(r"```(?:.*?)\n(.*?)```", flags=re.DOTALL)
BANNED_TOKENS = ['sorry', 'admit', 'by_contra']
ALLOWED_TACTICS = [
    'apply',
    'by_contra',
    'cases\'',
    'constructor',
    'exact',
    'exact?',
    'ext',
    'have',
    'intro',
    'intros',
    'rw',
    'use',
]
EQUIV_PROVING_PROMPT_TEMPLATE = """Given two Lean 4 theorems, please prove `thm_Q` with `thm_P`.
You can only use the following tactics: {ALLOWED_TACTICS}
`thm_P` should be used at least once in the proof.
DO NOT add any extra explanation.
Here are some examples:

Input:
```
import Mathlib

open Topology Filter Real Complex TopologicalSpace Finset
open scoped BigOperators
noncomputable section


theorem thm_P : ¬ ∃ (x : ℚ), ( x ^ 2 = 12 ) :=
sorry

theorem thm_Q (q : ℚ ) :q ^ 2 ≠ 12 := by
```
Output:
```
exact (not_exists.mp thm_P) q
```

---

Input:
```
import Mathlib

open Fintype Subgroup Set Polynomial Ideal
open scoped BigOperators
noncomputable section


theorem thm_P {p q r : ℕ} {G : Type*} [Group G]
  [Fintype G]  (hpqr : p < q ∧ q < r)
  (hpqr1 : p.Prime ∧ q.Prime ∧ r.Prime)(hG : card G = p*q*r) :
  Nonempty (Sylow p G) ∨ Nonempty (Sylow q G) ∨ Nonempty (Sylow r G) :=
sorry

theorem thm_Q {p : ℕ } {q : ℕ } {r : ℕ } {G : Type u_1} [Group G] [Fintype G] (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hpq : p < q) (hqr : q < r) (hG : Fintype.card G = p * q * r) :Nonempty (Sylow p G) ∨ Nonempty (Sylow q G) ∨ Nonempty (Sylow r G) := by
```
Output:
```
exact thm_P (And.intro hpq hqr) (And.intro hp (And.intro hq hr)) hG
```

---

Input:
```
import Mathlib

open Fintype Complex Polynomial LinearMap FiniteDimensional Module Module.End
open scoped BigOperators


theorem thm_P {F V : Type*} [AddCommGroup V] [Field F]
  [Module F V] (S T : End F V) :
  (S * T).Eigenvalues = (T * S).Eigenvalues :=
sorry

theorem thm_Q {K : Type v} {V : Type w} [Field K] [AddCommGroup V] [Module K V] (S : Module.End K V) (T : Module.End K V) :Module.End.Eigenvalues (S * T) = Module.End.Eigenvalues (T * S) := by
```
Output:
```
exact @thm_P K V _ _ _ S T
```

---

Input:
```
import Mathlib

open Function Fintype Subgroup Ideal Polynomial Submodule Zsqrtd
open scoped BigOperators
noncomputable section


theorem thm_P
    {p : ℕ} {hp : Nat.Prime p} (h : ∃ r : ℕ, p = 2 ^ r + 1) :
    ∃ (k : ℕ), p = 2 ^ (2 ^ k) + 1 :=
sorry

theorem thm_Q {p : ℕ } (hp : Nat.Prime p) (h : ∃ (r : ℕ ), p = 2 ^ r + 1) :∃ (k : ℕ ), p = 2 ^ 2 ^ k + 1 := by
```
Output:
```
exact @thm_P p hp h
```

---

Input:
```
import Mathlib

open Fintype Set Real Ideal Polynomial
open scoped BigOperators
noncomputable section


theorem thm_P {G : Type*} [Group G]
  [Fintype G] (hG2 : Even (card G)) :
  ∃ (a : G), a ≠ 1 ∧ a = a⁻¹ :=
sorry

theorem thm_Q {G : Type*} [Group G] [Fintype G] (h : Fintype.card G % 2 = 0) :
    ∃ a : G, a ≠ 1 ∧ a = a⁻¹ := by
```
Output:
```
have hG : Even (card G) := by exact?
exact thm_P hG
```

---

According to the task description and examples, given the following two Lean 4 theorems, please prove `thm_Q` with `thm_P`.

Input:
```
{autoformalization_result}
```
Output:
""".replace('{ALLOWED_TACTICS}', '[' + ', '.join(ALLOWED_TACTICS) + ']')


class LocalModel:
    def __init__(self, name_or_path, gpus):
        self.name_or_path = name_or_path
        self.gpus = gpus
        self.sampling_params = {
            "max_tokens": 512,
            "temperature": 0.1,
            "top_p": 0.9,
            "stop": ["[UNUSED_TOKEN_146]", "[UNUSED_TOKEN_145]", "<|im_end|>"]
        }
        self.prompt = EQUIV_PROVING_PROMPT_TEMPLATE
        self._init_model()

    def _init_model(self):
        self.model = LLM(
            model=self.name_or_path,
            tensor_parallel_size=self.gpus,
            trust_remote_code=True,
            dtype="bfloat16",
            seed=42,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.name_or_path, trust_remote_code=True)

    def get_query(self, task):
        query = self.prompt.replace('{autoformalization_result}', task)
        message = [
                {"role": "system", "content": ""},
                {"role": "user", "content": query}
            ]
        return self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    
    def generate_batch(self, task_list):
        queries = [self.get_query(task) for task in task_list]
        responses = self.model.generate(queries, sampling_params=SamplingParams(**self.sampling_params))

        for response in responses:
            if response.outputs:
                mathes = re.findall(CODEBLOCK_PATTERN, response.outputs[0].text)
                response.outputs[0].text = "null"
                if mathes:
                    if any(token in mathes[0].strip() for token in BANNED_TOKENS):
                        continue
                    for l, line in enumerate(mathes[0].strip().split("\n")):
                        if any(tactic in line for tactic in ALLOWED_TACTICS):
                            response.outputs[0].text = mathes[0].strip()
                            break
        return [[o.text for o in response.outputs] for response in responses]

    def __del__(self):
        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()


class BEqer(DefinitionalEqualityer):
    def __init__(self, benchmark, metric="beq"):
        super().__init__(benchmark, metric)

    def update_result(self, new_data):
        old_data = self.read_json(self.file_path)
        for new_item in new_data:
            for old_item in old_data:
                if old_item["NL"] == new_item["NL"]:
                    if new_item["sub_questions"][0].get("beq") == "same":
                        old_item.update(new_item)
                        break
        self.write_json(self.file_path, old_data)

    def one_side_prepare(self, method, tag, replace_content):
        if method == "heuristic":
            data = self.read_json(self.ori_file_path)
        elif method == "llm":
            data = self.read_json(self.file_path)

        formal_statements_list = []
        for item in data:
            for subitem in item["sub_questions"]:
                if subitem.get("beq") == "same":
                    continue
                header = self.merge_headers(item["header"], subitem["header"])
                if tag == "label --> predict":
                    label_statement = item["FL (Label)"].replace(item["name"], "tm_nameP")
                    predict_statement = subitem["FL (Prediction)"].replace(subitem["name"], "tm_nameQ").replace(":= by sorry", replace_content)
                    formal_statements_list.append(header + "\n" + label_statement + "\n" + predict_statement)
                elif tag == "predict --> label":
                    predict_statement = subitem["FL (Prediction)"].replace(subitem["name"], "tm_nameP")
                    label_statement = item["FL (Label)"].replace(item["name"], "tm_nameQ").replace(":= by sorry", replace_content)
                    formal_statements_list.append(header + "\n" + predict_statement + "\n" + label_statement)
        return formal_statements_list
    
    def one_side_process(self, method, tag, outputs_list, responses=None):
        if method == "heuristic":
            data = self.read_json(self.ori_file_path)
        elif method == "llm":
            data = []
            total_data = self.read_json(self.file_path)
            for item in total_data:
                if item["sub_questions"][0].get("beq") == "same":
                    continue
                else:
                    data.append(item)
            
        for index, outputs in enumerate(outputs_list):
            if outputs.get("errors") == [] and outputs.get("pass") is True:
                infos = outputs.get("infos", "")
                for info in infos:
                    info_data = info.get("data", "")
                    if "Try this: exact" in info_data and "tm_nameP" in info_data:
                        data[index]["sub_questions"][0][tag] = True
                        data[index]["sub_questions"][0][tag+" | verified_code"] = outputs.get("verified_code")
                        data[index]["sub_questions"][0][tag+" | proof_content"] = responses[index][0] if responses else "exact?"
                        break
                    if method == "llm" and "tm_nameP" in responses[index][0]:
                        data[index]["sub_questions"][0][tag] = True
                        data[index]["sub_questions"][0][tag+" | verified_code"] = outputs.get("verified_code")
                        data[index]["sub_questions"][0][tag+" | proof_content"] = responses[index][0]
                        break
        return data

    def heuristic_exact(self):
        formal_statements_list1 = self.one_side_prepare("heuristic", "label --> predict", ":= by exact?")
        formal_statements_list2 = self.one_side_prepare("heuristic", "predict --> label", ":= by exact?")
        outputs_list1 = self.verify_statement(formal_statements_list1)
        outputs_list2 = self.verify_statement(formal_statements_list2)
        data1 = self.one_side_process("heuristic", "label --> predict", outputs_list1)
        data2 = self.one_side_process("heuristic", "predict --> label", outputs_list2)

        success_count = 0
        for index, item in enumerate(data1):
            if item["sub_questions"][0].get("label --> predict"):
                if data2[index]["sub_questions"][0].get("predict --> label"):
                    item["sub_questions"][0]["predict --> label | verified_code"] = data2[index]["sub_questions"][0]["predict --> label | verified_code"]
                    item["sub_questions"][0]["predict --> label | proof_content"] = data2[index]["sub_questions"][0]["predict --> label | proof_content"]
                    item["sub_questions"][0]["beq"] = "same"
                    del item["sub_questions"][0]["label --> predict"]
                    success_count += 1
                else:
                    del item["sub_questions"][0]["label --> predict"]
                    del item["sub_questions"][0]["label --> predict | verified_code"]
                    del item["sub_questions"][0]["label --> predict | proof_content"]
        print(f"Heuristic Exact Success Count: {success_count} / {len(data1)}.")
        self.write_json(self.file_path, data1)
    
    def llm_exact(self, i, InternLM):
        formal_statements_list1 = self.one_side_prepare("llm", "label --> predict", ":= by \n") 
        formal_statements_list2 = self.one_side_prepare("llm", "predict --> label", ":= by \n")
        responses1 = InternLM.generate_batch(formal_statements_list1)
        responses2 = InternLM.generate_batch(formal_statements_list2)
        for index, response in enumerate(responses1):
            formal_statements_list1[index] = formal_statements_list1[index]+response[0].strip()
            formal_statements_list2[index] = formal_statements_list2[index]+responses2[index][0].strip()
        outputs_list1 = self.verify_statement(formal_statements_list1)
        outputs_list2 = self.verify_statement(formal_statements_list2)
        data1 = self.one_side_process("llm", "label --> predict", outputs_list1, responses=responses1)
        data2 = self.one_side_process("llm", "predict --> label", outputs_list2, responses=responses2)

        success_count = 0
        for index, item in enumerate(data1):
            if item["sub_questions"][0].get("label --> predict"):
                if data2[index]["sub_questions"][0].get("predict --> label"):
                    item["sub_questions"][0]["predict --> label | verified_code"] = data2[index]["sub_questions"][0]["predict --> label | verified_code"]
                    item["sub_questions"][0]["predict --> label | proof_content"] = data2[index]["sub_questions"][0]["predict --> label | proof_content"]
                    item["sub_questions"][0]["beq"] = "same"
                    del item["sub_questions"][0]["label --> predict"]
                    success_count += 1
                else:
                    del item["sub_questions"][0]["label --> predict"]
                    del item["sub_questions"][0]["label --> predict | verified_code"]
                    del item["sub_questions"][0]["label --> predict | proof_content"]
        print(f"LLM Exact Success Count: {success_count} / {len(data1)} at iteration {i}.")
        self.update_result(data1)

    def beqer(self):
        self.heuristic_exact()
        InternLM = LocalModel(name_or_path="../Models/Shanghai_AI_Laboratory/internlm2-math-plus-20b", gpus=1)
        for i in range(1, 17):
            self.llm_exact(i, InternLM)
        del InternLM
        
        data = self.read_json(self.file_path)
        for item in data:
            for subitem in item["sub_questions"]:
                if "beq" not in subitem:
                    subitem["beq"] = "different"
        self.write_json(self.file_path, data)
        definitional_equality_result = self.summary(data, "beq")
        results = {"beq": definitional_equality_result}
        self.write_json(self.acc_path, results)


class TreeSimilarer(Evaluator):
    def __init__(self, benchmark, metric="gted"):
        super().__init__(benchmark, metric)

    def rewrite(self, threshold):
        data = self.read_json(self.file_path)
        for item in data:
            for subitem in item["sub_questions"]:
                subitem["gted"] = "same" if subitem["gted"] >= threshold else "different"
        return self.summary(data, "gted")

    def treesimilarer(self):
        results, step = [], 0.001
        for i in range(0, 1001):
            threshold = i * step
            gted_result = self.rewrite(threshold)
            results.append({
                "threshold": threshold,
                "gted": gted_result,
            })
        self.write_json(self.acc_path, results)
        self.find_best_element("gted")