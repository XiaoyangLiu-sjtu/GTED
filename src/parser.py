import time
from src.verifier import FLVerifier


class Reorganizer:
    def _check_brackets_balanced(self, string):
        stack, bracket_pairs = [], {"(": ")", "[": "]", "{": "}"}
        for char in string:
            if char in bracket_pairs:
                stack.append(char)
            elif char in bracket_pairs.values():
                if not stack or bracket_pairs[stack.pop()] != char:
                    return False
        return not stack 

    def _find_colons_after_matching_parentheses(self, string):
        stack, brackets, colon_positions = [], {"(": ")", "[": "]", "{": "}"}, []
        for i, char in enumerate(string):
            if char in brackets:  
                stack.append(char)
            elif char in brackets.values(): 
                if stack and brackets[stack[-1]] == char:
                    stack.pop()
            elif char == ":" and not stack:  
                colon_positions.append(i)
        return colon_positions

    def _extract_brackets_content(self, string):
        stack, bracket_pairs = [], {"(": ")", "[": "]", "{": "}"}
        start, matches = -1, []
        for i, char in enumerate(string):
            if char in bracket_pairs:
                if not stack:
                    start = i
                stack.append(char)
            elif char in bracket_pairs.values():
                if stack and bracket_pairs[stack[-1]] == char:
                    stack.pop()
                    if not stack:
                        matches.append(string[start:i+1])
        return matches

    def _split_elements(self, input_list):
        result_list = []
        for item in input_list:
            if item[0] == "(" and item[-1] == ")" and ":" in item:
                item = item[1:-1].strip()
                prefix, suffix = item.split(":", 1)
                elements = prefix.strip().split()
                for element in elements:
                    result_list.append(f"({element.strip()} : {suffix.strip()})")
            else:
                result_list.append(item)
        return result_list
    
    def parse_formal_statement(self, formal_statement):
        keyword_position = formal_statement.find("theorem")
        theorem_content = formal_statement[keyword_position:]
        theorem_content = " ".join(theorem_content.split())
        split_theorem_content = theorem_content.split(" ")
        if ".{" in split_theorem_content[1] and "}" not in split_theorem_content[1]:
            for i in range(2, len(split_theorem_content)):
                if "}" in split_theorem_content[i]:
                    theorem_name = " ".join(split_theorem_content[:i+1])
                    theorem_lines = split_theorem_content[i+1:]
                    break
        else:
            theorem_name = " ".join(split_theorem_content[:2])
            theorem_lines = split_theorem_content[2:]  

        theorem_variables_hypotheses, theorem_conclusion, temp = [], "", ""
        for index, line in enumerate(theorem_lines):
            temp = line if not temp else f"{temp} {line}" 
            balanced = self._check_brackets_balanced(temp) 
            if balanced:
                colon_mark = self._find_colons_after_matching_parentheses(temp) 
                if colon_mark == []:
                    matches = self._extract_brackets_content(temp)
                    theorem_variables_hypotheses.extend(matches)
                    temp = ""
                else:
                    matches = self._extract_brackets_content(temp[:colon_mark[0]])
                    theorem_variables_hypotheses.extend(matches)
                    theorem_conclusion += temp[colon_mark[0]:]
                    remaining_content = " ".join(theorem_lines[index + 1:])
                    if remaining_content:
                        theorem_conclusion += f" {remaining_content}"
                    break
            else:
                continue

        return {
            "theorem_name": theorem_name,
            "theorem_variables_hypotheses": theorem_variables_hypotheses,
            "theorem_conclusion": theorem_conclusion,
        }

    def reorganize_statement(self, formal_statement):
        parse_result = self.parse_formal_statement(formal_statement)
        input_list = parse_result.get("theorem_variables_hypotheses")
        output_list = self._split_elements(input_list) if input_list is not None else []
        
        reorganized_formal_statement = (
            parse_result["theorem_name"] + " "
            + (" ".join(output_list) + " " if output_list else "")
            + parse_result["theorem_conclusion"]
        )
        return reorganized_formal_statement
    

class NameExtractor(Reorganizer):
    def run_proofsteps(self, data):
        formal_statements_list = [
            data["header"] + "\n" + data["formal_statement"] + "\n" + data["tactic"]
        ]

        start_time = time.time()
        lean4_scheduler = FLVerifier(max_concurrent_requests=1)
        request_id_list = lean4_scheduler.submit_all_request(formal_statements_list)
        outputs_list = lean4_scheduler.get_all_request_outputs(request_id_list)
        lean4_scheduler.close()
        end_time = time.time()
        print(f"Verification time: {end_time - start_time:.2f} seconds")
        return outputs_list[0]

    def parse_formal_statement_further(self, header, formal_statement):
        parse_result = self.parse_formal_statement(formal_statement)
        reorganized_formal_statement = (
            parse_result["theorem_name"] + "\n"
                + ("\n".join(parse_result["theorem_variables_hypotheses"]) + "\n" if parse_result["theorem_variables_hypotheses"] else "")
                + parse_result["theorem_conclusion"]
            ).replace("sorry", "")
        
        count, tactic, theorem_varibles, theorem_hypotheses = 0, "", [], []
        for index, item in enumerate(parse_result["theorem_variables_hypotheses"]):
            if item.startswith("{") or item.startswith("["):
                parse_result["theorem_variables_hypotheses"][index] = item+" | Type"
            else:
                item = item.split(":", 1)[1][:-1]
                tactic += "#check " + item + "\n"
                count += 1
        temp_data = {"header": header, "formal_statement": reorganized_formal_statement, "tactic": tactic, "count": count, "result": parse_result}

        output = self.run_proofsteps(temp_data)
        mark_content = output["infos"]
        for i in range(temp_data["count"]):
            try:
                temp = mark_content[i]["data"]
                last_colon_index = temp.rindex(" : ")
                mark = temp[last_colon_index+3:].strip()
            except:
                mark = "Type"
            for index_inner, item in enumerate(temp_data["result"]["theorem_variables_hypotheses"]):
                if item.endswith("Type") or item.endswith("Prop"):
                    continue
                else:
                    if "Prop" in mark:
                        temp_data["result"]["theorem_variables_hypotheses"][index_inner] = item+" | Prop"
                    else:
                        temp_data["result"]["theorem_variables_hypotheses"][index_inner] = item+" | Type"
                    break
            
        theorem_varibles = [item.replace(" | Type", "") for item in temp_data["result"]["theorem_variables_hypotheses"] if item.endswith("Type")]
        theorem_hypotheses = [item.replace(" | Prop", "") for item in temp_data["result"]["theorem_variables_hypotheses"] if item.endswith("Prop")]
        
        return {
            "theorem_name": temp_data["result"]["theorem_name"],
            "theorem_variables": theorem_varibles,
            "theorem_hypotheses": theorem_hypotheses,
            "theorem_conclusion": temp_data["result"]["theorem_conclusion"]
        }
    
    def name_extractor(self, header, formal_statement):
        header = "import Mathlib\n" + header
        parse_result = self.parse_formal_statement_further(header, formal_statement)
        result = {}
        variable = parse_result["theorem_variables"]
        result["variables_name"] = [
            item.split(":")[0].replace(" ", "").replace("(", "")
            for item in variable
        ]
        hypotheses = parse_result["theorem_hypotheses"]
        result["hypotheses_name"] = [
            item.split(":")[0].replace(" ", "").replace("(", "")
            for item in hypotheses
        ]
        return result