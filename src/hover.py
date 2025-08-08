import re
import sys
import time
import json
import threading
import subprocess


class HoverExtractor:
    """
    Given Lean code, extract hover information using Lean's LSP server.
    """

    class IdGenerator:
        def __init__(self):
            self.current_id = 1

        def next(self):
            used_id = self.current_id
            self.current_id += 1
            return used_id
    
    def __init__(self, lean_bin="lean", virtual_uri="file:///virtual/code.lean"):
        self.lean_bin = lean_bin  # Lean executable path
        self.uri = virtual_uri  # Virtual file URI
        self.language_id = "lean4"  # Language ID for Lean 4
        self.proc = None  # Lean process
        self.idgen = self.IdGenerator()  # ID generator
        self.response_dict = {}       # id -> response
        self.lock = threading.Lock()  # Lock for thread-safe access to response_dict
        self.stdout_thread = None  # Thread for reading stdout
        self.stderr_thread = None  # Thread for reading stderr
        self.running = False  # Flag to indicate if the server is running

    def make_lsp_message(self, json_obj):
        content = json.dumps(json_obj)  # Convert JSON object to string
        return f"Content-Length: {len(content)}\r\n\r\n{content}".encode("utf-8")  # Create LSP message format

    def start_server(self):  
        try:
            self.proc = subprocess.Popen([self.lean_bin, "--server"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd="../ATLAS/src/repl") # Start Lean process
        except FileNotFoundError:
            print("Failed to start Lean. Please ensure Lean is in your PATH.")
            sys.exit(1)
        except Exception as e:
            print("Error starting Lean: ", e)
            sys.exit(1)
        self.running = True
        self.stdout_thread = threading.Thread(target=self.read_stdout, daemon=True)  # Thread for reading LSP responses
        self.stdout_thread.start()
        self.stderr_thread = threading.Thread(target=self.read_stderr, daemon=True)  # Thread for reading stderr
        self.stderr_thread.start()

    def stop_server(self):
        self.running = False
        if self.proc:
            self.proc.terminate()
            self.proc = None
        if self.stdout_thread:
            self.stdout_thread.join(timeout=1)
        if self.stderr_thread:
            self.stderr_thread.join(timeout=1)

    def read_stdout(self):
        try:
            f = self.proc.stdout  # Get the stdout of the Lean process
            while self.running:
                header = b""
                while not header.endswith(b"\r\n\r\n"):
                    chunk = f.read(1)
                    if not chunk or not self.running:
                        return
                    header += chunk
                header_text = header.decode(errors="ignore")
                try:
                    content_length = int(header_text.split("Content-Length:")[1].split("\r\n")[0].strip())
                except Exception as e:
                    continue

                body = f.read(content_length)
                try:
                    parsed = json.loads(body)
                    if "id" in parsed:
                        with self.lock:
                            self.response_dict[parsed["id"]] = parsed
                except Exception:
                    continue
        except Exception:
            pass

    def read_stderr(self, stderr=None):
        if stderr is None and self.proc:
            stderr = self.proc.stderr
        if stderr is None:
            return
        try:
            for line in iter(stderr.readline, b""):
                if not self.running:
                    break
        except Exception:
            pass

    def lsp_request_and_wait(self, request, timeout=3.0):
        req_id = request["id"]
        with self.lock:
            if req_id in self.response_dict:
                del self.response_dict[req_id]
        self.proc.stdin.write(self.make_lsp_message(request))
        self.proc.stdin.flush()

        waited = 0.0
        interval = 0.01
        while waited < timeout:
            with self.lock:
                resp = self.response_dict.get(req_id)
                if resp is not None:
                    return resp
            time.sleep(interval)
            waited += interval
        return None  

    def init_lsp(self, code):
        initialize_req = {
            "jsonrpc": "2.0",
            "id": self.idgen.next(),
            "method": "initialize",
            "params": {
                "processId": None,
                "rootUri": None,
                "capabilities": {},
            }
        }
        self.proc.stdin.write(self.make_lsp_message(initialize_req))
        self.proc.stdin.flush()
        time.sleep(0.3)

        initialized_notify = {
            "jsonrpc": "2.0",
            "method": "initialized",
            "params": {}
        }
        self.proc.stdin.write(self.make_lsp_message(initialized_notify))
        self.proc.stdin.flush()
        time.sleep(0.1)

        didopen_req = {
            "jsonrpc": "2.0",
            "method": "textDocument/didOpen",
            "params": {
                "textDocument": {
                    "uri": self.uri,
                    "languageId": self.language_id,
                    "version": 1,
                    "text": code
                }
            }
        }
        self.proc.stdin.write(self.make_lsp_message(didopen_req))
        self.proc.stdin.flush()
        time.sleep(0.3)

    def extract(self, code, out_path=None):
        self.start_server()
        try:
            self.init_lsp(code)
            extract_results = []
            lines = code.splitlines()
            for line_idx, line in enumerate(lines):
                for char_idx, _ in enumerate(line):
                    hover_req = {
                        "jsonrpc": "2.0",
                        "id": self.idgen.next(),
                        "method": "textDocument/hover",
                        "params": {
                            "textDocument": {"uri": self.uri},
                            "position": {"line": line_idx, "character": char_idx}
                        }
                    }
                    resp = self.lsp_request_and_wait(hover_req)
                    new_entry = {"character": line[char_idx], "row": line_idx, "column": char_idx}           
                    if resp and "result" in resp and resp["result"]:
                        hover_content = resp["result"]
                        new_entry.update(hover_content)
                    else:
                        hover_content = None
                    extract_results.append(new_entry)
            if out_path:
                with open(out_path, "w", encoding="utf-8") as out_f:
                    for item in extract_results:
                        out_f.write(json.dumps(item, ensure_ascii=False) + "\n")   
            return extract_results
        finally:
            self.stop_server()


class HoverRewriter:
    """
    Rewrite the formal statement according to the hover information.
    """

    def process_with_range_contents(self, extract_results):
        char_map = {}
        for item in extract_results:
            if len(item) == 3:  # No hover information
                continue
            char, range, contents = item.get("character"), item.get("range"), item.get("contents")
            k = (
                range["start"]["line"], range["start"]["character"],
                range["end"]["line"], range["end"]["character"]
            )
            # Store both the character and its contents
            char_map.setdefault(k, {"chars": [], "contents": contents})
            char_map[k]["chars"].append(char)

        process_results = [
            json.dumps(
                {
                    "".join(char_map[k]["chars"]): [[k[0], k[1]], [k[2], k[3]]],
                    "contents": char_map[k]["contents"]
                },
                ensure_ascii=False
            )
            for k in sorted(char_map)
        ]
        return process_results

    def get_theorem_name_of_sorry_line(self, process_results):
        sorry_line = None
        for item_str in process_results:
            item = json.loads(item_str)
            key, value = next(iter(item.items()))
            if key.lower() == "sorry":
                sorry_line = value[0][0] 
                break 
        if sorry_line is None:
            return None

        elements_on_line = []
        for item_str in process_results:
            item = json.loads(item_str)
            key, value = next(iter(item.items()))
            start_line, start_char = value[0][0], value[0][1]
            if start_line == sorry_line:
                elements_on_line.append((start_char, item))

        first_element = min(elements_on_line, key=lambda x: x[0])
        return first_element[1]        

    def rewrite(self, extract_results):
        process_results = self.process_with_range_contents(extract_results)
        first_element = self.get_theorem_name_of_sorry_line(process_results)
        contents = first_element.get("contents")
        matches = re.findall(r"```lean\n(.*?)```", contents.get("value"), re.DOTALL)
        value = matches[0].replace("\n", "").strip() if matches else "TBD"
        return "theorem " + value + " := by sorry"
    

class HoverProcessor(HoverRewriter):
    """
    Process the hover information further.
    """
    
    def keep_from_element(self, lst, key):
        for i, s in enumerate(lst):
            if key in s:
                return lst[i+1:-2]

    def add_underline(self, key):  # , --> _,
        brackets = {"(": ")", "[": "]", "{": "}"}
        if key and key[0] in brackets and key[-1] == brackets[key[0]]:
            inner_content = key[1:-1]
            if re.fullmatch(r"[,\s]*", inner_content) and "," in key and any(c.isspace() for c in key):
                inner_new = inner_content.replace(",", "_,")
                inner_new += "_"
                return key[0] + inner_new + key[-1]
        elif re.fullmatch(r"[,\s]*", key) and "," in key and any(c.isspace() for c in key):
                key = key.replace(",", "_,")
                key += "_"
                return key
        return key

    def reduce_space(self, key, symbol, sep):
        symbol_index = key.find(symbol)
        if symbol_index == -1:
            return key  
        sep_index = key.find(sep, symbol_index)
        if sep_index == -1:
            return key  
        between = key[symbol_index + len(symbol): sep_index]
        if set(between) <= {" "} and len(between) > 0:
            new_between = between[:-1]
            return key[:symbol_index + len(symbol)] + new_between + key[sep_index:]
        return key

    def key_concatenation(self, process_results):
        concatenated_results, last_key = [], None
        for result in process_results:
            data = json.loads(result)
            key = next(iter(data))
            value = data[key]
            key = "_._" if key == "." else key
            key = "_._ " if key == ". " else key
            key = self.add_underline(key)
            if key.isspace():
                if last_key is not None:
                    key = last_key + key
                    del concatenated_results[-1]
            else:
                last_key = key
            concatenated_result = {key: value}
            concatenated_results.append(json.dumps(concatenated_result, ensure_ascii=False))
        return concatenated_results

    def key_modify(self, process_results):
        concatenated_results, modified_results = self.key_concatenation(process_results), []
        for result in concatenated_results:
            data = json.loads(result)
            key = next(iter(data))
            value = data[key]  
            key = self.reduce_space(key, "∏", ",")
            key = self.reduce_space(key, "∑", ",")
            key = self.reduce_space(key, "{", "|")
            key = self.reduce_space(key, "fun", "=>")
            key = self.reduce_space(key, "λ", "=>")
            key = re.sub(r"∀ +,", "∀ ,", key)  # the number of space is not accurate, but it is consistent
            key = re.sub(r"∃ +,", "∃ ,", key)  # the number of space is not accurate, but it is consistent
            key = key.replace("(", "").replace(")", "")  # Remove parentheses ()
            key = key.replace(" ", " _ ").replace(":", " _ : _ ") if "_," not in key else key
            if key != "":  # Avoid empty keys
                modified_result = {key: value}
                modified_results.append(json.dumps(modified_result, ensure_ascii=False))
        return modified_results
        
    def process(self, extract_results, out_path=None):
        char_map = {}
        for item in extract_results:
            if len(item) == 3:
                continue
            char, range = item.get("character"), item.get("range")
            k = (
                range["start"]["line"], range["start"]["character"],
                range["end"]["line"], range["end"]["character"]
            )
            char_map.setdefault(k, []).append(char)
        process_results = [
            json.dumps(
                {"".join(char_map[k]): [[k[0], k[1]], [k[2], k[3]]]},
                ensure_ascii=False
            )
            for k in sorted(char_map)
        ]

        temp_results = self.process_with_range_contents(extract_results)
        first_element = self.get_theorem_name_of_sorry_line(temp_results)
        process_results = self.keep_from_element(process_results, next(iter(first_element)))

        modified_results = self.key_modify(process_results)
        if out_path:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(modified_results) + "\n")
        return modified_results