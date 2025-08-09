import re
import sys
import time
import json
import threading
import subprocess


class HoverExtractor:
    """
    Given Lean code, extract hover information using a persistent Lean's LSP server.
    """

    class IdGenerator:
        def __init__(self):
            self.current_id = 1
        def next(self):
            used_id = self.current_id
            self.current_id += 1
            return used_id

    def __init__(self, lean_bin="lean", virtual_uri="file:///virtual/code.lean"):
        self.lean_bin = lean_bin
        self.uri = virtual_uri
        self.language_id = "lean4"
        self.preamble = "import Mathlib\n" # Common preamble
        self.doc_version = 1 # Document version for LSP
        self.proc = None
        self.idgen = self.IdGenerator()
        self.response_dict = {}
        self.lock = threading.Lock()
        self.stdout_thread = None
        self.stderr_thread = None
        self.running = False

    def __enter__(self):
        """Starts the server and initializes it with the preamble."""
        self._start_server()
        self._initialize_session()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stops the server."""
        self._stop_server()

    def make_lsp_message(self, json_obj):
        content = json.dumps(json_obj)
        return f"Content-Length: {len(content)}\r\n\r\n{content}".encode("utf-8")

    def _start_server(self):
        try:
            self.proc = subprocess.Popen([self.lean_bin, "--server"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd="../ATLAS/src/repl")
        except FileNotFoundError:
            print("Failed to start Lean. Please ensure Lean is in your PATH.")
            sys.exit(1)
        except Exception as e:
            print(f"Error starting Lean: {e}")
            sys.exit(1)
        self.running = True
        self.stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self.stdout_thread.start()
        self.stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self.stderr_thread.start()
        print("Lean server started and warming up with Mathlib...")

    def _stop_server(self):
        print("Stopping Lean server...")
        self.running = False
        if self.proc:
            shutdown_req = {"jsonrpc": "2.0", "id": self.idgen.next(), "method": "shutdown"}
            self.proc.stdin.write(self.make_lsp_message(shutdown_req))
            self.proc.stdin.flush()
            time.sleep(0.1)
            exit_notify = {"jsonrpc": "2.0", "method": "exit"}
            self.proc.stdin.write(self.make_lsp_message(exit_notify))
            self.proc.stdin.flush()

            self.proc.terminate()
            self.proc.wait(timeout=2)
            self.proc = None
        if self.stdout_thread:
            self.stdout_thread.join(timeout=1)
        if self.stderr_thread:
            self.stderr_thread.join(timeout=1)

    def _read_stdout(self):
        try:
            f = self.proc.stdout
            while self.running:
                header = b""
                while not header.endswith(b"\r\n\r\n"):
                    chunk = f.read(1)
                    if not chunk or not self.running: return
                    header += chunk

                header_text = header.decode(errors="ignore")
                try:
                    content_length = int(header_text.split("Content-Length:")[1].strip())
                except (IndexError, ValueError):
                    continue

                body = f.read(content_length)
                try:
                    parsed = json.loads(body)
                    if "id" in parsed:
                        with self.lock:
                            self.response_dict[parsed["id"]] = parsed
                except json.JSONDecodeError:
                    continue
        except Exception:
            pass

    def _read_stderr(self):
        try:
            for line in iter(self.proc.stderr.readline, b""):
                if not self.running: break
        except Exception:
            pass

    def _lsp_request_and_wait(self, request, timeout=3.0):
        req_id = request["id"]
        with self.lock:
            if req_id in self.response_dict:
                del self.response_dict[req_id]

        self.proc.stdin.write(self.make_lsp_message(request))
        self.proc.stdin.flush()

        start_time = time.monotonic()
        while time.monotonic() - start_time < timeout:
            with self.lock:
                if req_id in self.response_dict:
                    return self.response_dict.pop(req_id)
            time.sleep(0.01)
        return None

    def _initialize_session(self):
        """Sends the initial LSP handshake and opens the virtual document with the preamble."""
        # 1. Initialize
        initialize_req = {
            "jsonrpc": "2.0", "id": self.idgen.next(), "method": "initialize",
            "params": {"processId": None, "rootUri": None, "capabilities": {}}
        }
        self._lsp_request_and_wait(initialize_req) # Wait for response to ensure server is ready

        # 2. Initialized
        initialized_notify = {"jsonrpc": "2.0", "method": "initialized", "params": {}}
        self.proc.stdin.write(self.make_lsp_message(initialized_notify))
        self.proc.stdin.flush()

        # 3. DidOpen with only the preamble
        didopen_req = {
            "jsonrpc": "2.0", "method": "textDocument/didOpen",
            "params": {
                "textDocument": {
                    "uri": self.uri, "languageId": self.language_id,
                    "version": self.doc_version, "text": self.preamble
                }
            }
        }
        self.proc.stdin.write(self.make_lsp_message(didopen_req))
        self.proc.stdin.flush()

        # Give Lean time to process Mathlib. This is a crucial wait.
        time.sleep(10) 
        print("Mathlib loaded. Server is ready.")

    def extract(self, code_snippet, out_path=None):
        """Extracts hover info for a new code snippet, reusing the existing server."""
        # Combine preamble with the new code
        full_code = self.preamble + code_snippet
        self.doc_version += 1

        # Use `didChange` to update the document content instead of `didOpen`
        didchange_req = {
            "jsonrpc": "2.0",
            "method": "textDocument/didChange",
            "params": {
                "textDocument": {"uri": self.uri, "version": self.doc_version},
                "contentChanges": [{"text": full_code}]
            }
        }
        self.proc.stdin.write(self.make_lsp_message(didchange_req))
        self.proc.stdin.flush()

        # Short sleep to allow the server to process the change
        time.sleep(0.5)

        extract_results = []
        lines = full_code.splitlines()
        preamble_lines = len(self.preamble.splitlines())

        # We only iterate over the new code snippet
        for line_idx_offset, line in enumerate(code_snippet.splitlines()):
            line_idx_actual = line_idx_offset + preamble_lines # Adjust line number
            for char_idx, _ in enumerate(line):
                hover_req = {
                    "jsonrpc": "2.0", "id": self.idgen.next(), "method": "textDocument/hover",
                    "params": {
                        "textDocument": {"uri": self.uri},
                        "position": {"line": line_idx_actual, "character": char_idx}
                    }
                }
                resp = self._lsp_request_and_wait(hover_req)

                new_entry = {"character": line[char_idx], "row": line_idx_actual, "column": char_idx}
                if resp and "result" in resp and resp["result"]:
                    new_entry.update(resp["result"])

                extract_results.append(new_entry)

        if out_path:
            with open(out_path, "w", encoding="utf-8") as out_f:
                for item in extract_results:
                    out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return extract_results


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