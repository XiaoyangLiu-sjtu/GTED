import re
import sys
import time
import json
import threading
import subprocess
from typing import Any, Dict, List, Optional


class HoverExtractor:
    """
    Given Lean code, extract hover information using a persistent Lean's LSP server.
    This class is designed to be used as a context manager for efficiency.
    """

    class IdGenerator:
        """Simple sequential ID generator for LSP requests."""
        def __init__(self):
            self.current_id = 1
        def next(self) -> int:
            used_id = self.current_id
            self.current_id += 1
            return used_id

    def __init__(self, lean_bin: str = "lean", virtual_uri: str = "file:///virtual/code.lean"):
        self.lean_bin = lean_bin
        self.uri = virtual_uri
        self.language_id = "lean4"
        self.preamble = "import Mathlib\n"
        self.doc_version = 1
        self.proc: Optional[subprocess.Popen] = None
        self.idgen = self.IdGenerator()
        self.response_dict: Dict[int, Any] = {}
        self.lock = threading.Lock()
        self.stdout_thread: Optional[threading.Thread] = None
        self.stderr_thread: Optional[threading.Thread] = None
        self.running = False
        self.diagnostics_event = threading.Event()  # For robust synchronization

    def __enter__(self):
        """Starts the server and initializes it with the preamble."""
        self._start_server()
        self._initialize_session()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stops the server."""
        self._stop_server()

    def make_lsp_message(self, json_obj: Dict[str, Any]) -> bytes:
        """Formats a JSON object into an LSP message."""
        content = json.dumps(json_obj)
        return f"Content-Length: {len(content)}\r\n\r\n{content}".encode("utf-8")

    def _start_server(self):
        """Launches the Lean server as a subprocess."""
        try:
            self.proc = subprocess.Popen(
                [self.lean_bin, "--server"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd="../ATLAS/src/repl"  # Adjust this path as needed
            )
        except FileNotFoundError:
            print("Failed to start Lean. Please ensure the 'lean' command is in your PATH.")
            sys.exit(1)
        except Exception as e:
            print(f"Error starting Lean: {e}")
            sys.exit(1)
        
        self.running = True
        self.stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self.stdout_thread.start()
        self.stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self.stderr_thread.start()

    def _stop_server(self):
        """Sends shutdown notifications and terminates the Lean server process."""
        print("Stopping Lean server...")
        self.running = False
        if self.proc:
            try:
                shutdown_req = {"jsonrpc": "2.0", "id": self.idgen.next(), "method": "shutdown"}
                self.proc.stdin.write(self.make_lsp_message(shutdown_req))
                self.proc.stdin.flush()
                time.sleep(0.1)
                exit_notify = {"jsonrpc": "2.0", "method": "exit"}
                self.proc.stdin.write(self.make_lsp_message(exit_notify))
                self.proc.stdin.flush()
                self.proc.terminate()
                self.proc.wait(timeout=2)
            except (IOError, BrokenPipeError, ValueError):
                # Process might already be dead, which is fine
                pass
            finally:
                self.proc = None
        
        if self.stdout_thread:
            self.stdout_thread.join(timeout=1)
        if self.stderr_thread:
            self.stderr_thread.join(timeout=1)

    def _read_stdout(self):
        """
        Reads and parses LSP messages from stdout.
        
        This method now checks for 'textDocument/publishDiagnostics' messages
        to signal that the server has finished processing a file.
        """
        try:
            while self.running and self.proc and self.proc.stdout:
                header_lines = []
                content_length = -1
                while self.running:
                    line = self.proc.stdout.readline()
                    if not line:
                        return  # End of stream
                    header_lines.append(line)
                    if line.strip() == b'':
                        break  # End of headers
                    if line.lower().startswith(b'content-length:'):
                        content_length = int(line.split(b':')[1].strip())

                if content_length == -1:
                    continue

                body = self.proc.stdout.read(content_length)
                try:
                    parsed = json.loads(body.decode("utf-8"))
                    if "id" in parsed:
                        with self.lock:
                            self.response_dict[parsed["id"]] = parsed
                    elif parsed.get("method") == "textDocument/publishDiagnostics":
                        self.diagnostics_event.set()
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
        except (IOError, ValueError):
            pass

    def _read_stderr(self):
        """Reads the stderr stream to prevent the process buffer from filling up."""
        try:
            while self.running and self.proc and self.proc.stderr:
                line = self.proc.stderr.readline()
                if not line:
                    break
        except (IOError, ValueError):
            pass

    def _lsp_request_and_wait(self, request: Dict[str, Any], timeout: float = 3.0) -> Optional[Dict[str, Any]]:
        """Sends an LSP request and waits for a response."""
        req_id = request["id"]
        with self.lock:
            # Clear any stale responses for this ID
            self.response_dict.pop(req_id, None)

        if not (self.proc and self.proc.stdin):
            return None

        try:
            self.proc.stdin.write(self.make_lsp_message(request))
            self.proc.stdin.flush()
        except (IOError, BrokenPipeError):
            return None # Server process likely died

        start_time = time.monotonic()
        while time.monotonic() - start_time < timeout:
            with self.lock:
                if req_id in self.response_dict:
                    return self.response_dict.pop(req_id)
            time.sleep(0.01)
        return None

    def _initialize_session(self):
        """Sends the initial LSP handshake and opens the virtual document."""
        print("Lean server started, initializing session...")
        initialize_req = {
            "jsonrpc": "2.0", "id": self.idgen.next(), "method": "initialize",
            "params": {"processId": None, "rootUri": None, "capabilities": {}}
        }
        self._lsp_request_and_wait(initialize_req)

        initialized_notify = {"jsonrpc": "2.0", "method": "initialized", "params": {}}
        self.proc.stdin.write(self.make_lsp_message(initialized_notify))
        self.proc.stdin.flush()

        didopen_req = {
            "jsonrpc": "2.0", "method": "textDocument/didOpen",
            "params": {
                "textDocument": {"uri": self.uri, "languageId": self.language_id, "version": self.doc_version, "text": self.preamble}
            }
        }
        
        self.diagnostics_event.clear()
        self.proc.stdin.write(self.make_lsp_message(didopen_req))
        self.proc.stdin.flush()

        print("Warming up with Mathlib...")
        ready = self.diagnostics_event.wait(timeout=45.0) # Timeout for Mathlib
        if not ready:
            print("Warning: Timed out waiting for initial diagnostics. Server might be slow or failing.")
        print("Mathlib loaded. Server is ready.")

    def extract(self, code_snippet: str, out_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Extracts hover info for a new code snippet by updating the server's state."""
        full_code = self.preamble + code_snippet
        self.doc_version += 1

        didchange_req = {
            "jsonrpc": "2.0", "method": "textDocument/didChange",
            "params": {
                "textDocument": {"uri": self.uri, "version": self.doc_version},
                "contentChanges": [{"text": full_code}]
            }
        }
        
        self.diagnostics_event.clear()
        self.proc.stdin.write(self.make_lsp_message(didchange_req))
        self.proc.stdin.flush()

        ready = self.diagnostics_event.wait(timeout=15.0)
        if not ready:
            print(f"Warning: Timed out waiting for diagnostics on snippet: {code_snippet[:60]}...")

        extract_results = []
        preamble_lines = self.preamble.count('\n')

        for line_idx_offset, line in enumerate(code_snippet.splitlines()):
            line_idx_actual = line_idx_offset + preamble_lines
            for char_idx, char in enumerate(line):
                hover_req = {
                    "jsonrpc": "2.0", "id": self.idgen.next(), "method": "textDocument/hover",
                    "params": {
                        "textDocument": {"uri": self.uri},
                        "position": {"line": line_idx_actual, "character": char_idx}
                    }
                }
                resp = self._lsp_request_and_wait(hover_req)

                new_entry = {"character": char, "row": line_idx_actual, "column": char_idx+1}
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
    For 3.1 Syntax Standardization: Theorem Rewriting.
    """

    def process_range_contents(self, extract_results):
        """
        Process the extracted results to map characters to their range and contents.
        """
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

    def get_theorem_name_element(self, process_results):
        """
        Find the theorem name element.
        """
        last_element = json.loads(process_results[-1])
        _, value = next(iter(last_element.items()))
        theorem_name_line = value[0][0]

        candidates_element = []
        for item_str in process_results:
            item = json.loads(item_str)
            _, value = next(iter(item.items()))
            start_line, start_char = value[0][0], value[0][1]
            if start_line == theorem_name_line:
                candidates_element.append((start_char, item))
        return min(candidates_element, key=lambda x: x[0])[1]

    def rewrite(self, extract_results):
        process_results_range = self.process_range_contents(extract_results)
        theorem_name_element = self.get_theorem_name_element(process_results_range)
        contents = theorem_name_element.get("contents")
        matches = re.findall(r"```lean\n(.*?)```", contents.get("value"), re.DOTALL)
        value = matches[0].replace("\n", "").strip() if matches else "TBD: something wrong!"
        return f"theorem {value} := by sorry"

        
class HoverProcessor(HoverRewriter):
    """
    Process the hover information further.
    For 3.2 OPT Construction: Placeholder Representation & Parentheses Removal.
    """
    
    def remove_header(self, lst, key):
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

    def key_concatenation(self, process_results_name):
        concatenated_results, last_key = [], None
        for result in process_results_name:
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
            concatenated_result = {key: value, "contents": data["contents"]["value"]}
            concatenated_results.append(json.dumps(concatenated_result, ensure_ascii=False))
        return concatenated_results

    def add_placeholder(self, process_results_name):
        concatenated_results, modified_results = self.key_concatenation(process_results_name), []
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
        process_results_range = self.process_range_contents(extract_results)
        theorem_name_element = self.get_theorem_name_element(process_results_range)
        process_results_name = self.remove_header(process_results_range, next(iter(theorem_name_element)))

        process_results_placeholder = self.add_placeholder(process_results_name)
        if out_path:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(process_results_placeholder) + "\n")
        return process_results_placeholder