import re
import json
import math
import uuid
import pathlib
from graphviz import Digraph
from typing import List, Dict
from apted import APTED, Config


class OPTBuilder:
    class Node:
        def __init__(self, formal_content, range, hover_information):
            self.formal_content = formal_content
            self.range = range
            self.hover_information = hover_information
            self.children = []

        def to_dict(self):
            return {
                "formal_content": self.formal_content,
                "hover_information": self.hover_information,
                "children": [child.to_dict() for child in self.children]
            }

    @staticmethod
    def range_contains(a, b):
        sl1, sc1, el1, ec1 = a
        sl2, sc2, el2, ec2 = b
        starts_before = (sl1 < sl2) or (sl1 == sl2 and sc1 <= sc2)
        ends_after = (el1 > el2) or (el1 == el2 and ec1 >= ec2)
        return starts_before and ends_after and a != b

    def read_jsonl_data(self, process_results):
        nodes = []
        for item in process_results:
            obj = json.loads(item)
            formal_content = next(iter(obj))
            range_content = obj[formal_content]  
            hover_information = obj["hover_information"]
            (sl, sc), (el, ec) = range_content
            nodes.append(self.Node(formal_content, (sl, sc, el, ec), hover_information))
        return nodes

    def build_tree(self, nodes):
        root = self.Node("theorem", (-1, -1, math.inf, math.inf), "")
        nodes.sort(key=lambda n: (n.range[0], n.range[1], -n.range[2], -n.range[3]))
        for n in nodes:
            parent = root
            for cand in nodes:
                if cand is n:
                    continue
                if self.range_contains(cand.range, n.range):
                    if parent is root or self.range_contains(parent.range, cand.range):
                        parent = cand
            parent.children.append(n)

        def strip_range(node):
            node.range = None
            for ch in node.children:
                strip_range(ch)
        strip_range(root)
        return root
    
    def add_placeholder_further(self, tree_results, character, delimiter):
        for item in tree_results:
            formal_content = item.get("formal_content")
            if character in formal_content and delimiter in formal_content:
                num_children = len(item.get("children", []))
                placeholders = " _" * (num_children - 1)

                escaped_char = re.escape(character)
                escaped_delim = re.escape(delimiter)
                pattern = rf"({escaped_char})(.*?)({escaped_delim})"
                
                replacement_logic = lambda m: m.group(1) + placeholders + m.group(3)
                new_content = re.sub(pattern, replacement_logic, formal_content, count=1)
                if delimiter == "=>":
                    new_content = new_content.replace("=>", " =>")
                item["formal_content"] = new_content.replace("_:", ":")

            if "children" in item and item["children"]:
                self.add_placeholder_further(item["children"], character, delimiter)
        return tree_results

    def build(self, process_results, out_path, informal_statement, formal_statement, reorganized_formal_statement):
        nodes = self.read_jsonl_data(process_results)
        tree_results = self.build_tree(nodes).to_dict()
        tree_results = [tree_results]
        if tree_results != [{"formal_content": "theorem", "children": [], "hover_information": ""}]:
            # placeholder further
            self.add_placeholder_further(tree_results, "_", ":")
            self.add_placeholder_further(tree_results, "∏", ",")
            self.add_placeholder_further(tree_results, "∑", ",")
            self.add_placeholder_further(tree_results, "∀", ",")
            self.add_placeholder_further(tree_results, "∃", ",")
            self.add_placeholder_further(tree_results, "fun", "=>")
            self.add_placeholder_further(tree_results, "λ", "=>")

        tree_results.insert(0, {"informal_statement": informal_statement, "formal_statement": formal_statement, "reorganized_formal_statement": reorganized_formal_statement})
        if out_path:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(tree_results, f, ensure_ascii=False, indent=4)
        return tree_results[1]


class OPTVisualizer:
    def __init__(self,node_shape="box", font_name="Helvetica", font_size="20"):
        self.node_shape = node_shape
        self.font_name = font_name
        self.font_size = font_size

    def _build_dot(self, node, g: Digraph, parent_id=None, fltree=True):
        nid = f"n{uuid.uuid4().hex}"
        formal_content = node.get("formal_content", "") if fltree else node.get("informal_content", "")
        safe_formal_content = formal_content.replace('"', r'\"')
        g.node(nid, safe_formal_content)
        if parent_id:
            g.edge(parent_id, nid)
        for ch in node.get("children", []):
            self._build_dot(ch, g, nid, fltree=fltree)

    def visualize(self, tree_results, out_path=None, out_format="png", fltree=True):
        g = Digraph("OPT", graph_attr={"rankdir": "TB"}, node_attr={"shape": self.node_shape, "fontname": self.font_name, "fontsize": self.font_size})
        self._build_dot(tree_results, g, fltree=fltree)
        if out_path:
            out_path = pathlib.Path(out_path)
            g.render(out_path.stem, out_path.parent, format=out_format, cleanup=True)


class Node:
    __slots__ = ("formal_content", "children", "hypothesis_freecost", "variable_freecost")
    def __init__(self, formal_content: str, children: List["Node"] = None, 
                 hypothesis_freecost: bool = False, variable_freecost: bool = False):
        self.formal_content: str = formal_content
        self.children: List[Node] = children or []
        self.hypothesis_freecost: bool = hypothesis_freecost
        self.variable_freecost: bool = variable_freecost

    def get_children(self) -> List["Node"]:
        return self.children


class OPTUtils:
    @staticmethod
    def load_json_opt(data: Dict) -> Node:
        def build(d: Dict) -> Node:
            return Node(
                d["formal_content"], 
                [build(c) for c in d.get("children", [])],
                d.get("hypothesis_freecost", False),
                d.get("variable_freecost", False)
            )
        return build(data)

    @staticmethod
    def tree_size(node: Node) -> int:
        return 1 + sum(OPTUtils.tree_size(ch) for ch in node.children)

    @staticmethod
    def subtree_code(node: Node) -> str:
        if not node.children:
            return node.formal_content
        return f"{node.formal_content}(" + ",".join(OPTUtils.subtree_code(c) for c in node.children) + ")"


class _APConfig(Config):
    def children(self, node: Node) -> List[Node]:
        return node.children

    def insert(self, node: Node, _=None) -> int:
        return 1

    def delete(self, node: Node, _=None) -> int:
        return 1

    def rename(self, n1: Node, n2: Node) -> int:
        # Uncomment the following lines to enable freecost renaming
        # If either node has hypothesis_freecost or variable_freecost, cost is 0
        # If do not want to rename nodes with freecost, uncomment the following lines, then it is standard TED
        # if (n1.hypothesis_freecost and n2.hypothesis_freecost) or \
        #    (n1.variable_freecost and n2.variable_freecost):
        #     print(f"Renaming {n1.formal_content} to {n2.formal_content} with freecost, cost is 0")
        #     return 0
        if n1.formal_content == n2.formal_content:
            return 0
        else:
            return 1


class TreeEditDistance:
    def __init__(self, a: Node, b: Node):
        self.a = a
        self.b = b

    def compute_distance(self) -> int:
        return APTED(self.a, self.b, _APConfig()).compute_edit_distance()


class SubtreeJaccard:
    @staticmethod
    def collect_codes(node: Node, multiset: Dict[str, int]):
        code = OPTUtils.subtree_code(node)
        multiset[code] = multiset.get(code, 0) + 1
        for ch in node.children:
            SubtreeJaccard.collect_codes(ch, multiset)

    @staticmethod
    def calculate(a: Node, b: Node) -> float:
        ca, cb = {}, {}
        SubtreeJaccard.collect_codes(a, ca)
        SubtreeJaccard.collect_codes(b, cb)
        inter = sum(min(ca[k], cb.get(k, 0)) for k in ca)
        union = sum(ca.values()) + sum(cb.values()) - inter
        return inter / union if union else 1.0


class OPTSimilarer:
    def similarer(self, data_a=None, data_b=None, path_a=None, path_b=None) -> Dict[str, float]:
        if data_a is None and data_b is None:
            with open(path_a, "r", encoding="utf-8") as f:
                data_a = json.load(f)
            with open(path_b, "r", encoding="utf-8") as f:
                data_b = json.load(f)
        self.tree_a = OPTUtils.load_json_opt(data_a)
        self.tree_b = OPTUtils.load_json_opt(data_b)

        ted = TreeEditDistance(self.tree_a, self.tree_b).compute_distance()
        size_a = OPTUtils.tree_size(self.tree_a)
        size_b = OPTUtils.tree_size(self.tree_b)
        ted_similarity = 1.0 - (ted / max(size_a, size_b)) if max(size_a, size_b) > 0 else 1.0
        jacc = SubtreeJaccard.calculate(self.tree_a, self.tree_b)
        return {
            "size_a": size_a,
            "size_b": size_b,
            "ted": ted,
            "ted_similarity": ted_similarity,
            "subtree_jaccard": jacc,
        }