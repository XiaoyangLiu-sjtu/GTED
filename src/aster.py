import json
import math
import uuid
import pathlib
from graphviz import Digraph
from typing import List, Dict
from apted import APTED, Config


class ASTBuilder:
    class Node:
        def __init__(self, content, range):
            self.content = content
            self.range = range
            self.children = []

        def to_dict(self):
            return {
                "content": self.content,
                "children": [child.to_dict() for child in self.children]
            }

    @staticmethod
    def range_contains(a, b):
        sl1, sc1, el1, ec1 = a
        sl2, sc2, el2, ec2 = b
        starts_before = (sl1 < sl2) or (sl1 == sl2 and sc1 <= sc2)
        ends_after = (el1 > el2) or (el1 == el2 and ec1 >= ec2)
        return starts_before and ends_after and a != b

    def read_jsonl(self, process_results):
        nodes = []
        for item in process_results:
            obj = json.loads(item)
            assert len(obj) == 1
            content, range = next(iter(obj.items()))
            (sl, sc), (el, ec) = range
            nodes.append(self.Node(content, (sl, sc, el, ec)))
        return nodes

    def build_tree(self, nodes):
        root = self.Node("theorem", (-1, -1, math.inf, math.inf))
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

    def add_underscores(self, data):
        if isinstance(data, dict):
            if 'children' in data and len(data['children']) > 0:
                if data.get('content') == 'theorem':
                    outer_children = data['children'][0]
                    if 'children' in outer_children:
                        num_children = len(outer_children['children'])
                        outer_children['content'] = outer_children['content'] + '_' * num_children
                for child in data['children']:
                    self.add_underscores(child)
        elif isinstance(data, list):
            for item in data:
                self.add_underscores(item)

    def build(self, process_results, out_path=None):
        nodes = self.read_jsonl(process_results)
        tree_results = self.build_tree(nodes).to_dict()
        if tree_results == {"content": "theorem", "children": []}:
            return tree_results
        content, number = tree_results["children"][0]["content"], len(tree_results["children"][0]["children"])-1
        parts = content.split(":", 1)
        new_left = " _ " * number if number > 0 else ""
        new_content = new_left + ":" + parts[1] if len(parts) > 1 else content
        tree_results["children"][0]["content"] = new_content
        if out_path:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(tree_results, f, ensure_ascii=False, indent=4)
        return tree_results


class ASTVisualizer:
    def __init__(self,node_shape="box", font_name="Helvetica", font_size="20"):
        self.node_shape = node_shape
        self.font_name = font_name
        self.font_size = font_size

    def _build_dot(self, node, g: Digraph, parent_id=None):
        nid = f"n{uuid.uuid4().hex}"
        content = node.get("content", "")
        safe_content = content.replace('"', r'\"')
        g.node(nid, safe_content)
        if parent_id:
            g.edge(parent_id, nid)
        for ch in node.get("children", []):
            self._build_dot(ch, g, nid)

    def visualize(self, tree_results, out_path=None, out_format="png"):
        g = Digraph("AST", graph_attr={"rankdir": "TB"}, node_attr={"shape": self.node_shape, "fontname": self.font_name, "fontsize": self.font_size})
        self._build_dot(tree_results, g)
        if out_path:
            out_path = pathlib.Path(out_path)
            g.render(out_path.stem, out_path.parent, format=out_format, cleanup=True)


class Node:
    __slots__ = ("content", "children", "hypothesis_freecost", "variable_freecost")
    def __init__(self, content: str, children: List["Node"] = None, 
                 hypothesis_freecost: bool = False, variable_freecost: bool = False):
        self.content: str = content
        self.children: List[Node] = children or []
        self.hypothesis_freecost: bool = hypothesis_freecost
        self.variable_freecost: bool = variable_freecost

    def get_children(self) -> List["Node"]:
        return self.children


class ASTUtils:
    @staticmethod
    def load_json_ast(data: Dict) -> Node:
        def build(d: Dict) -> Node:
            return Node(
                d["content"], 
                [build(c) for c in d.get("children", [])],
                d.get("hypothesis_freecost", False),
                d.get("variable_freecost", False)
            )
        return build(data)

    @staticmethod
    def tree_size(node: Node) -> int:
        return 1 + sum(ASTUtils.tree_size(ch) for ch in node.children)

    @staticmethod
    def subtree_code(node: Node) -> str:
        if not node.children:
            return node.content
        return f"{node.content}(" + ",".join(ASTUtils.subtree_code(c) for c in node.children) + ")"


class _APConfig(Config):
    def children(self, node: Node) -> List[Node]:
        return node.children

    def insert(self, node: Node, _=None) -> int:
        return 1

    def delete(self, node: Node, _=None) -> int:
        return 1

    def rename(self, n1: Node, n2: Node) -> int:
        # If either node has hypothesis_freecost or variable_freecost, cost is 0
        # If do not want to rename nodes with freecost, uncomment the following lines, then it is standard TED
        if (n1.hypothesis_freecost and n2.hypothesis_freecost) or \
           (n1.variable_freecost and n2.variable_freecost):
            print(f"Renaming {n1.content} to {n2.content} with freecost, cost is 0")
            return 0
        if n1.content == n2.content:
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
        code = ASTUtils.subtree_code(node)
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


class ASTSimilarer:
    def similarer(self, data_a=None, data_b=None, path_a=None, path_b=None) -> Dict[str, float]:
        if data_a is None and data_b is None:
            with open(path_a, "r", encoding="utf-8") as f:
                data_a = json.load(f)
            with open(path_b, "r", encoding="utf-8") as f:
                data_b = json.load(f)
        self.tree_a = ASTUtils.load_json_ast(data_a)
        self.tree_b = ASTUtils.load_json_ast(data_b)

        ted = TreeEditDistance(self.tree_a, self.tree_b).compute_distance()
        size_a = ASTUtils.tree_size(self.tree_a)
        size_b = ASTUtils.tree_size(self.tree_b)
        ted_similarity = 1.0 - (ted / max(size_a, size_b)) if max(size_a, size_b) > 0 else 1.0
        jacc = SubtreeJaccard.calculate(self.tree_a, self.tree_b)
        return {
            "size_a": size_a,
            "size_b": size_b,
            "ted": ted,
            "ted_similarity": ted_similarity,
            "subtree_jaccard": jacc,
        }