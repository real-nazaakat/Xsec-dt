
from typing import Iterable, Tuple, Dict, Any, List, Optional

try:
    import networkx as nx
    HAS_NX = True
except Exception:
    HAS_NX = False

DependencyRow = Tuple[str, str, str]


class SimpleDiGraph:
    def __init__(self):
        self._adj = {}
        self.nodes = set()

    def add_node(self, n):
        self.nodes.add(n)
        self._adj.setdefault(n, {})

    def add_edge(self, u, v, **attrs):
        self.add_node(u)
        self.add_node(v)
        self._adj[u][v] = dict(attrs)

    def successors(self, n):
        return list(self._adj.get(n, {}).keys())

    def predecessors(self, n):
        return [u for u, nbrs in self._adj.items() if n in nbrs]

    def has_cycle(self):
        visited, stack = set(), set()

        def visit(n):
            if n in stack:
                return True
            if n in visited:
                return False
            visited.add(n)
            stack.add(n)
            for v in self.successors(n):
                if visit(v):
                    return True
            stack.remove(n)
            return False

        return any(visit(n) for n in self.nodes)

    def topological_sort(self):
        indeg = {n: 0 for n in self.nodes}
        for u in self._adj:
            for v in self._adj[u]:
                indeg[v] += 1

        q = [n for n, d in indeg.items() if d == 0]
        order = []

        while q:
            n = q.pop(0)
            order.append(n)
            for m in self.successors(n):
                indeg[m] -= 1
                if indeg[m] == 0:
                    q.append(m)

        if len(order) != len(self.nodes):
            raise ValueError("Graph has cycles")

        return order


class DependencyGraphBuilder:
    def __init__(self, use_networkx: Optional[bool] = None):
        self.use_networkx = HAS_NX if use_networkx is None else (use_networkx and HAS_NX)

    def build(self, rows: Iterable[DependencyRow], node_attrs: Dict[str, Dict[str, Any]] = None):
        node_attrs = node_attrs or {}

        if self.use_networkx:
            G = nx.DiGraph()
            for n, a in node_attrs.items():
                G.add_node(n, **a)
            for s, t, d in rows:
                G.add_edge(s, t, dependency_type=d)
            return G

        G = SimpleDiGraph()
        for n in node_attrs:
            G.add_node(n)
        for s, t, d in rows:
            G.add_edge(s, t, dependency_type=d)
        return G
