from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import networkx as nx
from pyvis.network import Network


TYPE_COLORS = {
    "Topic": "#1f77b4",
    "Level": "#ff7f0e",
    "Training": "#2ca02c",
    "Document": "#d62728",
    "Chunk": "#9467bd",
    "Entity": "#17becf",
}

EDGE_COLORS = {
    "has_entity": "#f4b400",
    "contains": "#999999",
    "part_of": "#999999",
    "mentions": "#2ca02c",
    "related_to": "#7f7f7f",
    "supports": "#1f77b4",
    "contradicts": "#d62728",
}


def _read_jsonl(file_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_graph(topic_dir: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    nodes_path = topic_dir / "nodes.jsonl"
    edges_path = topic_dir / "edges.jsonl"
    if not nodes_path.is_file():
        raise FileNotFoundError(f"Missing nodes file: {nodes_path}")
    if not edges_path.is_file():
        raise FileNotFoundError(f"Missing edges file: {edges_path}")

    nodes = {str(node["id"]): node for node in _read_jsonl(nodes_path)}
    edges = _read_jsonl(edges_path)
    return nodes, edges


def _node_label(node: dict[str, Any]) -> str:
    props = node.get("properties", {}) or {}
    label = props.get("label") or props.get("name") or node.get("id", "")
    node_type = node.get("type", "")
    return f"{label}\n[{node_type}]"


def _edge_label(edge: dict[str, Any]) -> str:
    props = edge.get("properties", {}) or {}
    relation = props.get("relation") or props.get("relation_type") or edge.get("type", "")
    return str(relation)


def _build_nx_graph(nodes: dict[str, dict[str, Any]], edges: list[dict[str, Any]]) -> nx.DiGraph:
    graph = nx.DiGraph()
    for node_id, node in nodes.items():
        graph.add_node(node_id, **node)
    for edge in edges:
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        if not source or not target:
            continue
        graph.add_edge(source, target, **edge)
    return graph


def _topic_node_id(nodes: dict[str, dict[str, Any]]) -> str:
    topic_nodes = [node_id for node_id, node in nodes.items() if node.get("type") == "Topic"]
    if not topic_nodes:
        raise ValueError("No Topic node found in graph")
    return sorted(topic_nodes)[0]


def _resolve_center_node(nodes: dict[str, dict[str, Any]], center: str | None) -> str:
    if center:
        if center in nodes:
            return center
        normalized = center.strip().lower()
        for node_id, node in nodes.items():
            props = node.get("properties", {}) or {}
            label = str(props.get("label") or props.get("name") or "").strip().lower()
            if label == normalized:
                return node_id
            if node_id.strip().lower() == normalized:
                return node_id
        raise ValueError(f"Could not find center node matching: {center}")

    entity_nodes = [node_id for node_id, node in nodes.items() if node.get("type") == "Entity"]
    if entity_nodes:
        return sorted(entity_nodes)[0]

    by_degree = sorted(nodes.keys())
    if not by_degree:
        raise ValueError("Graph is empty")
    return by_degree[0]


def _collect_subgraph(graph: nx.DiGraph, center: str, depth: int, include_entity_neighbors: bool = True) -> nx.DiGraph:
    visited = {center}
    frontier = deque([(center, 0)])

    while frontier:
        current, current_depth = frontier.popleft()
        if current_depth >= depth:
            continue
        neighbors = set(graph.predecessors(current)) | set(graph.successors(current))
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.append((neighbor, current_depth + 1))

    subgraph = graph.subgraph(visited).copy()

    # Optionally bring in direct entity neighbors one more hop away to make entity views useful.
    if include_entity_neighbors and graph.nodes[center].get("type") == "Entity":
        extra = set()
        for predecessor in graph.predecessors(center):
            extra.add(predecessor)
        for successor in graph.successors(center):
            extra.add(successor)
        for node_id in extra:
            subgraph.add_node(node_id, **graph.nodes[node_id])
            for _, target, data in graph.out_edges(node_id, data=True):
                if target in subgraph:
                    subgraph.add_edge(node_id, target, **data)
            for source, _, data in graph.in_edges(node_id, data=True):
                if source in subgraph:
                    subgraph.add_edge(source, node_id, **data)

    return subgraph


def _collect_entity_overview(
    graph: nx.DiGraph,
    max_entities: int,
    center: str | None = None,
) -> nx.DiGraph:
    topic_nodes = [node_id for node_id, data in graph.nodes(data=True) if data.get("type") == "Topic"]
    if not topic_nodes:
        raise ValueError("No Topic node found in graph")
    topic_id = sorted(topic_nodes)[0]

    entity_nodes = []
    for node_id, data in graph.nodes(data=True):
        if data.get("type") != "Entity":
            continue
        props = data.get("properties", {}) or {}
        entity_nodes.append(
            (
                node_id,
                int(props.get("occurrence_count", 0) or 0),
                float(props.get("confidence", 0) or 0),
                str(props.get("label") or node_id),
            )
        )

    if center:
        normalized_center = center.strip().lower()
        center_id = None
        for node_id, _, _, label in entity_nodes:
            if node_id.lower() == normalized_center or label.strip().lower() == normalized_center:
                center_id = node_id
                break
        if center_id is None:
            raise ValueError(f"Could not find entity center matching: {center}")

        selected = {center_id}
        for neighbor in set(graph.predecessors(center_id)) | set(graph.successors(center_id)):
            if graph.nodes[neighbor].get("type") == "Entity":
                selected.add(neighbor)
        selected_entities = sorted(selected)
    else:
        entity_nodes.sort(key=lambda item: (-item[1], -item[2], item[3]))
        selected_entities = [node_id for node_id, _, _, _ in entity_nodes[:max_entities]]

    subgraph = nx.DiGraph()
    subgraph.add_node(topic_id, **graph.nodes[topic_id])

    selected_set = set(selected_entities)
    for node_id in selected_entities:
        subgraph.add_node(node_id, **graph.nodes[node_id])
        subgraph.add_edge(
            topic_id,
            node_id,
            source=topic_id,
            target=node_id,
            type="has_entity",
            properties={"origin": "entity_overview"},
        )

    for source, target, data in graph.edges(data=True):
        if source not in selected_set or target not in selected_set:
            continue
        edge_type = str(data.get("type", ""))
        if edge_type in {"mentions", "contains", "part_of"}:
            continue
        subgraph.add_edge(source, target, **data)

    return subgraph


def _add_to_pyvis(network: Network, subgraph: nx.DiGraph) -> None:
    for node_id, data in subgraph.nodes(data=True):
        node_type = data.get("type", "Unknown")
        props = data.get("properties", {}) or {}
        title_lines = [f"<b>ID:</b> {node_id}", f"<b>Type:</b> {node_type}"]
        for key in ("label", "name", "entity_type", "training_id", "document_id", "chunk_id", "confidence"):
            if key in props and props[key] not in (None, ""):
                title_lines.append(f"<b>{key}:</b> {props[key]}")
        network.add_node(
            node_id,
            label=props.get("label") or props.get("name") or node_id,
            title="<br>".join(title_lines),
            color=TYPE_COLORS.get(node_type, "#cccccc"),
            shape="dot" if node_type != "Chunk" else "box",
            size=24 if node_type == "Topic" else (18 if node_type == "Entity" else 12),
        )

    for source, target, data in subgraph.edges(data=True):
        edge_type = data.get("type", "related_to")
        props = data.get("properties", {}) or {}
        title_lines = [f"<b>Type:</b> {edge_type}"]
        for key in ("origin", "relation", "relation_type", "confidence"):
            if key in props and props[key] not in (None, ""):
                title_lines.append(f"<b>{key}:</b> {props[key]}")
        network.add_edge(
            source,
            target,
            label=edge_type,
            title="<br>".join(title_lines),
            color=EDGE_COLORS.get(edge_type, "#888888"),
            arrows="to",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a topic graph with NetworkX and PyVis.")
    parser.add_argument("--graph-root", type=str, default="resources/artifacts/DC_data/graph")
    parser.add_argument("--topic", type=str, default="hydraulics")
    parser.add_argument(
        "--center",
        type=str,
        default="",
        help="Center node id or label to visualize around. If omitted, the first Entity node is used.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help="How many hops away from the center node to include.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output HTML path. Defaults to <topic>_graph.html inside the topic folder.",
    )
    parser.add_argument(
        "--show-labels",
        action="store_true",
        default=False,
        help="Keep labels visible for all nodes. By default labels are compact in the HTML view.",
    )
    parser.add_argument(
        "--entity-only",
        action="store_true",
        default=False,
        help="Show topic plus entity nodes and entity-to-entity semantic edges only. Excludes chunk/document/training nodes.",
    )
    parser.add_argument(
        "--max-entities",
        type=int,
        default=40,
        help="Maximum number of entity nodes to include in entity-only overview when no center is given.",
    )
    args = parser.parse_args()

    topic_dir = Path(args.graph_root) / args.topic.lower()
    nodes, edges = _load_graph(topic_dir)
    nx_graph = _build_nx_graph(nodes, edges)

    if args.entity_only:
        topic_id = _topic_node_id(nodes)
        center = args.center or topic_id
        subgraph = _collect_entity_overview(
            nx_graph,
            max_entities=max(1, args.max_entities),
            center=args.center or None,
        )
    else:
        center = _resolve_center_node(nodes, args.center or None)
        subgraph = _collect_subgraph(nx_graph, center=center, depth=max(0, args.depth))

    html_path = Path(args.output) if args.output else (topic_dir / f"{args.topic.lower()}_graph.html")
    html_path.parent.mkdir(parents=True, exist_ok=True)

    net = Network(height="900px", width="100%", directed=True, bgcolor="#0f172a", font_color="white")
    net.toggle_physics(True)
    net.barnes_hut(gravity=-20000, central_gravity=0.3, spring_length=180, spring_strength=0.01, damping=0.09)
    _add_to_pyvis(net, subgraph)

    if not args.show_labels:
        for node in net.nodes:
            if node.get("title"):
                node["label"] = node["label"][:28]

    net.show_buttons(filter_=["physics"])
    net.write_html(str(html_path), open_browser=False)

    print(f"Graph written to: {html_path}")
    print(f"Center node: {center}")
    print(f"Nodes in view: {subgraph.number_of_nodes()}")
    print(f"Edges in view: {subgraph.number_of_edges()}")


if __name__ == "__main__":
    main()
