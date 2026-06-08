"""Builds knowledge graph relationships and assigns tags."""

from typing import Dict, List


def build_relationships(raw_data: Dict) -> List[Dict]:
    """Convert raw Wikidata payload into graph edges."""
    edges = []
    for item in raw_data.get("results", {}).get("bindings", []):
        source = item.get("item", {}).get("value")
        label = item.get("itemLabel", {}).get("value")
        if source and label:
            edges.append({"id": source, "label": label, "tags": []})
    return edges


def assign_tags(nodes: List[Dict]) -> List[Dict]:
    """Assign tags to nodes based on relationships and labels."""
    for node in nodes:
        label = node.get("label", "").lower()
        node["tags"] = ["entity"]
        if "hashtag" in label:
            node["tags"].append("hashtag")
    return nodes


if __name__ == "__main__":
    example = {"results": {"bindings": [{"item": {"value": "http://www.wikidata.org/entity/Q1"}, "itemLabel": {"value": "Example"}}]}}
    print(assign_tags(build_relationships(example)))
