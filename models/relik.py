from typing import Tuple
from relik import Relik
from relik.inference.data.objects import Span, Triplets, RelikOutput
from models.bert import get_bert_embedding
from schemas.node import ChildNode
from schemas.relation import Relation
import traceback

relik_pretrained = None


def initialize_relik():
    global relik_pretrained
    try:
        relik_pretrained = Relik.from_pretrained(
            "relik-ie/relik-relation-extraction-small"
        )
        print("Relik model loaded successfully")
    except Exception as e:
        print(f"Error initializing Relik model: {str(e)}")
        print(traceback.format_exc())


def process_relik_output(input_message: str) -> RelikOutput:
    global relik_pretrained
    if relik_pretrained is None:
        raise Exception("Relik model not initialized")

    return relik_pretrained(input_message)


def merge_spans_and_triplets(spans, labels, triplets) -> Tuple[list[ChildNode], list[Relation]]:
    merged_spans = []
    span_map = {}

    for label in set(labels):
        cluster_spans = [span for span, l in zip(spans, labels) if l == label]
        if len(cluster_spans) == 1:
            merged_span = cluster_spans[0]
        else:
            merged_text = " | ".join(sorted(set(span.text for span in cluster_spans)))
            merged_start = min(span.start for span in cluster_spans)
            merged_end = max(span.end for span in cluster_spans)
            merged_span = Span(start=merged_start, end=merged_end, label=cluster_spans[0].label, text=merged_text)

        merged_spans.append(merged_span)
        for span in cluster_spans:
            span_map[span] = merged_span

    merged_triplets = set()
    for triplet in triplets:
        new_subject = span_map.get(triplet.subject, triplet.subject)
        new_object = span_map.get(triplet.object, triplet.object)

        if new_subject != new_object:
            new_triplet = Triplets(
                subject=new_subject, label=triplet.label, object=new_object, confidence=triplet.confidence)
            merged_triplets.add(new_triplet)

    child_nodes = [
        ChildNode(
            triplet_source_id=span.text.strip(),
            text=span.text.strip(),
            embedding=get_bert_embedding(span.text.strip()),
            metadata=None,
        )
        for span in merged_spans
    ]

    relations = [
        Relation(
            source_id=triplet.subject.text.strip(),
            target_id=triplet.object.text.strip(),
            label=triplet.label.replace(" ", "_").upper(),
            metadata=None,
        )
        for triplet in merged_triplets
    ]

    return child_nodes, relations
