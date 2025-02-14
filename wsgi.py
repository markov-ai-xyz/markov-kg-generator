from flask import Flask, request, jsonify
from models.bert import initialize_bert, get_bert_embedding
from models.relik import initialize_relik, process_relik_output, merge_spans_and_triplets
from models.neo4j import (
    initialize_neo4j,
    create_parent_node,
    create_entities_batch,
    create_filial_relations_batch,
    create_triplet_relations_batch,
)
from models.sklearn import cluster_spans
from schemas.node import ParentNode
from embeddings.openai import generate_text_embeddings
from utils.parsing_utils import is_relevant
import traceback


# TODO: Add auth to every endpoint
def create_app():
    flask = Flask(__name__)

    with flask.app_context():
        initialize_bert()
        initialize_relik()
        initialize_neo4j()

    @flask.route("/graph-db", methods=["POST"])
    def load_to_graph_db():
        payload = request.get_json()
        if not payload:
            return jsonify({"error": "Invalid request payload"}), 400

        input_message = payload.get("input")
        if not input_message:
            return jsonify({"error": "Input message is required"}), 400

        try:
            text_embeddings = generate_text_embeddings(input_message)
            parent_node = ParentNode(
                node_id="parent_1",
                source_id="source_1",
                text=input_message,
                embedding=text_embeddings,
                metadata="Some metadata about the parent node.",
            )
            create_parent_node(parent_node)

            relik_out = process_relik_output(input_message)
            spans = relik_out.spans
            triplets = relik_out.triplets

            relevant_spans = [span for span in spans if is_relevant(span, spans)]
            labels = cluster_spans(relevant_spans, get_bert_embedding)
            child_nodes, relations = merge_spans_and_triplets(relevant_spans, labels, triplets)

            create_entities_batch(child_nodes)
            create_triplet_relations_batch(relations)
            create_filial_relations_batch(parent_node, child_nodes)

            return jsonify({"response": "OK"}), 200

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print(traceback.format_exc())
            return jsonify({"output": f"An error occurred: {str(e)}"}), 500

    return flask


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
