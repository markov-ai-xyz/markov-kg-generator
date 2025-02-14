from dotenv import load_dotenv
from neo4j import GraphDatabase
import os
import traceback

load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
neo4j_driver = None


def initialize_neo4j():
    global neo4j_driver
    try:
        neo4j_driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
        )
        print("Neo4j loaded successfully")
    except Exception as e:
        print(f"Error initializing Neo4j: {str(e)}")
        print(traceback.format_exc())


def create_parent_node(node):
    with neo4j_driver.session() as session:
        with session.begin_transaction() as tx:
            tx.run(
                """
                MERGE (n:Entity:Parent {name: $name, embedding: $embedding, node_id: $node_id})
            """,
                name=node.text.strip(),
                embedding=node.embedding,
                node_id=node.node_id,
            )


def create_entities_batch(nodes):
    with neo4j_driver.session() as session:
        with session.begin_transaction() as tx:
            tx.run(
                """
                UNWIND $nodes AS node
                MERGE (n:Entity:Child {name: node.name, embedding: node.embedding})
            """,
                nodes=[
                    {"name": node.text.strip(), "embedding": node.embedding}
                    for node in nodes
                ],
            )


def create_filial_relations_batch(parent_node, child_nodes):
    with neo4j_driver.session() as session:
        with session.begin_transaction() as tx:
            tx.run(
                """
                UNWIND $child_nodes AS child_node
                MATCH (parent:Entity:Parent {node_id: $parent_node_id})
                MATCH (child:Entity:Child {name: child_node.name})
                MERGE (parent)-[:MENTIONS]->(child)
                """,
                child_nodes=[
                    {"name": child_node.text.strip(), "embedding": child_node.embedding}
                    for child_node in child_nodes
                ],
                parent_node_id=parent_node.node_id
            )


def create_triplet_relations_batch(triplets):
    with neo4j_driver.session() as session:
        with session.begin_transaction() as tx:
            tx.run(
                """
                UNWIND $triplets AS triplet
                MATCH (a:Entity {name: triplet.source_id})
                MATCH (b:Entity {name: triplet.target_id})
                CALL apoc.merge.relationship(a, triplet.label, {}, {}, b, {})
                YIELD rel
                RETURN count(rel)
            """,
                triplets=[
                    {
                        "source_id": triplet.source_id.strip(),
                        "target_id": triplet.target_id.strip(),
                        "label": triplet.label.strip(),
                    }
                    for triplet in triplets
                ],
            )
