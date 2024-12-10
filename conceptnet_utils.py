import random

from file_io import read_json_file, read_txt_file


PROTOTYPE_MAP = read_json_file("conceptnet/form_of.json")
RELATED_MAP = read_json_file("conceptnet/related_to.json")

ENGLISH_WORDS = set(line.strip().lower() for line in read_txt_file("vocabulary/full_network.txt"))
ALL_CONCEPTS = list(line.strip().lower() for line in read_txt_file("conceptnet/concepts.txt"))


def get_prototype(query_concept):
    """Remove tense and plurality for query concept using ConceptNet."""
    query_concept = query_concept.lower()

    if query_concept.endswith(("s", "ed", "ing")) and not query_concept.endswith("ss"):
        if query_concept in PROTOTYPE_MAP:
            return PROTOTYPE_MAP[query_concept]

    return query_concept


def get_related_concepts(query_concept):
    """Get related concepts for query concept using ConceptNet."""
    related_concepts = list()

    query_concept = get_prototype(query_concept)
    if query_concept in RELATED_MAP:
        all_related_concepts = RELATED_MAP[query_concept]
        for related_concept in all_related_concepts:
            related_concept = get_prototype(related_concept)
            if related_concept in ENGLISH_WORDS:
                if related_concept != query_concept:
                    if related_concept not in related_concepts:
                        related_concepts.append(related_concept)

    return related_concepts


def get_random_concept():
    """Get a random concept from ConceptNet."""
    random_concept = random.choice(ALL_CONCEPTS)
    random_concept = get_prototype(random_concept)
    return random_concept
