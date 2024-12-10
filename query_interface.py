

def relation_extraction(sentence, head_entity, tail_entity, prompt_type="ic"):
    """The interface that constitutes prompt to guide llms to conduct relation extraction."""
    ic_incontext = (
        "<Instruction> Select the most suitable relation between the given head and tail entities in the"
        " given sentence. The relation type must be chosen from the candidate relations. </Instruction>\n"
        "<Instance>\n"
        "Given Sentence: These apples are from the store at the corner.\n"
        "Head Entity: apples\n"
        "Tail Entity: store\n"
        "Candidate Relations: message-topic, entity-origin, entity-destination, content-container,"
        " cause-effect, component-whole, member-collection, instrument-agency, product-producer\n"
        "Relation Between the Head Entity and Tail Entity: entity-origin\n"
        "</Instance>\n"
        "<Instance>\n"
        "Given Sentence: These apples are moved to the store at the corner.\n"
        "Head Entity: apples\n"
        "Tail Entity: store\n"
        "Candidate Relations: message-topic, entity-origin, entity-destination, content-container,"
        " cause-effect, component-whole, member-collection, instrument-agency, product-producer\n"
        "Relation Between the Head Entity and Tail Entity: entity-destination\n"
        "</Instance>\n"
        "Hint: complete the remaining content and maintain consistency with the format of the above examples.\n"
        "<Instance>\n")

    ic1_incontext = (
        "<Instruction> Select the most suitable relation between the given head and tail entities in the"
        " given sentence. The relation type must be chosen from the candidate relations. </Instruction>\n"
        "<Instance>\n"
        "Given Sentence: These apples are from the store at the corner.\n"
        "Head Entity: apples\n"
        "Tail Entity: store\n" 
        "Candidate Relations: message-topic, entity-origin, entity-destination, content-container,"
        " cause-effect, component-whole, member-collection, instrument-agency, product-producer\n"
        "Relation Between the Head Entity and Tail Entity: entity-origin\n"
        "</Instance>\n"
        "Hint: complete the remaining content and maintain consistency with the format of the above examples.\n"
        "<Instance>\n")

    ic3_incontext = (
        "<Instruction> Select the most suitable relation between the given head and tail entities in the"
        " given sentence. The relation type must be chosen from the candidate relations. </Instruction>\n"
        "<Instance>\n"
        "Given Sentence: These apples are from the store at the corner.\n"
        "Head Entity: apples\n"
        "Tail Entity: store\n"
        "Candidate Relations: message-topic, entity-origin, entity-destination, content-container,"
        " cause-effect, component-whole, member-collection, instrument-agency, product-producer\n"
        "Relation Between the Head Entity and Tail Entity: entity-origin\n"
        "</Instance>\n"
        "<Instance>\n"
        "Given Sentence: These apples are moved to the store at the corner.\n"
        "Head Entity: apples\n"
        "Tail Entity: store\n"
        "Candidate Relations: message-topic, entity-origin, entity-destination, content-container,"
        " cause-effect, component-whole, member-collection, instrument-agency, product-producer\n"
        "Relation Between the Head Entity and Tail Entity: entity-destination\n"
        "</Instance>\n"
        "<Instance>\n"
        "Given Sentence: These books talk about causal discovery.\n"
        "Head Entity: books\n"
        "Tail Entity: causal discovery\n"
        "Candidate Relations: message-topic, entity-origin, entity-destination, content-container,"
        " cause-effect, component-whole, member-collection, instrument-agency, product-producer\n"
        "Relation Between the Head Entity and Tail Entity: message-topic\n"
        "</Instance>\n"
        "Hint: complete the remaining content and maintain consistency with the format of the above examples.\n"
        "<Instance>\n")

    ic4_incontext = (
        "<Instruction> Select the most suitable relation between the given head and tail entities in the"
        " given sentence. The relation type must be chosen from the candidate relations. </Instruction>\n"
        "<Instance>\n"
        "Given Sentence: These apples are from the store at the corner.\n"
        "Head Entity: apples\n"
        "Tail Entity: store\n"
        "Candidate Relations: message-topic, entity-origin, entity-destination, content-container,"
        " cause-effect, component-whole, member-collection, instrument-agency, product-producer\n"
        "Relation Between the Head Entity and Tail Entity: entity-origin\n"
        "</Instance>\n"
        "<Instance>\n"
        "Given Sentence: These apples are moved to the store at the corner.\n"
        "Head Entity: apples\n"
        "Tail Entity: store\n"
        "Candidate Relations: message-topic, entity-origin, entity-destination, content-container,"
        " cause-effect, component-whole, member-collection, instrument-agency, product-producer\n"
        "Relation Between the Head Entity and Tail Entity: entity-destination\n"
        "</Instance>\n"
        "<Instance>\n"
        "Given Sentence: These books talk about causal discovery.\n"
        "Head Entity: books\n"
        "Tail Entity: causal discovery\n"
        "Candidate Relations: message-topic, entity-origin, entity-destination, content-container,"
        " cause-effect, component-whole, member-collection, instrument-agency, product-producer\n"
        "Relation Between the Head Entity and Tail Entity: message-topic\n"
        "</Instance>\n"
        "<Instance>\n"
        "Given Sentence: These eggs are contained in the box.\n"
        "Head Entity: eggs\n"
        "Tail Entity: box\n"
        "Candidate Relations: message-topic, entity-origin, entity-destination, content-container,"
        " cause-effect, component-whole, member-collection, instrument-agency, product-producer\n"
        "Relation Between the Head Entity and Tail Entity: content-container\n"
        "</Instance>\n"
        "Hint: complete the remaining content and maintain consistency with the format of the above examples.\n"
        "<Instance>\n")

    ic5_incontext = (
        "<Instruction> Select the most suitable relation between the given head and tail entities in the"
        " given sentence. The relation type must be chosen from the candidate relations. </Instruction>\n"
        "<Instance>\n"
        "Given Sentence: These apples are from the store at the corner.\n"
        "Head Entity: apples\n"
        "Tail Entity: store\n"
        "Candidate Relations: message-topic, entity-origin, entity-destination, content-container,"
        " cause-effect, component-whole, member-collection, instrument-agency, product-producer\n"
        "Relation Between the Head Entity and Tail Entity: entity-origin\n"
        "</Instance>\n"
        "<Instance>\n"
        "Given Sentence: These apples are moved to the store at the corner.\n"
        "Head Entity: apples\n"
        "Tail Entity: store\n"
        "Candidate Relations: message-topic, entity-origin, entity-destination, content-container,"
        " cause-effect, component-whole, member-collection, instrument-agency, product-producer\n"
        "Relation Between the Head Entity and Tail Entity: entity-destination\n"
        "</Instance>\n"
        "<Instance>\n"
        "Given Sentence: These books talk about causal discovery.\n"
        "Head Entity: books\n"
        "Tail Entity: causal discovery\n"
        "Candidate Relations: message-topic, entity-origin, entity-destination, content-container,"
        " cause-effect, component-whole, member-collection, instrument-agency, product-producer\n"
        "Relation Between the Head Entity and Tail Entity: message-topic\n"
        "</Instance>\n"
        "<Instance>\n"
        "Given Sentence: These eggs are contained in the box.\n"
        "Head Entity: eggs\n"
        "Tail Entity: box\n"
        "Candidate Relations: message-topic, entity-origin, entity-destination, content-container,"
        " cause-effect, component-whole, member-collection, instrument-agency, product-producer\n"
        "Relation Between the Head Entity and Tail Entity: content-container\n"
        "</Instance>\n"
        "<Instance>\n"
        "Given Sentence: His mistake cause_concepts this accident.\n"
        "Head Entity: mistake\n"
        "Tail Entity: accident\n"
        "Candidate Relations: message-topic, entity-origin, entity-destination, content-container,"
        " cause-effect, component-whole, member-collection, instrument-agency, product-producer\n"
        "Relation Between the Head Entity and Tail Entity: cause-effect\n"
        "</Instance>\n"
        "Hint: complete the remaining content and maintain consistency with the format of the above examples.\n"
        "<Instance>\n")

    cot_incontext = (
        "<Instruction> Select the most suitable relation between the given head and tail entities in the"
        " given sentence. The relation type must be chosen from the candidate relations. </Instruction>\n"
        "<Instance>\n"
        "Given Sentence: These apples are from the store at the corner.\n"
        "Head Entity: apples\n"
        "Tail Entity: store\n"
        "Candidate Relations: message-topic, entity-origin, entity-destination, content-container,"
        " cause-effect, component-whole, member-collection, instrument-agency, product-producer\n"
        "Chain of Thought: Head entity apple is a fruit, which refers to the stuff entity in the"
        " context. Tail entity store is a place, which refers to the location entity in the context."
        " According to the context, apples are from the store, indicates that the store is the"
        " origin of apples, hence the relation between apples and store is entity-origin.\n"
        "Relation Between the Head Entity and Tail Entity: entity-origin\n"
        "</Instance>\n"
        "<Instance>\n"
        "Given Sentence: These apples are moved to the store at the corner.\n"
        "Head Entity: apples\n"
        "Tail Entity: store\n"
        "Candidate Relations: message-topic, entity-origin, entity-destination, content-container,"
        " cause-effect, component-whole, member-collection, instrument-agency, product-producer\n"
        "Chain of Thought: Head entity apple is a fruit, which refers to the stuff entity in the"
        " context. Tail entity store is a place, which refers to the location entity in the context."
        " According to the context, apples are moved to the store, indicates that the store is the"
        " destination of apples, hence the relation between apples and store is entity-destination.\n"
        "Relation Between the Head Entity and Tail Entity: entity-destination\n"
        "</Instance>\n"
        "Hint: complete the remaining content and maintain consistency with the format of the above examples.\n"
        "<Instance>\n")

    ic_prompt = (ic_incontext
                 + f"Given Sentence: {sentence}\n"
                 + f"Head Entity: {head_entity}\n"
                 + f"Tail Entity: {tail_entity}\n"
                 "Candidate Relations: message-topic, entity-origin, entity-destination, content-container,"
                 " cause-effect, component-whole, member-collection, instrument-agency, product-producer\n"
                 "Relation Between the Head Entity and Tail Entity:")

    ic1_prompt = (ic1_incontext
                  + f"Given Sentence: {sentence}\n"
                  + f"Head Entity: {head_entity}\n"
                  + f"Tail Entity: {tail_entity}\n"
                  "Candidate Relations: message-topic, entity-origin, entity-destination, content-container,"
                  " cause-effect, component-whole, member-collection, instrument-agency, product-producer\n"
                  "Relation Between the Head Entity and Tail Entity:")

    ic3_prompt = (ic3_incontext
                  + f"Given Sentence: {sentence}\n"
                  + f"Head Entity: {head_entity}\n"
                  + f"Tail Entity: {tail_entity}\n"
                  "Candidate Relations: message-topic, entity-origin, entity-destination, content-container,"
                  " cause-effect, component-whole, member-collection, instrument-agency, product-producer\n"
                  "Relation Between the Head Entity and Tail Entity:")

    ic4_prompt = (ic4_incontext
                  + f"Given Sentence: {sentence}\n"
                  + f"Head Entity: {head_entity}\n"
                  + f"Tail Entity: {tail_entity}\n"
                  "Candidate Relations: message-topic, entity-origin, entity-destination, content-container,"
                  " cause-effect, component-whole, member-collection, instrument-agency, product-producer\n"
                  "Relation Between the Head Entity and Tail Entity:")

    ic5_prompt = (ic5_incontext
                  + f"Given Sentence: {sentence}\n"
                  + f"Head Entity: {head_entity}\n"
                  + f"Tail Entity: {tail_entity}\n"
                  "Candidate Relations: message-topic, entity-origin, entity-destination, content-container,"
                  " cause-effect, component-whole, member-collection, instrument-agency, product-producer\n"
                  "Relation Between the Head Entity and Tail Entity:")

    cot_prompt = (cot_incontext
                  + f"Given Sentence: {sentence}\n"
                  + f"Head Entity: {head_entity}\n"
                  + f"Tail Entity: {tail_entity}\n"
                    "Candidate Relations: message-topic, entity-origin, entity-destination, content-container,"
                    " cause-effect, component-whole, member-collection, instrument-agency, product-producer\n"
                    "Chain of Thought:")

    if prompt_type == "ic":
        prompt = ic_prompt
    elif prompt_type == "ic1":
        prompt = ic1_prompt
    elif prompt_type == "ic3":
        prompt = ic3_prompt
    elif prompt_type == "ic4":
        prompt = ic4_prompt
    elif prompt_type == "ic5":
        prompt = ic5_prompt
    elif prompt_type == "cot":
        prompt = cot_prompt
    else:
        raise ValueError("Please select prompt from ic or cot.")

    return prompt


def entity_typing(sentence, entity, prompt_type="ic"):
    """The interface that constitutes prompt to guide llms to conduct entity typing."""
    ic_incontext = (
        "<Instruction> Based on the contextual semantics of the given sentence, select the most"
        " suitable entity type for the given entity from the candidate types. </Instruction>\n"
        "<Instance>\n"
        "Given Sentence: Michael Jordan is the best basketball player of all time.\n"
        "Given Entity: Michael Jordan\n"
        "Candidate Types: actor, author, athlete, director, politician, scholar,"
        " soldier, airplane, car, food, game, ship, software, weapon\n"
        "Entity Type of the Given Entity: athlete\n"
        "</Instance>\n"
        "<Instance>\n"
        "Given Sentence: Spielberg directed the famous movie Titanic 30 years ago.\n"
        "Given Entity: Spielberg\n"
        "Candidate Types: actor, author, athlete, director, politician, scholar,"
        " soldier, airplane, car, food, game, ship, software, weapon\n"
        "Entity Type of the Given Entity: director\n"
        "</Instance>\n"
        "Hint: Complete the remaining content and maintain consistency with the format of the above examples.\n"
        "<Instance>\n")

    ic_prompt = (ic_incontext
                 + f"Given Sentence: {sentence}\n"
                 + f"Given Entity: {entity}\n"
                 "Candidate Types: actor, author, athlete, director, politician, scholar,"
                 " soldier, airplane, car, food, game, ship, software, weapon\n"
                 "Entity Type of the Given Entity:")

    prompt = ic_prompt
    return prompt


def event_detection(sentence, trigger, prompt_type="ic"):
    """The interface that constitutes prompt to guide llms to conduct event detection."""
    ic_incontext = (
        "<Instruction> Based on the contextual semantics of the given sentence, select the most"
        " suitable event type for the given trigger word from the candidate event types. </Instruction>\n"
        "<Instance>\n"
        "Given Sentence: Peter leaved the world last year in the small village.\n"
        "Given Trigger Word: leaved\n"
        "Candidate Event Types: conflict:attack, movement:transport, life:die, contact:meet, personnel:end-position,"
        " transaction:transfer-money, personnel:elect, life:injure, transaction:transfer-ownership, contact:phone-write,"
        " personnel:start-position, justice:trial-hearing, justice:charge-indict, justice:sentence, justice:arrest-jail,"
        " conflict:demonstrate, life:marry, justice:convict, Justice:Sue\n"
        "Event Type of the Given Trigger Word: life:die\n"
        "</Instance>\n"
        "<Instance>\n"
        "Given Sentence: Maria left all her property to her youngest son.\n"
        "Given Trigger Word: left\n"
        "Candidate Event Types: conflict:attack, movement:transport, life:die, contact:meet, personnel:end-position,"
        " transaction:transfer-money, personnel:elect, life:injure, transaction:transfer-ownership, contact:phone-write,"
        " personnel:start-position, justice:trial-hearing, justice:charge-indict, justice:sentence, justice:arrest-jail,"
        " conflict:demonstrate, life:marry, justice:convict, Justice:Sue\n"
        "Event Type of the Given Trigger Word: transaction:transfer-money\n"
        "</Instance>\n"
        "Hint: Complete the remaining content and maintain consistency with the format of the above examples.\n"
        "<Instance>\n")

    ic_prompt = (ic_incontext
                 + f"Given Sentence: {sentence}\n"
                 + f"Given Trigger Word: {trigger}\n"
                 "Candidate Event Types: conflict:attack, movement:transport, life:die, contact:meet, personnel:end-position,"
                 " transaction:transfer-money, personnel:elect, life:injure, transaction:transfer-ownership, contact:phone-write,"
                 " personnel:start-position, justice:trial-hearing, justice:charge-indict, justice:sentence, justice:arrest-jail,"
                 " conflict:demonstrate, life:marry, justice:convict, Justice:Sue\n"
                 "Event Type of the Given Trigger Word:")

    prompt = ic_prompt
    return prompt
