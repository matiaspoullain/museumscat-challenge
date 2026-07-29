from src.classify_card_relevance import evaluate, is_relevant_card


def test_card_with_a_date_word_is_relevant():
    assert is_relevant_card(["27.5.2022", "Coll.", "Johansen"]) is True


def test_card_with_only_confound_words_is_not_relevant():
    assert is_relevant_card(["Coll.", "J.", "P.", "Dassco"]) is False


def test_empty_word_list_is_not_relevant():
    assert is_relevant_card([]) is False


def test_evaluate_computes_precision_recall_accuracy():
    matches = {
        "a.jpeg": {"status": "matched", "selections": []},
        "b.jpeg": {"status": "no_relevant_text", "selections": []},
        "c.jpeg": {"status": "skipped", "selections": []},
    }
    predictions_by_card = {
        "a.jpeg": ["27.5.2022"],   # correctly predicted relevant
        "b.jpeg": ["Coll.", "X"],  # correctly predicted not relevant
    }
    precision, recall, accuracy = evaluate(predictions_by_card, matches)
    assert precision == 1.0
    assert recall == 1.0
    assert accuracy == 1.0
