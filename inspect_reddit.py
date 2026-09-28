from pathlib import Path
from convokit import Corpus

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "corpus" / "reddit-corpus-small"


def main():
    print("Loading Reddit Small...")
    corpus = Corpus(filename=DATASET_PATH)

    print(f"Conversations: {len(corpus.get_conversation_ids())}")
    print(f"Utterances: {len(corpus.utterances)}")
    print(f"Speakers: {len(corpus.speakers)}")
    print()

    valid_parent = 0
    valid_grandparent = 0
    examples = []

    for utterance in corpus.iter_utterances():
        parent_id = utterance.reply_to

        if not parent_id:
            continue

        parent = corpus.get_utterance(parent_id)

        if parent is None or not parent.text:
            continue

        valid_parent += 1

        grandparent_id = parent.reply_to

        if not grandparent_id:
            continue

        grandparent = corpus.get_utterance(grandparent_id)

        if grandparent is None or not grandparent.text:
            continue

        valid_grandparent += 1

        if len(examples) < 5:
            examples.append(
                {
                    "grandparent": grandparent.text,
                    "parent": parent.text,
                    "target": utterance.text,
                }
            )

    print("Valid target -> parent links:", valid_parent)
    print("Valid Grandparent -> Parent -> Target chains:", valid_grandparent)
    print()

    print("=== SAMPLE CHAINS ===")

    for i, example in enumerate(examples, start=1):
        print(f"\n--- Example {i} ---")

        print("GRANDPARENT:")
        print(example["grandparent"])

        print("\nPARENT:")
        print(example["parent"])

        print("\nTARGET:")
        print(example["target"])


if __name__ == "__main__":
    main()