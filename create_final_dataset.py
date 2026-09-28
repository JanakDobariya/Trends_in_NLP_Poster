import random
import re
import sys
from pathlib import Path

import pandas as pd
from convokit import Corpus

BASE_DIR = Path(__file__).resolve().parent

CORPUS_PATH = BASE_DIR / "corpus" / "reddit-corpus-small"
DATA_DIR = BASE_DIR / "data"

EXCEL_PATH = DATA_DIR / "final_dataset.xlsx"

SAMPLE_SIZE = 50
SAMPLING_SEED = 2026

BOT_INDICATORS = [
    "i am a bot",
    "i'm a bot",
    "this is a bot",
    "automoderator",
    "automated message",
    "automated response",
    "this message was generated automatically",
    "this is an automated message",
    "remindmebot",
    "reminderbot",
    "wiki bot",
    "moderation bot",
]

DELETED_MARKERS = {
    "[deleted]",
    "[removed]",
    "deleted",
    "removed",
}

EXPLICIT_SEXUAL_PATTERNS = [
    r"\bporn(?:ography|ographic)?\b",
    r"\bpornhub\b",
    r"\bsexually explicit\b",
    r"\bexplicit sexual\b",
    r"\bblowjob\b",
    r"\bblow job\b",
    r"\bhandjob\b",
    r"\bhand job\b",
    r"\bcumshot\b",
    r"\bsemen\b",
    r"\bejaculat(?:e|ed|ing|ion)\b",
    r"\borgasm(?:ed|ic)?\b",
    r"\bpenetrat(?:e|ed|ing|ion)\b",
    r"\bintercourse\b",
    r"\banal sex\b",
    r"\boral sex\b",
    r"\bgenitals?\b",
    r"\bgenitalia\b",
    r"\bpenis\b",
    r"\bpenises\b",
    r"\bvagina\b",
    r"\bvaginal\b",
    r"\bclitoris\b",
    r"\bclitoral\b",
    r"\btesticles?\b",
    r"\bbreasts?\b",
    r"\bnipples?\b",
    r"\bnaked\b",
    r"\bnudity\b",
    r"\bnude\b",
    r"\bnsfw\b",
]

EXPLICIT_SEXUAL_REGEXES = [
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in EXPLICIT_SEXUAL_PATTERNS
]

URL_REGEX = re.compile(
    r"(?i)\b(?:https?://|www\.)[^\s<>\"']+"
)

def get_text(utterance) -> str:
    # Return the utterance text as it is in the corpus

    text = getattr(utterance, "text", None)

    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    return text


def is_nonempty(text: str) -> bool:
    return bool(text and text.strip())


def contains_bot_indicator(text: str) -> bool:
    lowered = text.lower()
    return any(indicator in lowered for indicator in BOT_INDICATORS)


def is_deleted_or_removed(text: str) -> bool:
    stripped = text.strip().lower()

    if stripped in DELETED_MARKERS:
        return True

    if re.search(r"\[\s*(?:deleted|removed)\s*\]", stripped):
        return True

    return False

def contains_explicit_sexual_content(text: str) -> bool:
    return any(
        regex.search(text)
        for regex in EXPLICIT_SEXUAL_REGEXES
    )


def is_identifier_only(text: str) -> bool:
    # True if the whole utterance is just a user or subreddit name

    stripped = text.strip()
    if not stripped:
        return False

    # ignore surrounding spaces and trailing dots
    candidate = stripped.rstrip(".")

    identifier_pattern = re.compile(
        r"^(?:u/|/u/|@)[A-Za-z0-9_-]+$|^(?:r/|/r/)[A-Za-z0-9_+&-]+$",
        flags=re.IGNORECASE,
    )
    return bool(identifier_pattern.fullmatch(candidate))


def contains_url(text: str) -> bool:
    # True if the text contains an http(s)
    return bool(URL_REGEX.search(text))


def is_eligible_utterance(text: str) -> bool:
    if not is_nonempty(text):
        return False

    if contains_bot_indicator(text):
        return False

    if is_deleted_or_removed(text):
        return False

    if contains_explicit_sexual_content(text):
        return False

    if is_identifier_only(text):
        return False

    if contains_url(text):
        return False

    return True


def chain_is_eligible(
    grandparent_text: str,
    parent_text: str,
    target_text: str,
) -> bool:
    return (
        is_eligible_utterance(grandparent_text)
        and is_eligible_utterance(parent_text)
        and is_eligible_utterance(target_text)
    )

def load_corpus():
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            "ConvoKit Reddit Small corpus was not found.\n\n"
            "Expected location:\n"
            f"{CORPUS_PATH}\n\n"
            "Place the corpus inside the project's corpus/ directory "
            "before running this script."
        )

    if not CORPUS_PATH.is_dir():
        raise RuntimeError(
            f"Expected corpus directory, but found something else:\n"
            f"{CORPUS_PATH}"
        )

    return Corpus(filename=str(CORPUS_PATH))


# Finding grandparent/parent/target chains

def extract_eligible_chains(corpus):

    utterances = {}

    for utt in corpus.iter_utterances():
        utt_id = getattr(utt, "id", None)

        if utt_id is not None:
            utterances[str(utt_id)] = utt

    candidates_by_conversation = {}

    for target in utterances.values():

        target_id = str(
            getattr(target, "id", "")
        )

        parent_id = getattr(
            target,
            "reply_to",
            None,
        )

        if not parent_id:
            continue

        parent_id = str(parent_id)

        parent = utterances.get(parent_id)

        if parent is None:
            continue

        grandparent_id = getattr(
            parent,
            "reply_to",
            None,
        )

        if not grandparent_id:
            continue

        grandparent_id = str(grandparent_id)

        grandparent = utterances.get(
            grandparent_id
        )

        if grandparent is None:
            continue

        target_conversation_id = getattr(
            target,
            "conversation_id",
            None,
        )

        parent_conversation_id = getattr(
            parent,
            "conversation_id",
            None,
        )

        grandparent_conversation_id = getattr(
            grandparent,
            "conversation_id",
            None,
        )

        if target_conversation_id is None:
            continue

        conversation_id = str(
            target_conversation_id
        )


        if (
            parent_conversation_id is None
            or grandparent_conversation_id is None
            or str(parent_conversation_id) != conversation_id
            or str(grandparent_conversation_id) != conversation_id
        ):
            continue

        grandparent_text = get_text(grandparent)
        parent_text = get_text(parent)
        target_text = get_text(target)

        if not chain_is_eligible(
            grandparent_text,
            parent_text,
            target_text,
        ):
            continue

        candidates_by_conversation.setdefault(
            conversation_id,
            [],
        ).append(
            {
                "conversation_id": conversation_id,
                "grandparent_id": grandparent_id,
                "parent_id": parent_id,
                "target_id": target_id,
                "grandparent_text": grandparent_text,
                "parent_text": parent_text,
                "human_target": target_text,
            }
        )

    # one chain per conversation (smallest IDs first)
    selected_chain_by_conversation = {}

    for conversation_id, candidates in (
        candidates_by_conversation.items()
    ):
        candidates.sort(
            key=lambda x: (
                x["target_id"],
                x["parent_id"],
                x["grandparent_id"],
            )
        )

        selected_chain_by_conversation[
            conversation_id
        ] = candidates[0]

    return selected_chain_by_conversation


# Building the two contexts
def build_condition_a(parent_text: str) -> str:
    return "Speaker B: " + parent_text


def build_condition_b(
    grandparent_text: str,
    parent_text: str,
) -> str:
    return (
        "Speaker A: "
        + grandparent_text
        + "\nSpeaker B: "
        + parent_text
    )

# Checks on the sampled chains
def validate_selected_chains(selected):

    if len(selected) != SAMPLE_SIZE:
        raise RuntimeError(
            f"Expected exactly {SAMPLE_SIZE} selected conversations; "
            f"got {len(selected)}."
        )

    conversation_ids = [
        str(row["conversation_id"])
        for row in selected
    ]

    if len(set(conversation_ids)) != SAMPLE_SIZE:
        raise RuntimeError(
            "Selected Conversation IDs are not unique."
        )

    for row in selected:

        conversation_id = str(
            row["conversation_id"]
        )

        grandparent_text = row["grandparent_text"]
        parent_text = row["parent_text"]
        target_text = row["human_target"]

        # run the filters again on the raw texts
        if not chain_is_eligible(
            row["grandparent_text"],
            row["parent_text"],
            target_text,
        ):
            raise RuntimeError(
                f"Selected chain failed eligibility validation: "
                f"{conversation_id}"
            )

        condition_a = build_condition_a(
            parent_text
        )

        condition_b = build_condition_b(
            grandparent_text,
            parent_text,
        )

        expected_a = (
            "Speaker B: "
            + parent_text
        )

        expected_b = (
            "Speaker A: "
            + grandparent_text
            + "\nSpeaker B: "
            + parent_text
        )

        if condition_a != expected_a:
            raise RuntimeError(
                f"Condition A construction failed for "
                f"{conversation_id}."
            )

        if condition_b != expected_b:
            raise RuntimeError(
                f"Condition B construction failed for "
                f"{conversation_id}."
            )

        # the parent text must be the same in A and B
        parent_a = condition_a[
            len("Speaker B: "):
        ]

        separator = "\nSpeaker B: "

        separator_index = condition_b.find(
            separator
        )

        if (
            separator_index
            <= len("Speaker A: ")
        ):
            raise RuntimeError(
                f"Condition B separator invalid for "
                f"{conversation_id}."
            )

        extracted_grandparent = condition_b[
            len("Speaker A: "):
            separator_index
        ]

        extracted_parent = condition_b[
            separator_index
            + len(separator):
        ]

        if extracted_grandparent != grandparent_text:
            raise RuntimeError(
                f"Grandparent text mismatch for "
                f"{conversation_id}."
            )

        if extracted_parent != parent_text:
            raise RuntimeError(
                f"Parent text mismatch for "
                f"{conversation_id}."
            )

        if extracted_parent != parent_a:
            raise RuntimeError(
                f"Parent text differs between A and B for "
                f"{conversation_id}."
            )


        if target_text and target_text in condition_a:
            raise RuntimeError(
                f"Target leakage detected in Condition A for "
                f"{conversation_id}."
            )

        if target_text and target_text in condition_b:
            raise RuntimeError(
                f"Target leakage detected in Condition B for "
                f"{conversation_id}."
            )


def create_dataframe(selected):

    records = []

    sr_no = 1

    for row in selected:

        conversation_id = str(
            row["conversation_id"]
        )

        parent_text = row["parent_text"]
        grandparent_text = row["grandparent_text"]

        condition_a = build_condition_a(
            parent_text
        )

        condition_b = build_condition_b(
            grandparent_text,
            parent_text,
        )

        records.append(
            {
                "Sr.No.": sr_no,
                "Conversation ID": conversation_id,
                "Condition": "A",
                "Context": condition_a,
            }
        )

        sr_no += 1

        records.append(
            {
                "Sr.No.": sr_no,
                "Conversation ID": conversation_id,
                "Condition": "B",
                "Context": condition_b,
            }
        )

        sr_no += 1

    return pd.DataFrame(
        records,
        columns=[
            "Sr.No.",
            "Conversation ID",
            "Condition",
            "Context",
        ],
    )

# Checks on the finished table
def validate_dataframe(df, selected):

    expected_columns = [
        "Sr.No.",
        "Conversation ID",
        "Condition",
        "Context",
    ]

    if list(df.columns) != expected_columns:
        raise RuntimeError(
            "Excel dataset columns are incorrect.\n"
            f"Expected: {expected_columns}\n"
            f"Actual:   {list(df.columns)}"
        )

    if len(df) != 100:
        raise RuntimeError(
            f"Expected 100 rows; got {len(df)}."
        )

    if df["Conversation ID"].nunique() != 50:
        raise RuntimeError(
            "Expected exactly 50 unique Conversation IDs."
        )

    if (df["Condition"] == "A").sum() != 50:
        raise RuntimeError(
            "Expected exactly 50 Condition A rows."
        )

    if (df["Condition"] == "B").sum() != 50:
        raise RuntimeError(
            "Expected exactly 50 Condition B rows."
        )

    counts = df.groupby(
        "Conversation ID"
    ).size()

    if not (counts == 2).all():
        raise RuntimeError(
            "Every Conversation ID must occur exactly twice."
        )

    for conversation_id, group in df.groupby(
        "Conversation ID"
    ):

        conditions = list(
            group["Condition"]
        )

        if sorted(conditions) != ["A", "B"]:
            raise RuntimeError(
                f"Conversation {conversation_id} does not have "
                "exactly one A and one B row."
            )

        a_context = group.loc[
            group["Condition"] == "A",
            "Context",
        ].iloc[0]

        b_context = group.loc[
            group["Condition"] == "B",
            "Context",
        ].iloc[0]

        if not a_context.startswith(
            "Speaker B: "
        ):
            raise RuntimeError(
                f"Condition A has invalid structure for "
                f"{conversation_id}."
            )

        if not b_context.startswith(
            "Speaker A: "
        ):
            raise RuntimeError(
                f"Condition B has invalid structure for "
                f"{conversation_id}."
            )

        separator = "\nSpeaker B: "

        separator_index = b_context.find(
            separator
        )

        if (
            separator_index
            <= len("Speaker A: ")
        ):
            raise RuntimeError(
                f"Condition B lacks the required Speaker B "
                f"delimiter for {conversation_id}."
            )

        parent_a = a_context[
            len("Speaker B: "):
        ]

        parent_b = b_context[
            separator_index
            + len(separator):
        ]

        if parent_a != parent_b:
            raise RuntimeError(
                f"Parent text is not textually identical in "
                f"A/B for {conversation_id}."
            )

        source_row = next(
            item for item in selected
            if str(item["conversation_id"]) == str(conversation_id)
        )

        # none of the three original texts may contain a link
        if (
            contains_url(source_row["grandparent_text"])
            or contains_url(source_row["parent_text"])
            or contains_url(source_row["human_target"])
        ):
            raise RuntimeError(
                f"URL-containing conversation was selected for "
                f"{conversation_id}."
            )
        if is_identifier_only(source_row["grandparent_text"]):
            raise RuntimeError(
                f"Grandparent is identifier-only for {conversation_id}."
            )
        if is_identifier_only(source_row["parent_text"]):
            raise RuntimeError(
                f"Parent is identifier-only for {conversation_id}."
            )
        if is_identifier_only(source_row["human_target"]):
            raise RuntimeError(
                f"Target is identifier-only for {conversation_id}."
            )

    selected_ids = {
        str(row["conversation_id"])
        for row in selected
    }

    dataframe_ids = set(
        df["Conversation ID"].astype(str)
    )

    if dataframe_ids != selected_ids:
        raise RuntimeError(
            "Excel Conversation IDs do not exactly match "
            "the selected IDs."
        )

    if "human_target" in df.columns:
        raise RuntimeError(
            "Human target must not be an Excel column."
        )

    if len(df.columns) != 4:
        raise RuntimeError(
            "Excel workbook must contain exactly four columns."
        )

def main() -> int:

    try:
        print("=" * 70)
        print("FINAL 50-CONVERSATION DATASET CREATION")
        print("=" * 70)
        print()

        print(f"Project directory: {BASE_DIR}")
        print(f"Corpus: {CORPUS_PATH}")
        print(f"Excel output: {EXCEL_PATH}")
        print()

        print("Loading ConvoKit Reddit Small corpus...")

        corpus = load_corpus()

        print("Extracting eligible "
            "Grandparent - Parent - Target chains...")

        eligible_by_conversation = (extract_eligible_chains(corpus))

        eligible_pool_size = len(eligible_by_conversation)

        print("Eligible conversation pool before sampling:"f"{eligible_pool_size}")

        if eligible_pool_size < SAMPLE_SIZE:
            raise RuntimeError(f"Only {eligible_pool_size} eligible conversations "
                f"are available; exactly {SAMPLE_SIZE} are required.")

        # draw 50 conversations with a fixed seed
        conversation_ids = sorted(eligible_by_conversation.keys())

        rng = random.Random(SAMPLING_SEED)

        selected_ids = rng.sample(conversation_ids, SAMPLE_SIZE)

        # sort so the rows always come out in the same order
        selected_ids.sort()

        selected = [
            eligible_by_conversation[
                conversation_id
            ]
            for conversation_id in selected_ids
        ]

        # check everything before writing anything
        print(
            "Validating all selected conversations "
            "before dataset creation..."
        )

        validate_selected_chains(selected)

        # build the 100 row table
        df = create_dataframe(selected)

        validate_dataframe(df,selected)

        # data is only created once the checks have passed
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        df.to_excel(EXCEL_PATH,
            index=False,
            engine="openpyxl")

        if not EXCEL_PATH.exists():
            raise RuntimeError("Excel output file was not created.")

        print()
        print("=" * 70)
        print("FINAL DATASET REPORT")
        print("=" * 70)
        print("Eligible conversation pool size before sampling:"f"{eligible_pool_size}")
        print("Number of selected conversations = 50")
        print("Condition A rows = 50")
        print("Condition B rows = 50")
        print("Total rows = 100")
        print("Random seed = 2026")
        print(f"Excel output path = {EXCEL_PATH}")
        print("Original Reddit corpus modified = NO")
        print("Validation status = PASSED")
        print("=" * 70)

        return 0

    except Exception as exc:

        print()
        print("=" * 70)
        print("DATASET CREATION FAILED")
        print("=" * 70)
        print(f"ERROR: {exc}")
        print("Validation status = FAILED")
        print("=" * 70)

        return 1


if __name__ == "__main__":
    sys.exit(main())