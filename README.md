# Does More Conversational Context Improve LLM-Generated Social Media Replies?

A controlled human evaluation of parent-only vs. grandparent + parent context.

Prathamesh Anil Choudhary and Janak Dobariya, Trier University, course "Trends in NLP".

## What the project does

We test whether an LLM writes better Reddit replies when it also sees the message before the one it is replying to. Every conversation is a chain of three Reddit messages: grandparent, parent, and the real reply (target). The model writes a reply to the parent under two conditions:

| Condition | Context given to the model |
|---|---|
| A: Parent only | `Speaker B: <parent>` |
| B: Grandparent + Parent | `Speaker A: <grandparent>` and, on the next line, `Speaker B: <parent>` |

The real reply (target) is never shown to the model. We used 50 conversations, so there are 100 generated replies (one per condition per conversation). Humans rated each reply from 1 to 5 on Coherence, Relevance, Plausibility, Conversational Appropriateness and Contextual Fit. The Overall Score is the mean of these five and is treated as a secondary outcome.

The A and B replies of the same conversation form a pair, so the analysis uses a two-sided Wilcoxon signed-rank test for each dimension. Holm-Bonferroni correction is applied across the five primary dimensions (significance: adjusted p < 0.05). The Overall Score is tested the same way but is not corrected.

## Folder contents

```
NLP_Poster_Final/
├── README.md
├── requirements.txt              pinned Python packages
├── inspect_reddit.py             optional: quick look at the corpus
├── create_final_dataset.py       step 1: builds the 50-conversation dataset
├── run_qwen.py                   step 2: generates the 100 replies with Qwen via Ollama
├── statistical_analysis.py       step 4: statistics, tables and figures
├── corpus/reddit-corpus-small/   Reddit corpus (not included, see "Getting the corpus")
├── data/final_dataset.xlsx       output of step 1
├── generation/
│   ├── generated_replies.xlsx    output of step 2 (the replies that were rated)
│   └── generation_metadata.txt   model, system prompt and settings used in step 2
├── evaluation/
│   └── human_rated_results.xlsx  step 3: the human ratings
└── analysis/                     output of step 4
    ├── statistical_results.csv
    ├── paired_differences.csv
    ├── dimension_means.png
    ├── paired_differences.png
    └── overall_score.png
```

All scripts find their files relative to their own location, so they can be run from any working directory as long as the folder layout above is kept.

## Requirements

- Python 3.12. The project was run with Python 3.12.10 on Windows. Some pinned packages may not install on older Python versions.
- The packages in `requirements.txt`. The list is long because ConvoKit 4.1.2 itself depends on spaCy, sentence-transformers (which brings in PyTorch), datasets and several others. Expect a large download.
- For step 2 only: [Ollama](https://ollama.com) with the model `qwen3:4b-instruct-2507-q4_K_M` (about 2.5 GB). Ollama version used: 0.34.4.

## Setup

Windows (PowerShell):

```
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS / Linux:

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Getting the corpus

The Reddit data is not included in this submission. We used the ConvoKit "Reddit Corpus (small)": 100 comment threads from each of 100 active subreddits, September 2018 (8,286 conversations, 297,132 utterances). Documentation: https://convokit.cornell.edu/documentation/reddit-small.html

Download it with ConvoKit (installed by `requirements.txt`):

```
python -c "from convokit import download; print(download('reddit-corpus-small'))"
```

This prints the folder the corpus was saved to. Copy that folder to `corpus/reddit-corpus-small/`, running the commands below from the project folder (`Trends_in_NLP_Poster`).

macOS / Linux: ConvoKit saves the corpus to `~/.convokit/saved-corpora/reddit-corpus-small`. Finder does not show this folder because `.convokit` starts with a dot, which makes it hidden on macOS (press Cmd + Shift + . in Finder to show hidden folders). The folder exists and can be copied from Terminal:

```
mkdir -p corpus
cp -r ~/.convokit/saved-corpora/reddit-corpus-small corpus/
```

Windows (PowerShell), using the path printed by the download command:

```
mkdir corpus
Copy-Item -Recurse "<printed path>" corpus\reddit-corpus-small
```

Check the result:

```
ls corpus/reddit-corpus-small
```

It should list `conversations.json`, `corpus.json`, `index.json`, `users.json` and `utterances.json`.

Our copy was downloaded in September 2026. If ConvoKit ever publishes a changed version of this corpus, step 1 could select different conversations. The dataset we actually used is included as `data/final_dataset.xlsx`, so steps 2 to 4 do not depend on the download.

## Running the pipeline

Run the commands from the project folder with the virtual environment activated.

### Quickest check: step 4 only

If you only want to check the reported numbers, run the analysis on the included ratings. This needs neither the corpus nor Ollama, and it is fully deterministic:

```
python statistical_analysis.py
```

It checks the rating file (100 rows, 50 conversations each with one A and one B row, ratings between 1 and 5), runs the tests and writes the five files in `analysis/`. Near the end it prints `Analysis status = PASSED`. The numbers in `analysis/statistical_results.csv` should match the table under "Expected results".

### Optional: look at the corpus

```
python inspect_reddit.py
```

Prints the corpus size, the number of grandparent → parent → target chains and five example chains. Nothing is written.

### Step 1: build the dataset

```
python create_final_dataset.py
```

Loads the corpus and keeps grandparent → parent → target chains where all three messages are in the same conversation. A chain is dropped if any of the three messages is empty, deleted or removed, looks like a bot message, contains a link, contains explicit sexual terms (keyword list in the script), or is only a user or subreddit name. One chain is kept per conversation (the one with the smallest IDs). From these, 50 conversations are drawn with `random.Random(2026)`. The script then writes `data/final_dataset.xlsx` with 100 rows (columns `Sr.No.`, `Conversation ID`, `Condition`, `Context`) and prints `Validation status = PASSED` at the end.

Note: this overwrites `data/final_dataset.xlsx`. Keep a copy if you want to compare the new file with ours.

### Step 2: generate the replies

Start Ollama and download the model once:

```
ollama pull qwen3:4b-instruct-2507-q4_K_M
```

Ollama must be running at `http://localhost:11434` (start it with `ollama serve` if it is not already running). Then:

```
python run_qwen.py
```

The script checks the dataset and that the model is installed, then sends each of the 100 contexts to the model with a fixed system prompt and these settings: temperature 0.7, top-p 0.8, seed 2026, at most 128 new tokens, thinking disabled. It writes `generation/generated_replies.xlsx` (the dataset plus a `Generated Reply` column) and `generation/generation_metadata.txt`, and prints `Generation status = PASSED` at the end. It stops at the first failed request without writing the replies file.

Note: this overwrites the replies that were rated. The seed is fixed, but LLM output can still differ between Ollama versions and hardware, so a new run may not give exactly the same text. The replies that were actually evaluated are the ones in the included `generation/generated_replies.xlsx`.

### Step 3: human rating

This step was done by hand. The 100 generated replies were copied into `evaluation/human_rated_results.xlsx` and each reply got a rating from 1 to 5 for each of the five dimensions (columns `Sr.No.`, `Conversation ID`, `Condition`, `Context`, `Generated Reply`, `Coherence`, `Relevance`, `Plausibility`, `Conversational Appropriateness`, `Contextual Fit`). The Generated replies were evaluated by Ms. Anushka Choudhary (Master's student, Saarland University).

### Step 4: statistical analysis

```
python statistical_analysis.py
```

See "Quickest check" above. For every measure the script reports the A and B means, the mean paired difference B − A with a 95% t-based confidence interval, the Wilcoxon statistic and p-value (`scipy.stats.wilcoxon`, two-sided, `zero_method="wilcox"`, so pairs with identical ratings are left out), the Holm-adjusted p-value for the five primary dimensions, and Cohen's dz.

## Expected results

N = 50 pairs. Values from `analysis/statistical_results.csv`, rounded.

| Measure | A mean | B mean | B − A | 95% CI | Wilcoxon p | Holm-adjusted p |
|---|---|---|---|---|---|---|
| Coherence | 4.24 | 4.44 | +0.20 | −0.098 to 0.498 | 0.269 | 0.269 |
| Relevance | 4.00 | 4.56 | +0.56 | 0.278 to 0.842 | 0.00053 | 0.0021 |
| Plausibility | 4.04 | 4.50 | +0.46 | 0.177 to 0.743 | 0.0028 | 0.0085 |
| Conversational Appropriateness | 3.96 | 4.40 | +0.44 | 0.100 to 0.780 | 0.026 | 0.052 |
| Contextual Fit | 3.48 | 4.42 | +0.94 | 0.603 to 1.277 | 0.000011 | 0.000054 |
| Overall Score (secondary) | 3.944 | 4.464 | +0.52 | 0.234 to 0.806 | 0.0012 | not corrected |

Condition B has the higher mean on all five dimensions. After Holm correction the difference is significant for Relevance, Plausibility and Contextual Fit, but not for Coherence or Conversational Appropriateness, so the hypothesis is only partially supported.

## Fixed settings

| Setting | Value | Where |
|---|---|---|
| Sample size | 50 conversations | `create_final_dataset.py` |
| Sampling seed | 2026 | `create_final_dataset.py` |
| Model | `qwen3:4b-instruct-2507-q4_K_M` (Ollama) | `run_qwen.py` |
| Temperature / top-p | 0.7 / 0.8 | `run_qwen.py` |
| Generation seed | 2026 | `run_qwen.py` |
| Max new tokens | 128 | `run_qwen.py` |
| Significance level | 0.05 (Holm-adjusted) | `statistical_analysis.py` |

## Data and citation

The Reddit texts come from ConvoKit, which does not state a separate license for this corpus, so we do not redistribute it. Please download it from the source as described above. The files in `data/`, `generation/` and `evaluation/` quote the Reddit messages of the 50 sampled conversations and are included only so the results can be checked.

- Chang, J. P., Chiam, C., Fu, L., Wang, A., Zhang, J., & Danescu-Niculescu-Mizil, C. (2020). ConvoKit: A toolkit for the analysis of conversations. In Proceedings of the 21th Annual Meeting of the Special Interest Group on Discourse and Dialogue (pp. 57–60). https://doi.org/10.18653/v1/2020.sigdial-1.8
- Qwen Team. (2025). Qwen3 technical report (arXiv:2505.09388). https://doi.org/10.48550/arXiv.2505.09388

The full method and reference list are in the appendix that accompanies the poster.
