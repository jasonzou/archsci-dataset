
import re

def preprocess_footnote(text:str) -> str:
    # Regex pattern with DOTALL flag to include newlines 
    footnote_pattern = re.compile(r'^\d+\s.*\.', re.DOTALL)
    match = footnote_pattern.search(text)
    if match:
        #print(match.group())
        return ""
    else:
        #print("no match")
        return text

def preprocess_footnote_intext_replace(text:str, matched_list:list) -> str:
    # Regex pattern to match footnote numbers

    footnote_pattern = re.compile( 
        r''' 
        (?<![dD][oO][iI])# Negative lookbehind for "doi" (case-insensitive) 
        (?<!10\.)        # Negative lookbehind for "10." (common DOI prefix)
        (?<=             # Positive lookbehind to ensure the number is preceded by: 
          [a-zA-Z,.;’”]  # A letter or common punctuation (e.g., comma, period, quote)
        ) 
        ({})             # Match specific numbers (formatted into the pattern)
        (?=              # Positive lookahead to ensure the number is followed by: 
          \W|$           # A non-word character (e.g., space, semicolon) or end of string
        ) 
        '''.format('|'.join(matched_list)),
        flags=re.VERBOSE
    ) 
    
    # Remove the footnote numbers from the text
    clean_text = footnote_pattern.sub('', text)
    return clean_text

def preprocess_footnote_intext(text:str) -> list:
    # Regex pattern to match footnote numbers

    footnote_pattern = re.compile( 
        r''' 
        (?<![dD][oO][iI])# Negative lookbehind for "doi" (case-insensitive) 
        (?<!10\.)        # Negative lookbehind for "10." (common DOI prefix)
        (?<=             # Positive lookbehind to ensure the number is preceded by: 
          [a-zA-Z,.;’”]  # A letter or common punctuation (e.g., comma, period, quote)
        ) 
        (\d+)            # Match the footnote number (one or more digits) 
        (?=              # Positive lookahead to ensure the number is followed by: 
          \W|$           # A non-word character (e.g., space, semicolon) or end of string
        ) 
        ''', 
        flags=re.VERBOSE
    ) 
    
    # Extract all footnote numbers 
    matches = footnote_pattern.findall(text)
    return matches 

with open("./data/journals/00001.md") as fp:
    content = fp.readlines()

text = ""
footnotes_identified = []
last_footnotes = ''
for line in content:
    tmp_str = preprocess_footnote(line)
    clean_text = ''
    if tmp_str != "": 
        matches_footnote = preprocess_footnote_intext(line)
        print("==============")
        print(matches_footnote)
        print("===== 1 =========")
        print(footnotes_identified)
        if len(footnotes_identified) > 0: 
            last_footnotes = footnotes_identified[-1]
        print("===== 2 =========")
        print(last_footnotes)
        if len(matches_footnote) > 0:
            if matches_footnote[0] > last_footnotes: 
                clean_text = preprocess_footnote_intext_replace(line, matches_footnote)
                footnotes_identified.extend(matches_footnote)
            
        if len(clean_text) > 0: 
            text += clean_text
        else:
            text += line

print("=========================")
print(text)
print("----------- 1. preprocess footnote ============")
# 1. preprocess_footnote
print(preprocess_footnote(text))

from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator="",
    chunk_size=384,
    chunk_overlap=56,
    length_function=len,
    is_separator_regex=False,
)
chunk_doc1 = text_splitter.split_text(text)
for index, elem in enumerate(chunk_doc1):
    print(index, elem)

from langchain_text_splitters import RecursiveCharacterTextSplitter

print(text)
text_splitter2 = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=1,
    length_function=len,
    is_separator_regex=False,
    separators=["\n\n", "\n", " ", ""],
)

chunk_doc2 = text_splitter2.split_text(text)

for i , _ in enumerate(chunk_doc2):
    print(f"chunk #{i} , size:{len(chunk_doc2[i])}")
    print(chunk_doc2[i])
    print("-----")
print("sssssemantic lllllll")
from sentence_transformers import SentenceTransformer  # noqa: E402
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import pandas as pd


embedding_model = "Lajavaness/bilingual-embedding-large"

my_embedding_model = SentenceTransformer(embedding_model, trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(embedding_model, trust_remote_code=True)


def plot_chunk(chunk_doc, embedding_name):
    tokenizer = AutoTokenizer.from_pretrained(embedding_model, trust_remote_code=True)
    length = [len(tokenizer.encode(doc.page_content)) for doc in chunk_doc2]
    fig = pd.Series(length).hist()
    plt.show()


content = text

split_char = [".", "?", "!"]

import re  # noqa: E402

sentences_list = re.split(r"(?<=[.?!])\s+", content)
print(sentences_list)
print(len(sentences_list))

sentences = [{"sentence": x, "index": i} for i, x in enumerate(sentences_list)]
print("Sssssentences")
print(sentences)
exit


def combine_sentences(sentences, buffer_size=1):
    combine_sentences = [
        " ".join(
            sentences[j]["sentence"]
            for j in range(
                max(i - buffer_size, 0), min(i + buffer_size + 1, len(sentences))
            )
        )
        for i in range(len(sentences))
    ]

    for i, combine_sentence in enumerate(combine_sentences):
        sentences[i]["combined_sentence"] = combine_sentence

    return sentences


sentences = combine_sentences(sentences=sentences)

print(sentences)
print(len(sentences))
embeddings = my_embedding_model.encode(
    sentences=[x["combined_sentence"] for x in sentences]
)
for i, elem in enumerate(sentences):
    print(elem)
    elem["combined_sentence_embedding"] = embeddings[i]

print("combinedddddd")
for elem in sentences:
    print(elem)
print("combinedddddd___________")

import numpy as np


def cosine_similarity(vec1:np.ndarray, vec2:np.ndarray) -> float:
    # Calculate the similarity between two vectors
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def calculate_cosine_distances(sentences):
    distances = []
    for i in range(len(sentences) - 1):
        embedding_current = sentences[i]["combined_sentence_embedding"]
        embedding_next = sentences[i + 1]["combined_sentence_embedding"]

        # Calculate cosine similarity
        similarity = cosine_similarity(embedding_current, embedding_next)

        distance = 1 - similarity
        distances.append(distance)

        # Store into the sentences
        sentences[i]["distance_to_next"] = distance
    return distances, sentences


distances, sentences = calculate_cosine_distances(sentences=sentences)

print(sentences[-2]["distance_to_next"])

y_upper_bound = 0.25
plt.ylim(0, y_upper_bound)
plt.xlim(0, len(distances))

# percentile()
brekpoint_percentile_threshold = 95
brekpoint_distance_threshold = np.percentile(distances, brekpoint_percentile_threshold)

plt.axhline(y=brekpoint_distance_threshold, color="r", linestyle="-")
num_distances_above_threshold = len(
    [x for x in distances if x > brekpoint_distance_threshold]
)
plt.text(
    x=(len(distances) * 0.01),
    y=y_upper_bound / 50,
    s=f"{num_distances_above_threshold +1} chunks",
)

indices_above_thresh = [
    i for i, x in enumerate(distances) if x > brekpoint_distance_threshold
]

colors = ["b", "g", "r", "c", "m", "y", "k"]

for i, breakpoint_index in enumerate(indices_above_thresh):
    start_index = 0 if i == 0 else indices_above_thresh[i - 1]
    end_index = (
        breakpoint_index if i <= len(indices_above_thresh) - 1 else len(distances)
    )

    plt.axvspan(start_index, end_index, facecolor=colors[i % len(colors)], alpha=0.25)
    plt.text(
        x=np.average([start_index, end_index]),
        y=brekpoint_distance_threshold + (y_upper_bound) / 20,
        s=f"Chunk #{i}",
        horizontalalignment="center",
        rotation="vertical",
    )

    if indices_above_thresh:
        last_breakpoint = indices_above_thresh[-1]
        if last_breakpoint < len(distances):
            plt.axvspan(
                last_breakpoint,
                len(distances),
                facecolor=colors[len(indices_above_thresh) % len(colors)],
                alpha=0.25,
            )
            plt.text(
                x=np.average([last_breakpoint, len(distances)]),
                y=brekpoint_distance_threshold + (y_upper_bound) / 20,
                s=f"Chunk #{i+1}",
                horizontalalignment="center",
                rotation="vertical",
            )

plt.plot(distances)
plt.show()

start_index = 0
chunks = []

for index in indices_above_thresh:
    end_index = index
    group = sentences[start_index : end_index + 1]
    combined_text = " ".join([d["sentence"] for d in group])
    chunks.append(combined_text)

    start_index = index + 1


if start_index < len(sentences):
    combine_sentences = " ".join([d["sentence"] for d in sentences[start_index:]])
    chunks.append(combined_text)

for i, chunk in enumerate(chunks[:1]):
    buffer = 200
    print(f"Chunk #{i}")
    print(chunk[:buffer].strip())
    print("...")
    print(chunk[-buffer:].strip())
    print("\n")


for i, chunk in enumerate(chunks):
    print(i, chunk)

print(chunks)
