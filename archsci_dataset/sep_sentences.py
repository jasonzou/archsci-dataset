from sentence_transformers import SentenceTransformer  # noqa: E402
from transformers import pipeline
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import re


def plot_chunk(chunk_doc, embedding_name):
    tokenizer = AutoTokenizer.from_pretrained(embedding_name)
    length = [len(tokenizer.encode(doc.page_content)) for doc in chunk_doc]
    fig = pd.Series(length).hist()
    plt.show()


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


def cosine_similarity(vec1, vec2):
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

def generate_sentences_list_v1(filename: str, output_filename: str):
    embedding_model = "/home/jason/workspace/models/bge-large-en-v1.5"
    embedding_model = "NovaSearch/stella_en_1.5B_v5"
    

    my_embedding_model = SentenceTransformer(embedding_model)
    max_seq_length = my_embedding_model.max_seq_length
    print("embedding model -- max seq length --> {}".format(max_seq_length))

    tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    # print(chunk_doc[1].page_content)
    # print(len(chunk_doc[1].page_content))
    # print(len(tokenizer.encode(chunk_doc[1].page_content)))

    # plot_chunk(chunk_doc=chunk_doc, embedding_name=embedding_model)

    with open(filename) as file:
        content = file.read()

    split_char = [".", "?", "!"]

    sentences_list = re.split(r"(?<=[.?!])\s+", content)

    print(sentences_list)
    for sentence in sentences_list:
        print("----------------------")
        print(sentence)
        print("++++++++++----------------------")
    print(len(sentences_list))

    sentences_list1 = [re.split(r"\n\n", x) for x in sentences_list]
    print(sentences_list1)
    sentences = {}
    i = 0
    for sentences_list in sentences_list1:
        temp = [re.sub(r"\s\s", " ", re.sub(r"-\n", "", x)) for x in sentences_list]
        for elem in temp:
            if elem.strip() != "":
                sentences[i] = elem
                i = i + 1

    print(sentences)
    merge_index = []
    merged_sentences = {}
    i = 0
    for index, value in sentences.items():
        print("---")
        print(value)
        if value[0].islower():
            print(index)
            merge_index.append(index)
            merged_sentences[i - 1] = merged_sentences[i - 1] + sentences[index]
        else:
            merged_sentences[i] = value
            i = i + 1

    for index, value in merged_sentences.items():
        print(index)
        print("=================---")
        print(value)
        merged_sentences[index] = re.sub(r"\n", "", value)
    print(merged_sentences)

    ## merge neighbouring sentences
    sentences = []
    for index, value in merged_sentences.items():
        temp = {}
        temp["index"] = index
        temp["sentence"] = value
        if temp["sentence"].lower().startswith("keywords:"):
            temp["sentence"] = merged_sentences[0] + " has " + value
        if temp["sentence"].lower().startswith("related entries:"):
            temp["sentence"] = merged_sentences[0] + " has " + value
        #temp["combined_sentence_embedding"] = my_embedding_model.encode(value)
        sentences.append(temp)
    #distances, sentences = calculate_cosine_distances(sentences=sentences)
    print(sentences)
    
    with open(output_filename, "a") as fp:
        for sentence in sentences:
            # fp.writelines(spell_check(sentence["sentence"]) + "\n")
            fp.writelines(sentence["sentence"] + "\n")
    
def generate_sentences_list():
    embedding_model = "/home/jason/workspace/models/bge-large-en-v1.5"

    my_embedding_model = SentenceTransformer(embedding_model)
    max_seq_length = my_embedding_model.max_seq_length
    print("embedding model -- max seq length --> {}".format(max_seq_length))

    tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    # print(chunk_doc[1].page_content)
    # print(len(chunk_doc[1].page_content))
    # print(len(tokenizer.encode(chunk_doc[1].page_content)))

    # plot_chunk(chunk_doc=chunk_doc, embedding_name=embedding_model)

    with open("004.txt") as file:
        content = file.read()

    split_char = [".", "?", "!"]

    sentences_list = re.split(r"(?<=[.?!])\s+", content)

    print(sentences_list)
    for sentence in sentences_list:
        print("----------------------")
        print(sentence)
        print("++++++++++----------------------")
    print(len(sentences_list))

    sentences_list1 = [re.split(r"\n\n", x) for x in sentences_list]
    print(sentences_list1)
    sentences = {}
    i = 0
    for sentences_list in sentences_list1:
        temp = [re.sub(r"\s\s", " ", re.sub(r"-\n", "", x)) for x in sentences_list]
        for elem in temp:
            if elem.strip() != "":
                sentences[i] = elem
                i = i + 1

    print(sentences)
    merge_index = []
    merged_sentences = {}
    i = 0
    for index, value in sentences.items():
        print("---")
        print(value)
        if value[0].islower():
            print(index)
            merge_index.append(index)
            merged_sentences[i - 1] = merged_sentences[i - 1] + sentences[index]
        else:
            merged_sentences[i] = value
            i = i + 1

    for index, value in merged_sentences.items():
        print(index)
        print("=================---")
        print(value)
        merged_sentences[index] = re.sub(r"\n", "", value)
    print(merged_sentences)

    ## merge neighbouring sentences
    sentences = []
    for index, value in merged_sentences.items():
        temp = {}
        temp[index] = index
        temp["sentence"] = value
        if temp["sentence"].lower().startswith("keywords:"):
            temp["sentence"] = merged_sentences[0] + " has " + value
        if temp["sentence"].lower().startswith("related entries:"):
            temp["sentence"] = merged_sentences[0] + " has " + value
        temp["combined_sentence_embedding"] = my_embedding_model.encode(value)
        sentences.append(temp)
    distances, sentences = calculate_cosine_distances(sentences=sentences)
    print(sentences)

    # # percentile()
    # brekpoint_percentile_threshold = 20
    # brekpoint_distance_threshold = np.percentile(
    #     distances, brekpoint_percentile_threshold
    # )
    # indices_above_thresh = [
    #     i for i, x in enumerate(distances) if x > brekpoint_distance_threshold
    # ]

    # start_index = 0
    # chunks = []

    # for index in indices_above_thresh:
    #     end_index = index
    #     group = sentences[start_index : end_index + 1]
    #     combined_text = " ".join([d["sentence"] for d in group])
    #     chunks.append(combined_text)

    #     start_index = index + 1

    # if start_index < len(sentences):
    #     combined_text = " ".join([d["sentence"] for d in sentences[start_index:]])
    #     chunks.append(combined_text)

    # for i, chunk in enumerate(chunks[:1]):
    #     buffer = 200
    #     print(f"Chunk #{i}")
    #     print(chunk[:buffer].strip())
    #     print("...")
    #     print(chunk[-buffer:].strip())
    #     print("\n")

    # for i, chunk in enumerate(chunks):
    #     print(i, chunk)

    # print(chunks)
    # print(len(chunks))
    # with open("004-txt", "a") as fp:
    #     for chunk in chunks:
    #         fp.writelines(spell_check(chunk) + "\n")

    y_upper_bound = 0.9
    plt.ylim(0, y_upper_bound)
    plt.xlim(0, len(distances))

    # percentile()
    brekpoint_percentile_threshold = 15
    brekpoint_distance_threshold = np.percentile(
        distances, brekpoint_percentile_threshold
    )

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

        plt.axvspan(
            start_index, end_index, facecolor=colors[i % len(colors)], alpha=0.25
        )
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
        combined_text = " ".join([d["sentence"] for d in sentences[start_index:]])
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
    print(len(chunks))
    with open("004-txt", "a") as fp:
        for chunk in chunks:
            fp.writelines(spell_check(chunk) + "\n")


def main():
    embedding_model = "/home/jason/workspace/models/bge-large-en-v1.5"

    print(SentenceTransformer(embedding_model).max_seq_length)
    my_embedding_model = SentenceTransformer(embedding_model)

    tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    # print(chunk_doc[1].page_content)
    # print(len(chunk_doc[1].page_content))
    # print(len(tokenizer.encode(chunk_doc[1].page_content)))

    # plot_chunk(chunk_doc=chunk_doc, embedding_name=embedding_model)

    with open("004.txt") as file:
        content = file.read()

    split_char = [".", "?", "!"]

    sentences_list = re.split(r"(?<=[.?!])\s+", content)
    print(sentences_list)
    print(len(sentences_list))
    print("=======================================")

    sentences = [
        {"sentence": re.sub(r"-\n", "", x), "index": i}
        for i, x in enumerate(sentences_list)
    ]

    print(sentences)

    sentences = combine_sentences(sentences=sentences)

    print(sentences)
    print(len(sentences))
    embeddings = my_embedding_model.encode(
        sentences=[x["combined_sentence"] for x in sentences]
    )
    for i, elem in enumerate(sentences):
        print(elem)
        elem["combined_sentence_embedding"] = embeddings[i]

    for elem in sentences:
        print(elem)

    distances, sentences = calculate_cosine_distances(sentences=sentences)

    print(sentences[-2]["distance_to_next"])

    y_upper_bound = 0.15
    plt.ylim(0, y_upper_bound)
    plt.xlim(0, len(distances))

    # percentile()
    brekpoint_percentile_threshold = 20
    brekpoint_distance_threshold = np.percentile(
        distances, brekpoint_percentile_threshold
    )

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

        plt.axvspan(
            start_index, end_index, facecolor=colors[i % len(colors)], alpha=0.25
        )
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
        combined_text = " ".join([d["sentence"] for d in sentences[start_index:]])
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
    print(len(chunks))
    with open("004-txt", "a") as fp:
        for chunk in chunks:
            fp.writelines(spell_check(chunk) + "\n")


def spell_check(input: str) -> str:
    fix_spelling = pipeline(
        "text2text-generation", model="oliverguhr/spelling-correction-english-base"
    )
    print(fix_spelling(input, max_length=2048))
    return fix_spelling(input, max_length=2048)[0]["generated_text"]


if __name__ == "__main__":
    # main()
    with open("ency-list.txt") as fp:
       files = fp.readlines()
    for filename in files:
        print(filename) 
        filename = filename.strip()
        output_filename = filename.replace("ency", "ency-output")
        output_filename = output_filename.replace(".md", ".txt")
        output_filename = output_filename.replace(" ", "-")
        print(output_filename)
        generate_sentences_list_v1(filename, output_filename)
