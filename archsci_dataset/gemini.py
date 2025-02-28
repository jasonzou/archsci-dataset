from sentence_transformers import SentenceTransformer  # noqa: E402
from transformers import pipeline
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import re


def plot_chunk(chunk_doc, embedding_name):
    """
    Plots the histogram of token lengths for a list of documents.

    Args:
        chunk_doc (list): A list of documents, where each document is expected to have a 'page_content' attribute.
        embedding_name (str): The name of the embedding model used for tokenization.

    Raises:
        TypeError: If `chunk_doc` is not a list or if elements within the list don't have a `page_content` attribute.
        ValueError: If `embedding_name` is not a valid model name recognizable by `AutoTokenizer`.
    """
    if not isinstance(chunk_doc, list):
        raise TypeError("chunk_doc must be a list")
    for doc in chunk_doc:
        if not hasattr(doc, 'page_content'):
            raise TypeError("Elements in chunk_doc must have a 'page_content' attribute.")

    tokenizer = AutoTokenizer.from_pretrained(embedding_name)
    length = [len(tokenizer.encode(doc.page_content)) for doc in chunk_doc]
    fig = pd.Series(length).hist()
    plt.show()


def combine_sentences(sentences, buffer_size=1):
    """
    Combines sentences within a specified buffer range.

    For each sentence in the input list, it creates a new "combined_sentence"
    by concatenating the current sentence with its neighboring sentences
    within the buffer range.

    Args:
        sentences (list): A list of dictionaries, where each dictionary
                          represents a sentence and must contain a
                          "sentence" key.
        buffer_size (int, optional): The number of neighboring sentences to
                                     include on either side. Defaults to 1.

    Returns:
        list: A new list of dictionaries, where each dictionary now includes
              a "combined_sentence" key containing the concatenated sentence.

    Raises:
        TypeError: If `sentences` is not a list or if elements within the list are not dictionary or missing 'sentence' key.
        ValueError: If `buffer_size` is negative.
    """
    if not isinstance(sentences, list):
        raise TypeError("sentences must be a list")
    for sentence in sentences:
        if not isinstance(sentence, dict) or 'sentence' not in sentence:
            raise TypeError("Elements in sentences must be a dictionary with 'sentence' key.")
    if buffer_size < 0:
        raise ValueError("buffer_size must be non-negative")

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
    """
    Calculates the cosine similarity between two vectors.

    Args:
        vec1 (numpy.ndarray): The first vector.
        vec2 (numpy.ndarray): The second vector.

    Returns:
        float: The cosine similarity between the two vectors.

    Raises:
        TypeError: If `vec1` or `vec2` are not numpy arrays.
        ValueError: If `vec1` or `vec2` are empty arrays.
        ValueError: If `vec1` or `vec2` are not one-dimensional.
    """
    if not isinstance(vec1, np.ndarray) or not isinstance(vec2, np.ndarray):
      raise TypeError("vec1 and vec2 must be numpy arrays.")
    if vec1.size == 0 or vec2.size == 0:
        raise ValueError("vec1 and vec2 must not be empty.")
    if vec1.ndim != 1 or vec2.ndim != 1:
        raise ValueError("vec1 and vec2 must be one-dimensional.")

    # Calculate the similarity between two vectors
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def calculate_cosine_distances(sentences):
    """
    Calculates the cosine distance between consecutive sentence embeddings.

    For each consecutive pair of sentences in the input list, it calculates
    the cosine distance between their "combined_sentence_embedding" and stores
    the distance in a "distance_to_next" key in the first sentence of the pair.

    Args:
        sentences (list): A list of dictionaries, where each dictionary
                          represents a sentence and must contain a
                          "combined_sentence_embedding" key.

    Returns:
        tuple: A tuple containing:
               - distances (list): A list of cosine distances between consecutive
                                   sentence embeddings.
               - sentences (list): The modified list of dictionaries, with the
                                   "distance_to_next" key added to each
                                   applicable sentence.

    Raises:
        TypeError: if sentences is not a list or the elements are not dictionary or missing 'combined_sentence_embedding' key.
    """
    if not isinstance(sentences, list):
        raise TypeError("sentences must be a list")
    for sentence in sentences:
        if not isinstance(sentence, dict) or 'combined_sentence_embedding' not in sentence:
            raise TypeError("Elements in sentences must be a dictionary with 'combined_sentence_embedding' key.")


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
    """
    Generates a list of cleaned sentences from a text file, performs basic merging,
    and saves them to an output file. It does not perform embedding or chunking.

    This function reads a text file, splits it into sentences, cleans up the sentences
    (removing extra whitespace and line breaks), merges sentences where the latter starts with a lowercase letter,
    and optionally merges sentences starting with "keywords:" or "related entries:".
    Finally, it writes the processed sentences to an output text file.

    Args:
        filename (str): The path to the input text file.
        output_filename (str): The path to the output text file where the processed sentences will be saved.

    Raises:
        FileNotFoundError: If the input `filename` does not exist.
        OSError: If there is an error in file operations.
        TypeError: if `filename` or `output_filename` are not strings.
        ValueError: if `filename` or `output_filename` are empty.
    """

    if not isinstance(filename, str) or not isinstance(output_filename, str):
        raise TypeError("filename and output_filename must be string.")
    if not filename or not output_filename:
        raise ValueError("filename and output_filename must be not empty.")

    embedding_model = "NovaSearch/stella_en_1.5B_v5"
    
    my_embedding_model = SentenceTransformer(embedding_model)
    max_seq_length = my_embedding_model.max_seq_length
    print("embedding model -- max seq length --> {}".format(max_seq_length))

    tokenizer = AutoTokenizer.from_pretrained(embedding_model)

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
    """
    Generates a list of sentences from a text file, calculates cosine distances,
    identifies breakpoints, and saves chunks of text to an output file.

    This function reads a text file, splits it into sentences, cleans up the sentences,
    merges sentences where the latter starts with a lowercase letter,
    combines sentences with their neighbors, calculates cosine distances between embeddings
    of combined sentences, identifies breakpoints based on a distance percentile, and
    chunks the text based on these breakpoints. It also generates a plot visualizing the
    distances and the identified chunks. Finally, it saves the identified chunks to an output text file.

     Warning: this function do not have input filename parameter, which means you can't process different files.
     and it's hard to modify.

    """
    embedding_model = "/home/jason/workspace/models/bge-large-en-v1.5"

    my_embedding_model = SentenceTransformer(embedding_model)
    max_seq_length = my_embedding_model.max_seq_length
    print("embedding model -- max seq length --> {}".format(max_seq_length))

    tokenizer = AutoTokenizer.from_pretrained(embedding_model)

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

    for
