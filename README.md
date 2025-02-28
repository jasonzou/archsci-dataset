# archsci-dataset

Data generation

    Curator: Synthetic data generation tool that makes it easy to build pipelines around LLMs, use batching, and view data in progress. https://github.com/bespokelabsai/curator/
    Distilabel: General-purpose framework that can generate and augment data (SFT, DPO) with techniques like UltraFeedback and DEITA. https://github.com/argilla-io/distilabel
    Augmentoolkit: Framework to convert raw text into datasets using open-source and closed-source models. https://github.com/e-p-armstrong/augmentoolkit
    Data Prep Kit: Framework for data preparation for both code and language, with modules in Python, Ray, and Spark, and a wide range of scale from laptops to data centers. https://github.com/IBM/data-prep-kit

## Chunk textual documents

### Markdown files

- split by sections (^# and ^##)
- split by headings (^#)
- split by paragraph (empty line)

### chunk_size with Embedding model and LLM model

- chunk_size < MaxTokens of your embedding model = 512
-          overlap 10-25%;

chunk_size < LLM Max sequence length (top_k)

- CharacterTextSplitter
- RecursiveCharacterTextSplitter
- Document Specific Splitting
- Semantic Splitting