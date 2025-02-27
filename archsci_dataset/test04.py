from archsci_dataset.preprocess_footnotes import preprocess_footnotes_file
from loguru import logger

def main():
  logger.info("hell0")
  preprocess_footnotes_file("./data/journals/00001.md")
  
if __name__ == "__main__":
  main()