from urllib.parse import quote_plus
# Code for Wikipedia entity linking taken from:
# https://huggingface.co/facebook/genre-kilt
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils.config import TOPIC_MODEL

# OPTIONAL: load the prefix tree (trie), you need to additionally download
# https://huggingface.co/facebook/genre-kilt/blob/main/trie.py and
# https://huggingface.co/facebook/genre-kilt/blob/main/kilt_titles_trie_dict.pkl
# import pickle
# from trie import Trie
# with open("kilt_titles_trie_dict.pkl", "rb") as f:
#     trie = Trie.load_from_dict(pickle.load(f))

def generate_wikipedia_entities(query):

    tokenizer = AutoTokenizer.from_pretrained(TOPIC_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(TOPIC_MODEL).eval()

    outputs = model.generate(
        **tokenizer([query], return_tensors="pt"),
        num_beams=5,
        num_return_sequences=2,
        # OPTIONAL: use constrained beam search
        # prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist()),
    )

    topics = []
    for topic in tokenizer.batch_decode(outputs, skip_special_tokens=True):
        topics.append(wikipedia_search_url(topic))

    return topics


def wikipedia_search_url(query):
    return f"https://en.wikipedia.org/wiki/Special:Search?search={quote_plus(query)}&go=Go"