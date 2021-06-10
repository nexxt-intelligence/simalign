from .simalign import SentenceAligner

# Invoke model download
myaligner = SentenceAligner(model="bert", token_type="bpe", matching_methods="mai")
