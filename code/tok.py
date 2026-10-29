"""Tokenizer wrapper: 'gpt2' (tiktoken BPE, no neural weights) or a custom
byte-level BPE json trained by train_tokenizer.py (HuggingFace `tokenizers`)."""


class Tok:
    def __init__(self, spec):
        self.spec = spec
        if spec == "gpt2":
            import tiktoken
            self._enc = tiktoken.get_encoding("gpt2")
            self.kind = "tiktoken"
            self.n_vocab = self._enc.n_vocab
        else:
            from tokenizers import Tokenizer
            self._tok = Tokenizer.from_file(spec)
            self.kind = "hf"
            self.n_vocab = self._tok.get_vocab_size()

    def encode_batch(self, texts):
        if self.kind == "tiktoken":
            return self._enc.encode_ordinary_batch(texts)
        return [e.ids for e in self._tok.encode_batch(texts, add_special_tokens=False)]

    def encode(self, text):
        return self.encode_batch([text])[0]

    def decode(self, ids):
        ids = [int(i) for i in ids]
        if self.kind == "tiktoken":
            return self._enc.decode(ids)
        return self._tok.decode(ids, skip_special_tokens=False)
