import re
from typing import List

def split_into_word_chunks(text: str, words_per_chunk: int = 220, overlap_words: int = 40) -> List[str]:
    tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    chunks = []
    start = 0
    n = len(tokens)
    if n == 0:
        return chunks
    while start < n:
        end = min(start + words_per_chunk, n)
        piece = " ".join(tokens[start:end])
        piece = re.sub(r"\s+([.,!?;:])", r"\1", piece)
        piece = re.sub(r"\s+", " ", piece).strip()
        if piece:
            chunks.append(piece)
        if end == n:
            break
        start = max(0, end - overlap_words)
    return chunks

