from typing import List, Dict


class Solution:
    def _greedy(self, text: str, vocab: Dict[str, int], max_len: int):
        """Yield tokens using greedy left-to-right longest match."""
        i, n = 0, len(text)
        while i < n:
            # Try the longest possible candidate first, shrink until a match.
            j = min(n, i + max_len)
            while j > i + 1 and text[i:j] not in vocab:
                j -= 1
            # If nothing longer matched, j == i + 1: a single char
            # (whether or not it is in vocab, we consume it).
            yield text[i:j]
            i = j

    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        max_len = max((len(k) for k in vocab), default=1)
        return [list(self._greedy(str(num), vocab, max_len)) for num in numbers]

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        max_len = max((len(k) for k in vocab), default=1)
        return sum(1 for _ in self._greedy(text, vocab, max_len))

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        words = text.split()
        if not words:
            return 0.0
        return round(self.count_tokens(text, vocab) / len(words), 4)