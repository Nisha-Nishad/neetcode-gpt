from typing import List
from collections import Counter


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        tokens = list(corpus)
        merges = []

        for _ in range(num_merges):
            if len(tokens) < 2:
                break

            # Count adjacent pairs
            counts = Counter(zip(tokens, tokens[1:]))

            # Highest frequency; ties -> lexicographically smallest pair
            best = min(counts, key=lambda p: (-counts[p], p))
            a, b = best
            merges.append([a, b])

            # Merge non-overlapping occurrences, left to right
            merged = []
            i, n = 0, len(tokens)
            while i < n:
                if i < n - 1 and tokens[i] == a and tokens[i + 1] == b:
                    merged.append(a + b)
                    i += 2
                else:
                    merged.append(tokens[i])
                    i += 1
            tokens = merged

        return merges