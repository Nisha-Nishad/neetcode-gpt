import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        # Create position vector of shape (seq_len, 1)
        pos = np.arange(seq_len, dtype=np.float64)[:, np.newaxis]
        
        # Create division term for even dimension indices: 10000^(2i / d_model)
        i = np.arange(0, d_model, 2, dtype=np.float64)
        div_term = np.power(10000.0, i / d_model)
        
        # Compute angles using broadcasting: shape (seq_len, d_model // 2)
        angles = pos / div_term
        
        # Initialize output array and assign sine to even columns, cosine to odd columns
        pe = np.zeros((seq_len, d_model), dtype=np.float64)
        pe[:, 0::2] = np.sin(angles)
        pe[:, 1::2] = np.cos(angles)
        
        return np.round(pe, 5)