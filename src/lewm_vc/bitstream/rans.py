"""
Simple correct rANS entropy coder.

Uses 32-bit state with standard renormalization:
- State always in [L, 2^31) where L = 2^23
- Encode: pack symbol, output bytes while state >= 2^24
- Decode: read bytes while state < L, unpack symbol
"""

import numpy as np

M = 1 << 16  # total probability (65536)
L = 1 << 23  # state lower bound (8388608)


def discretize_gmm(mu, sigma, weight, num_bins=256, num_std=4):
    mu = np.atleast_1d(np.asarray(mu, dtype=np.float64))
    sigma = np.atleast_1d(np.asarray(sigma, dtype=np.float64))
    weight = np.atleast_1d(np.asarray(weight, dtype=np.float64))

    mu_avg = (mu * weight).sum()
    sigma_max = sigma.max()
    lo = mu_avg - num_std * sigma_max
    hi = mu_avg + num_std * sigma_max
    if hi - lo < 1e-8:
        hi = lo + 1.0
    bins = np.linspace(lo, hi, num_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    pdf = np.zeros(num_bins)
    for k in range(len(mu)):
        pdf += weight[k] * _norm_pdf(centers, mu[k], max(sigma[k], 1e-8))
    pdf = pdf / pdf.sum() * M
    pdf = np.maximum(pdf, 1.0)
    pdf = pdf / pdf.sum() * M
    cdf = np.zeros(num_bins + 1, dtype=np.int64)
    cdf[1:] = np.cumsum(pdf).astype(np.int64)
    cdf[0] = 0
    cdf[-1] = M

    # Build inverse decode tables (one entry per probability slot)
    start_lu = np.zeros(M, dtype=np.int32)
    count_lu = np.ones(M, dtype=np.int32)
    pos = 0
    for sym in range(num_bins):
        freq = int(cdf[sym + 1] - cdf[sym])
        for _ in range(freq):
            start_lu[pos] = int(cdf[sym])
            count_lu[pos] = freq
            pos += 1
    return cdf.astype(np.int32), start_lu, count_lu, bins, centers


def _norm_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


class RangeEncoder:
    def __init__(self):
        self.state = L
        self.bytes = bytearray()

    def encode(self, symbol: int, cdf: np.ndarray):
        freq = int(cdf[symbol + 1] - cdf[symbol])
        start = int(cdf[symbol])
        if freq < 1:
            freq = 1

        # Standard rANS encode: pack symbol into state
        q, r = divmod(self.state, freq)
        self.state = q * M + r + start

        # Renormalize: output bytes while state >= (L << 1)
        while self.state >= (L << 1):
            self.bytes.append(self.state & 0xFF)
            self.state >>= 8

    def flush(self):
        # Store final state as 4-byte big-endian
        buf = bytearray()
        state = self.state
        for _ in range(4):
            buf.append(state & 0xFF)
            state >>= 8
        self.bytes.extend(reversed(buf))
        return bytes(self.bytes)


class RangeDecoder:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = len(data) - 4
        self.state = int.from_bytes(data[self.pos : self.pos + 4], "big")
        self.pos -= 1

    def decode(self, cdf: np.ndarray, start_lookup: np.ndarray, count_lookup: np.ndarray) -> int:
        slot = self.state % M
        start = int(start_lookup[slot])
        freq = int(count_lookup[slot])

        self.state = freq * (self.state // M) + (self.state % M) - start

        while self.state < L and self.pos >= 0:
            self.state = (self.state << 8) | self.data[self.pos]
            self.pos -= 1

        lo, hi = 0, len(cdf) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if cdf[mid] > start:
                hi = mid - 1
            else:
                lo = mid
        return lo
