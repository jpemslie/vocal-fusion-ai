"""
VocalFusion AI — Mix Predictor
================================

Learns from user ratings to improve mixing decisions.

Rating storage (vf_data/ratings.json):
  List of dicts: {mix_id, song_a, song_b, direction, key_shift,
                  vox_rms, inst_rms, rating, emb_a, emb_b, timestamp}

Pending params (vf_data/mix_params.json):
  Dict keyed by mix_id — same fields without 'rating'

Model (vf_data/mix_predictor.npz):
  Pure numpy 2-layer MLP weights.
  Input: concat(emb_a, emb_b, [direction_feat, key_shift_norm]) = 258-dim
  Output: predicted quality 0-1

Training: triggered automatically when >= MIN_RATINGS_TO_TRAIN ratings exist.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


MIN_RATINGS_TO_TRAIN = 3


class MixPredictor:
    """Learn from user ratings to suggest better mixing parameters."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.ratings_file = self.data_dir / "ratings.json"
        self.params_file = self.data_dir / "mix_params.json"
        self.model_file = self.data_dir / "mix_predictor.npz"
        self._weights = None
        self._load_model()

    # ------------------------------------------------------------------
    # RATING STORAGE
    # ------------------------------------------------------------------

    def save_mix_params(self, mix_id: str, song_a: str, song_b: str,
                        direction: str, key_shift: int,
                        vox_rms: float, inst_rms: float,
                        emb_a: Optional[np.ndarray] = None,
                        emb_b: Optional[np.ndarray] = None):
        """Save mixing parameters before user rates."""
        params = self._load_json(self.params_file, default={})
        params[mix_id] = {
            'song_a': song_a,
            'song_b': song_b,
            'direction': direction,
            'key_shift': int(key_shift),
            'vox_rms': float(vox_rms),
            'inst_rms': float(inst_rms),
            'emb_a': emb_a.tolist() if emb_a is not None else None,
            'emb_b': emb_b.tolist() if emb_b is not None else None,
            'timestamp': datetime.now().isoformat(),
        }
        self._save_json(self.params_file, params)

    def add_rating(self, mix_id: str, rating: int) -> bool:
        """
        Record a user rating (1-5) for a mix.
        Returns True if successfully saved.
        """
        rating = max(1, min(5, int(rating)))
        params = self._load_json(self.params_file, default={})
        if mix_id not in params:
            print(f"MixPredictor: mix_id '{mix_id}' not found in pending params")
            return False

        entry = dict(params[mix_id])
        entry['mix_id'] = mix_id
        entry['rating'] = rating
        entry['rated_at'] = datetime.now().isoformat()

        ratings = self._load_json(self.ratings_file, default=[])
        ratings = [r for r in ratings if r.get('mix_id') != mix_id]
        ratings.append(entry)
        self._save_json(self.ratings_file, ratings)

        n = len([r for r in ratings if 'rating' in r])
        print(f"MixPredictor: Rating {rating}/5 saved. Total rated: {n}")

        if n >= MIN_RATINGS_TO_TRAIN:
            self.train(ratings)

        return True

    def get_ratings_count(self) -> int:
        ratings = self._load_json(self.ratings_file, default=[])
        return len([r for r in ratings if 'rating' in r])

    # ------------------------------------------------------------------
    # DIRECTION SUGGESTION
    # ------------------------------------------------------------------

    def suggest_direction(self, song_a: str, song_b: str) -> Optional[str]:
        """
        Suggest direction based on past ratings for this song pair.
        Returns 'a_vocals', 'b_vocals', or None (not enough data).
        """
        ratings = self._load_json(self.ratings_file, default=[])
        rated = [r for r in ratings if 'rating' in r]

        if len(rated) < MIN_RATINGS_TO_TRAIN:
            return None

        # Find ratings for this exact pair (or flipped)
        pair = []
        for r in rated:
            a, b = r.get('song_a'), r.get('song_b')
            if a == song_a and b == song_b:
                pair.append((r['direction'], r['rating']))
            elif a == song_b and b == song_a:
                # Flip direction to match the requested orientation
                flipped = 'b_vocals' if r['direction'] == 'a_vocals' else 'a_vocals'
                pair.append((flipped, r['rating']))

        if not pair:
            return None

        scores: Dict[str, List[float]] = {}
        for d, rval in pair:
            scores.setdefault(d, []).append(float(rval))

        best_dir = max(scores, key=lambda d: sum(scores[d]) / len(scores[d]))
        avg = sum(scores[best_dir]) / len(scores[best_dir])
        print(f"MixPredictor: Suggesting direction '{best_dir}' "
              f"(avg {avg:.1f} from {len(pair)} past mixes)")
        return best_dir

    # ------------------------------------------------------------------
    # MODEL — pure numpy 2-layer MLP
    # ------------------------------------------------------------------

    def train(self, ratings: List[dict] = None):
        """Train a tiny MLP on rated mixes. Pure numpy, no torch needed."""
        if ratings is None:
            ratings = self._load_json(self.ratings_file, default=[])

        rated = [r for r in ratings
                 if 'rating' in r and r.get('emb_a') is not None]
        if len(rated) < MIN_RATINGS_TO_TRAIN:
            print(f"MixPredictor: Need {MIN_RATINGS_TO_TRAIN} rated mixes "
                  f"with embeddings (have {len(rated)})")
            return

        X_list, y_list = [], []
        for r in rated:
            try:
                ea = np.array(r['emb_a'], dtype=np.float32)
                eb = np.array(r['emb_b'], dtype=np.float32)
                dir_feat = np.array(
                    [1.0 if r['direction'] == 'a_vocals' else 0.0])
                ks_feat = np.array([r.get('key_shift', 0) / 6.0])
                x = np.concatenate([ea, eb, dir_feat, ks_feat])
                X_list.append(x)
                y_list.append(r['rating'] / 5.0)
            except Exception:
                continue

        if len(X_list) < MIN_RATINGS_TO_TRAIN:
            return

        # If embeddings changed size (e.g. 128-dim → 768-dim MERT), keep only
        # the most common dimension so np.array() doesn't throw a shape error.
        from collections import Counter
        lengths = [len(x) for x in X_list]
        dominant_len = Counter(lengths).most_common(1)[0][0]
        paired = [(x, yv) for x, yv in zip(X_list, y_list) if len(x) == dominant_len]
        if len(paired) < MIN_RATINGS_TO_TRAIN:
            print(f"MixPredictor: Not enough ratings with consistent embedding "
                  f"dim={dominant_len} (have {len(paired)})")
            return
        X_list, y_list = zip(*paired)

        X = np.array(X_list, dtype=np.float32)  # (N, D)
        y = np.array(y_list, dtype=np.float32)   # (N,)
        D = X.shape[1]

        print(f"MixPredictor: Training on {len(X_list)} mixes (input_dim={D})...")
        np.random.seed(42)
        W1 = np.random.randn(D, 64).astype(np.float32) * np.sqrt(2.0 / D)
        b1 = np.zeros(64, dtype=np.float32)
        W2 = np.random.randn(64, 32).astype(np.float32) * np.sqrt(2.0 / 64)
        b2 = np.zeros(32, dtype=np.float32)
        W3 = np.random.randn(32, 1).astype(np.float32) * np.sqrt(2.0 / 32)
        b3 = np.zeros(1, dtype=np.float32)

        lr = 0.01
        for epoch in range(500):
            h1 = np.maximum(0, X @ W1 + b1)
            h2 = np.maximum(0, h1 @ W2 + b2)
            out = 1.0 / (1.0 + np.exp(-(h2 @ W3 + b3)))
            pred = out.squeeze()
            loss = float(np.mean((pred - y) ** 2))

            dl = 2.0 * (pred - y) / len(y)
            dout = dl[:, None] * out * (1.0 - out)
            dW3 = h2.T @ dout
            db3 = dout.sum(axis=0)
            dh2 = (dout @ W3.T) * (h2 > 0)
            dW2 = h1.T @ dh2
            db2 = dh2.sum(axis=0)
            dh1 = (dh2 @ W2.T) * (h1 > 0)
            dW1 = X.T @ dh1
            db1 = dh1.sum(axis=0)

            W1 -= lr * dW1; b1 -= lr * db1
            W2 -= lr * dW2; b2 -= lr * db2
            W3 -= lr * dW3; b3 -= lr * db3

            if epoch % 100 == 0:
                print(f"    Epoch {epoch:4d}: loss={loss:.4f}")

        self._weights = (W1, b1, W2, b2, W3, b3)
        np.savez(str(self.model_file),
                 W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)
        print(f"MixPredictor: Model saved to {self.model_file}")

    def predict_quality(self, emb_a: np.ndarray, emb_b: np.ndarray,
                        direction: str, key_shift: int) -> float:
        """Predict quality score 0-1 for this pair + params."""
        if self._weights is None:
            return 0.5
        W1, b1, W2, b2, W3, b3 = self._weights
        dir_feat = np.array([1.0 if direction == 'a_vocals' else 0.0])
        ks_feat = np.array([key_shift / 6.0])
        x = np.concatenate([emb_a, emb_b, dir_feat, ks_feat])[None, :]
        if x.shape[1] != W1.shape[0]:
            # Embedding dim changed (e.g. 128→768 after MERT upgrade);
            # model needs to retrain before it can make predictions.
            return 0.5
        h1 = np.maximum(0, x @ W1 + b1)
        h2 = np.maximum(0, h1 @ W2 + b2)
        out = 1.0 / (1.0 + np.exp(-(h2 @ W3 + b3)))
        return float(out[0, 0])

    def _load_model(self):
        if self.model_file.exists():
            try:
                d = np.load(str(self.model_file))
                self._weights = (d['W1'], d['b1'],
                                 d['W2'], d['b2'],
                                 d['W3'], d['b3'])
                print("MixPredictor: Loaded trained model.")
            except Exception as e:
                print(f"MixPredictor: Could not load model: {e}")

    # ------------------------------------------------------------------
    # UTILITIES
    # ------------------------------------------------------------------

    def _load_json(self, path: Path, default=None):
        if default is None:
            default = {}
        if not path.exists():
            return default
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return default

    def _save_json(self, path: Path, data):
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
