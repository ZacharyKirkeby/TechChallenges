import logging
import base64
import time
import random
import os
import numpy as np
from collections import Counter
from scipy.stats import entropy, skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import argparse

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class Server(object):
    url = 'https://mlb.praetorian.com'
    log = logging.getLogger(__name__)

    def __init__(self):
        self.session = requests.session()
        self.binary  = None
        self.hash    = None
        self.wins    = 0
        self.targets = []

    def _request(self, route, method='get', data=None):
        while True:
            try:
                if method == 'get':
                    r = self.session.get(self.url + route)
                else:
                    r = self.session.post(self.url + route, data=data)
                if r.status_code == 429:
                    raise Exception('Rate Limit Exception')
                if r.status_code == 500:
                    raise Exception('Unknown Server Exception')
                return r.json()
            except Exception as e:
                self.log.error(e)
                self.log.info('Waiting 60 seconds before next request')
                time.sleep(60)

    def get(self):
        r = self._request("/challenge")
        self.targets = r.get('target', [])
        self.binary  = base64.b64decode(r.get('binary', ''))
        return r

    def post(self, target):
        r = self._request("/solve", method="post", data={"target": target})
        self.wins = r.get('correct', 0)
        self.hash = r.get('hash', self.hash)
        self.ans  = r.get('target', 'unknown')
        return r

def extract_endianness_features(hex_bytes):
    big_score = 0
    little_score = 0

    for i in range(0, len(hex_bytes) - 4, 4):
        word = bytes(hex_bytes[i:i+4])  # Convert slice to bytes

        # Big-endian opcode patterns (e.g., opcode in first byte)
        if word[0] in {0x00, 0x3C, 0x8F, 0x27, 0xAF}:
            big_score += 1

        # Little-endian opcode pattern (e.g., opcode in last byte)
        if word[3] in {0x00, 0x3C, 0x8F, 0x27, 0xAF}:
            little_score += 1

        # Immediate pattern (e.g., 0x0008 encoded as 00 08 or 08 00)
        if word[:2] == b'\x00\x08':
            big_score += 0.5
        if word[2:] == b'\x08\x00':
            little_score += 0.5

    total = big_score + little_score + 1e-6
    big_ratio = big_score / total
    little_ratio = little_score / total

    return [big_score, little_score, big_ratio, little_ratio]

def sliding_window_entropy(data, window_size=16, stride=8):
    entropies = []
    for i in range(0, len(data) - window_size + 1, stride):
        window = data[i:i+window_size]
        probs = np.bincount(window, minlength=256) / window_size
        probs = probs[probs > 0]
        entropies.append(entropy(probs))
    # Aggregate: mean, std, max
    if entropies:
        return [np.mean(entropies), np.std(entropies), max(entropies)]
    return [0.0, 0.0, 0.0]

def longest_byte_run(data):
    max_run = run = 1
    for i in range(1, len(data)):
        if data[i] == data[i - 1]:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1
    return max_run


def extract_features(data: bytes):
    hex_bytes = np.frombuffer(data, dtype=np.uint8)
    total = len(hex_bytes)

    # Byte frequency (first 16 values)
    freq = np.bincount(hex_bytes, minlength=256)[:16]
    freq = freq / total

    # Entropy
    probs = freq[freq > 0]
    ent = entropy(probs)

    # Bigram frequency (top 10)
    bigrams = Counter()
    for i in range(len(hex_bytes) - 1):
        bigrams[(hex_bytes[i], hex_bytes[i + 1])] += 1
    top_bigrams = bigrams.most_common(10)
    bigram_features = [count / total for (_, count) in top_bigrams]
    bigram_features += [0] * (10 - len(bigram_features))  # pad

    # Statistical moments
    mean_val = np.mean(hex_bytes)
    var_val = np.var(hex_bytes)
    skew_val = skew(hex_bytes)
    kurt_val = kurtosis(hex_bytes)

    # Byte presence bitvector
    byte_presence = (np.bincount(hex_bytes, minlength=256) > 0).astype(int)[:256]

    # Architecture-specific opcode frequency (normalized)
        opcode_map = {
        # x86
        'x86_call': 0xE8,
        'x86_jmp':  0xE9,
        'x86_misc': 0xFF,

        # MIPS
        'mips_lui': 0x3C,
        'mips_lw':  0x8C,

        # ARM
        'arm_b':    0xEA,

        # SH4
        'sh4_trapa': 0xC3,   # TRAPA #xx
        'sh4_mov_l': 0xD0,   # MOV.L Rm,@(disp,Rn)

        # s390 (big-endian, 2-byte or 4-byte instructions)
        's390_prefix': 0xA7,  # Common prefix for many instructions
        's390_lay':    0xE3,  # LAY and other extended format
    }

    opcode_counts = []
    for name, code in opcode_map.items():
        count = np.sum(hex_bytes == code)
        opcode_counts.append(count / total)

    endian_feats = extract_endianness_features(hex_bytes)
    sliding_entropy = sliding_window_entropy(hex_bytes)
    longest_run = longest_byte_run(hex_bytes)
    return (
        list(freq)
        + [ent]
        + bigram_features
        + [mean_val, var_val, skew_val, kurt_val]
        + list(byte_presence)
        + opcode_counts
        + endian_feats
        + sliding_entropy
        + longest_run
    )

def save_data(X, y, path='data.npz'):
    np.savez_compressed(path, X=np.array(X), y=np.array(y))

def load_data(path='data.npz'):
    if os.path.exists(path):
        data = np.load(path, allow_pickle=True)
        return data['X'].tolist(), data['y'].tolist()
    return [], []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cache", action="store_true", help="Ignore saved dataset and start from scratch")
    parser.add_argument("--save", action="store_true", help="Save training data after each sample")
    args = parser.parse_args()

    s = Server()
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, oob_score=True, min_samples_split=5)
    label_encoder = LabelEncoder()

    # Load previous data
    if args.no_cache:
        X, y = [], []
        logging.info("Starting with empty dataset.")
    else:
        X, y = load_data()
        logging.info(f"Loaded dataset with {len(y)} samples.")

    streak = 0

    for i in range(100000):
        s.get()
        features = extract_features(s.binary)

        if len(y) >= 10:
            label_encoder.fit(y)
            clf.fit(X, label_encoder.transform(y))
            probs = clf.predict_proba([features])[0]
            label_to_index = {label: i for i, label in enumerate(label_encoder.classes_)}
            valid_probs = [(target, probs[label_to_index[target]]) for target in s.targets if target in label_to_index]
            target = max(valid_probs, key=lambda x: x[1])[0] if valid_probs else random.choice(s.targets)
        else:
            target = random.choice(s.targets)

        s.post(target)
        correct = (target == s.ans)
        streak = streak + 1 if correct else 0

        logging.info("Guess:[{: >9}]   Answer:[{: >9}]   Wins:[{: >3}]   Streak:[{: >3}]".format(
            target, s.ans, s.wins, streak))

        # Update dataset and retrain on-the-fly
        X.append(features)
        y.append(s.ans)
        label_encoder.fit(y)
        clf.fit(X, label_encoder.transform(y))

        if args.save:
            save_data(X, y)

        if s.hash:
            logging.info("You win! {}".format(s.hash))
            break
