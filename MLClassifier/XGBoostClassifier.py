import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
import signal
import sys
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Initial model and data
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
X = []
y = []
scaler = StandardScaler()

# Metrics for graphing
streaks = []
accuracies = []
confidences = []
current_streak = 0
total_correct = 0
attempts = 0

def extract_features(bytez):
    arr = [int(bytez[i:i+2], 16) for i in range(0, len(bytez), 2)]
    byte_counts = [0]*256
    for b in arr:
        byte_counts[b] += 1
    entropy = -sum((c/len(arr))*np.log2(c/len(arr)) for c in byte_counts if c > 0)
    avg = sum(arr) / len(arr)
    unique_bytes = len(set(arr))
    return byte_counts + [entropy, avg, unique_bytes]

def plot_graph_and_exit():
    print("\nInterrupted. Plotting learning progression...")
    if not streaks:
        print("No data to plot.")
        sys.exit(0)

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Streak #')
    ax1.set_ylabel('Accuracy', color='tab:blue')
    ax1.plot(range(len(streaks)), accuracies, color='tab:blue', label='Accuracy')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim([0, 1.05])

    ax2 = ax1.twinx()
    ax2.set_ylabel('Confidence', color='tab:red')
    ax2.plot(range(len(streaks)), confidences, color='tab:red', label='Confidence')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_ylim([0, 1.05])

    plt.title("Model Progression Over Streaks")
    fig.tight_layout()
    plt.show()
    sys.exit(0)

# Bind Ctrl-C (SIGINT) to graceful exit
signal.signal(signal.SIGINT, lambda sig, frame: plot_graph_and_exit())

def update_model():
    if len(set(y)) > 1:  # Need at least two classes
        X_scaled = scaler.fit_transform(X)
        model.fit(X_scaled, y)

def classify_sample(hexstr):
    global current_streak, total_correct, attempts

    features = extract_features(hexstr)
    if len(set(y)) > 1:
        X_scaled = scaler.transform([features])
        pred_proba = model.predict_proba(X_scaled)[0]
        guess = np.argmax(pred_proba)
        confidence = np.max(pred_proba)
    else:
        guess = random.randint(0, 9)
        confidence = 0.1

    return guess, confidence

# Example function to simulate guesses
def submit(hexstr, actual_arch):
    global current_streak, total_correct, attempts

    guess, conf = classify_sample(hexstr)
    is_correct = guess == actual_arch
    attempts += 1

    if is_correct:
        current_streak += 1
        total_correct += 1
    else:
        current_streak = 0

    confidences.append(conf)
    accuracies.append(total_correct / attempts)
    streaks.append(current_streak)

    print(f"Guess: {guess} (conf: {conf:.2f}) | Actual: {actual_arch} | {'✔️' if is_correct else '❌'} | Streak: {current_streak}")

    # Add to training set and retrain
    X.append(extract_features(hexstr))
    y.append(actual_arch)
    update_model()

# Dummy run (simulate interaction)
if __name__ == "__main__":
    arch_choices = list(range(10))  # Simulate 10 architectures

    while True:
        fake_hex = ''.join(random.choices('0123456789abcdef', k=64))
        correct_arch = random.choice(arch_choices)
        submit(fake_hex, correct_arch)

