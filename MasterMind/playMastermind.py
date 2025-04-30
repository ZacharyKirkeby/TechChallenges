import requests, json, sys, itertools, time

# CONFIG
level = 1                       # Level to solve
base_url = 'https://mastermind.praetorian.com'

# Args parsing
if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <email> [restart_on_fail]")
    sys.exit(1)


if sys.version_info < (3,0):
  sys.exit('Python version < 3.0 does not support modern TLS versions. You will have trouble connecting to our API using Python 2.X.')

email = sys.argv[1]
restart_on_fail = True  # Default
if len(sys.argv) >= 3:
    restart_on_fail = sys.argv[2].lower() in ('true', '1', 'yes')

# API Auth
r = requests.post(f'{base_url}/api-auth-token/', data={'email': email})
r.json()
#r.raise_for_status()
headers = r.json()
headers['Content-Type'] = 'application/json'

# Helper to calculate feedback
def get_feedback(guess, answer):
    correct_pos = sum(g == a for g, a in zip(guess, answer))
    correct_weapons = sum(min(guess.count(w), answer.count(w)) for w in set(guess))
    return [correct_weapons, correct_pos]

def solve_level():
    r = requests.get(f'{base_url}/level/{level}/', headers=headers)
    r.raise_for_status()
    level_info = r.json()
    
    num_gladiators = level_info['numGladiators']
    num_weapons = level_info['numWeapons']
    num_guesses = level_info['numGuesses']
    num_rounds = level_info['numRounds']

    weapons = list(range(num_weapons))
    for round_num in range(num_rounds):
        print(f"Starting round {round_num+1}/{num_rounds}")
        candidates = list(itertools.permutations(weapons, num_gladiators))
        for attempt in range(num_guesses):
            if not candidates:
                print(f"No candidates remain, failed at attempt {attempt+1}")
                return False
            guess = list(candidates[0])
            r = requests.post(f'{base_url}/level/{level}/', data=json.dumps({'guess': guess}), headers=headers)
            r.raise_for_status()
            resp = r.json()
            response = resp.get('response')
            if response == [num_gladiators, num_gladiators]:
                print(f"Round {round_num+1} solved with guess {guess}")
                break
            candidates = [c for c in candidates if get_feedback(guess, c) == response]
        else:
            print(f"Failed to solve round {round_num+1}")
            return False
    return True

# Main Loop
while True:
    success = solve_level()
    if success:
        print("Level completed successfully.")
        break
    elif restart_on_fail:
        print("Restarting level...")
        time.sleep(1)
    else:
        print("Level failed.")
        break

