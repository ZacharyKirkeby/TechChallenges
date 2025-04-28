import requests
import os
from solve import caesar_cipher, decode_base64

try: input = raw_input
except NameError: pass

# Global values
base = "http://crypto.praetorian.com/{}"
email = input("Enter your email address: ")
auth_token = None

# Used for authentication
def token(email):
	global auth_token
	if not auth_token:
		url = base.format("api-token-auth/")
		resp = requests.post(url, data={"email":email})
		auth_token = {"Authorization":"JWT " + resp.json()['token']}
		resp.close()
	return auth_token

# Fetch the challenge and hint for level n
def fetch(n):
	url = base.format("challenge/{}/".format(n))
	resp = requests.get(url, headers=token(email))
	resp.close()
	if resp.status_code != 200:
		raise Exception(resp.json()['detail'])
	return resp.json()

# Submit a guess for level n
def solve(n, guess):
	url = base.format("challenge/{}/".format(n))
	data = {"guess": guess}
	resp = requests.post(url, headers=token(email), data=data)
	resp.close()
	if resp.status_code != 200:
		raise Exception(resp.json()['detail'])
	return resp.json()


hashes = {}
for level in range(7):
    print("\n--- Level {} ---".format(level))
    data = fetch(level)

    if level == 0:
        # Level 0 is a freebie
        guess = data['challenge']
        print("Level 0 guess (freebie):", guess)
    elif level == 1:
        cipher =data['challenge']
        guess = caesar_cipher(cipher);
        print("Level 1 guess", guess)
    elif level == 2:
        cipher = data['challenge']
        if not os.path.exists('output.png'):
            decode_base64(cipher, 'output.png')
            guess = input("Your guess: ")
        else:
            guess = "PRAE2014AN"
        print("Level 2 guess", guess)
    else:
        print("Challenge:", data['challenge'])
        guess = input("Your guess: ")

    h = solve(level, guess)

    if 'hash' in h:
        hashes[level] = h['hash']
        print("Correct! Hash:", h['hash'])
        with open("hashes.txt", "w") as f:
            f.write("\n".join(hashes.values()))
    else:
        print("No hash returned.")
