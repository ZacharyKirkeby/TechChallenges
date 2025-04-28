import base64

def caesar_cipher(text, shift=23):
    result = []
    for char in text:
        if char.isalpha():
            base = ord('A') if char.isupper() else ord('a')
            shifted = (ord(char) - base + shift) % 26 + base
            result.append(chr(shifted))
        else:
            result.append(char)

    return ''.join(result)

def decode_base64(data_url: str, output_path: str) -> None:
    if not data_url.startswith('data:image/'):
        raise ValueError("Invalid Input.")
    header, encoded = data_url.split(',', 1)
    data = base64.b64decode(encoded)
    with open(output_path, 'wb') as f:
        f.write(data)


