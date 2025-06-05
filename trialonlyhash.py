import os
import cv2
import hashlib
import hmac
import secrets

# Function to generate a random secret key
def generate_secret_key():
    return secrets.token_bytes(32)  # Generate a 32-byte (256-bit) random key

# Function to perform iterative hashing using BLAKE2 algorithm
def iterative_blake2_hash(data, rounds):
    blake2_hash = hashlib.blake2b(data, digest_size=32)
    for _ in range(rounds):
        blake2_hash.update(blake2_hash.digest())
    return blake2_hash.digest()

# Function to compute HMAC hash
def compute_hmac_hash(data, secret_key):
    h = hmac.new(secret_key, digestmod=hashlib.sha256)
    h.update(data)
    return h.hexdigest()

# Load the sample image
sample_path = "C:\\Users\\prana\\Downloads\\Fingerprint Recognition Project\\SOCOFing\\Altered\\Altered-Hard\\117__F_Right_little_finger_Obl.BMP"
sample = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)

if sample is None:
    print("Error: Unable to read the image at", sample_path)
    exit()

best_score = 0
best_matching_image = None
filename = None
best_matching_hash = None

# Generate a random secret key
secret_key = generate_secret_key()

# Compute iterative BLAKE2 hash value for the sample image
sample_hash = iterative_blake2_hash(sample.tobytes(), rounds=10)

# Define a threshold for the score
threshold = 0  # You can adjust this threshold as needed

# Loop through real fingerprint images
for file in os.listdir("C:\\Users\\prana\\Downloads\\Fingerprint Recognition Project\\SOCOFing\\Real")[:1000]:
    fingerprint_image_path = os.path.join("C:\\Users\\prana\\Downloads\\Fingerprint Recognition Project\\SOCOFing\\Real", file)
    fingerprint_image = cv2.imread(fingerprint_image_path, cv2.IMREAD_GRAYSCALE)
    
    # Ensure the read image is not None
    if fingerprint_image is None:
        continue

    # Compute iterative BLAKE2 hash value for the fingerprint image
    fingerprint_hash = iterative_blake2_hash(fingerprint_image.tobytes(), rounds=10)

    # Compute HMAC hash for the BLAKE2 hash of the fingerprint image
    hmac_hash = compute_hmac_hash(fingerprint_hash, secret_key)

    # Compare the BLAKE2 hash values
    similarity_score = sum(a == b for a, b in zip(sample_hash, fingerprint_hash)) / len(sample_hash) * 100

    if similarity_score > threshold:
        if similarity_score > best_score:
            best_score = similarity_score
            best_matching_image = fingerprint_image
            filename = file
            best_matching_hash = hmac_hash  # Store HMAC hash instead of BLAKE2 hash

print("BEST MATCH:", filename)
print("Score:", best_score)
print("HMAC Hash of Best Matching Image:", best_matching_hash)

# Display the best matching image if found
if best_matching_image is not None:
    cv2.imshow("Best Matching Image", best_matching_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No matching image found.")
