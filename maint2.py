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
sample_path = "C:\\Users\\prana\\Downloads\\Fingerprint Recognition Project\\SOCOFing\\Altered\\Altered-Hard\\119__F_Right_little_finger_Obl.BMP"
sample = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)

if sample is None:
    print("Error: Unable to read the image at", sample_path)
    exit()

best_score = 0
best_matching_image = None
filename = None
best_matching_hash = None
kp1, kp2, mp = None, None, None

# Generate a random secret key
secret_key = generate_secret_key()

# Loop through real fingerprint images
for file in os.listdir("C:\\Users\\prana\\Downloads\\Fingerprint Recognition Project\\SOCOFing\\Real")[:1000]:
    fingerprint_image_path = os.path.join("C:\\Users\\prana\\Downloads\\Fingerprint Recognition Project\\SOCOFing\\Real", file)
    fingerprint_image = cv2.imread(fingerprint_image_path, cv2.IMREAD_GRAYSCALE)
    
    # Ensure the read image is not None
    if fingerprint_image is None:
        continue

    # Perform SIFT feature matching
    sift = cv2.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(sample, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_image, None)

    # Perform feature matching and scoring
    matches = cv2.BFMatcher().knnMatch(descriptors_1, descriptors_2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Calculate score
    score = len(good_matches) / max(len(keypoints_1), len(keypoints_2)) * 100

    # Compute iterative BLAKE2 hash values for the sample and fingerprint images
    sample_hash = iterative_blake2_hash(sample.tobytes(), rounds=10)
    fingerprint_hash = iterative_blake2_hash(fingerprint_image.tobytes(), rounds=10)

    # Compute HMAC hash for the BLAKE2 hash of the fingerprint image
    hmac_hash = compute_hmac_hash(fingerprint_hash, secret_key)

    # Compare the BLAKE2 hash values
    if sample_hash == fingerprint_hash:
        # If the BLAKE2 hashes match, consider it a perfect match
        score = 100

    if score > best_score:
        best_score = score
        best_matching_image = fingerprint_image
        filename = file
        best_matching_hash = hmac_hash  # Store HMAC hash instead of BLAKE2 hash
        kp1, kp2, mp = keypoints_1, keypoints_2, good_matches

print("BEST MATCH:", filename)
print("Score:", best_score)
print("HMAC Hash of Best Matching Image:", best_matching_hash)

# Draw matches
if best_matching_image is not None:
    result = cv2.drawMatches(sample, kp1, best_matching_image, kp2, mp, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    result = cv2.resize(result, None, fx=2.0, fy=2.0)  # Enlarge the result
    cv2.imshow("Results", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No matching image found.")
