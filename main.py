import os
import cv2

# Load the sample image
image_path = "C:\\Users\\prana\\Downloads\\Fingerprint Recognition Project\\SOCOFing\\Altered\\Altered-Hard\\114__F_Right_little_finger_Obl.BMP"
sample = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if sample is None:
    print("Error: Unable to read the image at", image_path)
    exit()

best_score = 0
filename = None
image = None
kp1, kp2, mp = None, None, None

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
        if m.distance < 0.65 * n.distance:  # Adjusted threshold
            good_matches.append(m)

    # Calculate score
    score = len(good_matches) / max(len(keypoints_1), len(keypoints_2)) * 100

    if score > best_score:
        best_score = score
        filename = file
        image = fingerprint_image
        kp1, kp2, mp = keypoints_1, keypoints_2, good_matches

print("BEST MATCH:", filename)
print("Score:", best_score)

# Draw matches
if image is not None:
    result = cv2.drawMatches(sample, kp1, image, kp2, mp, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    result = cv2.resize(result, None, fx=0.5, fy=0.5)
    cv2.imshow("Results", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No matching image found.")
