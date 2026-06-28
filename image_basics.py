import cv2
import numpy as np

# Load and resize (always first)
image = cv2.imread(r"C:\Users\LENOVO\OneDrive\Desktop\IDT project 02\mango.jpg.jpeg")
image = cv2.resize(image, (224, 224))

# ── STEP 1: Gaussian Blur ──────────────────────────────
blurred = cv2.GaussianBlur(image, (5, 5), 0)
# (5,5) is the kernel size — how many pixels to average together
# larger kernel = more blur. Always use odd numbers: 3,5,7...

print("Original pixel at (112,112):", image[112, 112])
print("Blurred pixel at (112,112): ", blurred[112, 112])
# Values will be slightly different — noise smoothed out

# ── STEP 2: Convert to HSV then threshold ──────────────
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

# Define the colour range we want to KEEP (the fruit)
# We want to isolate non-white areas (anything that isn't background)
# White in HSV has low Saturation — so we keep pixels with S > 30
lower = np.array([0,   30,  30])   # min H, S, V
upper = np.array([179, 255, 255])  # max H, S, V

mask = cv2.inRange(hsv, lower, upper)
# mask is a 2D array — 255 where fruit is, 0 where background is

print("\nMask shape:", mask.shape)   # (224, 224) — no colour channels
print("Mask unique values:", np.unique(mask))  # should be [0, 255]

# ── STEP 3: Apply mask to original image ───────────────
# This keeps only the fruit pixels, blacks out the background
fruit_only = cv2.bitwise_and(image, image, mask=mask)

# ── SHOW ALL 4 STAGES ──────────────────────────────────
cv2.imshow("1 - Original",    image)
cv2.imshow("2 - Blurred",     blurred)
cv2.imshow("3 - Mask",        mask)
cv2.imshow("4 - Fruit only",  fruit_only)
cv2.waitKey(5000)  # Wait for 5 seconds
cv2.destroyAllWindows()
