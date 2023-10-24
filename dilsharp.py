# import cv2
# import numpy as np
# import random
#
# # Load the image
# image = cv2.imread(r"C:\Users\rosha\Downloads\rainmap-crop (1).jpg")
#
# # Get the height and width of the image
# height, width, _ = image.shape
#
# # Create an empty output image
# output = np.zeros_like(image)
# # Define the kernel for dilation
# dilation_kernel = np.array([[0, 1, 0],
#                             [1, 1, 1],
#                             [0, 1, 0]], dtype=np.uint8)
# print("begining of for loop")
# # Iterate over the image pixels from left to right
# for y in range(height):
#     for x in range(width):
#         # Generate random sharpening weight between 0.5 and 2
#         sharpen_weight = random.uniform(0.5, 2)
#
#         # Extract the region of interest (3x3) centered at the current pixel
#         roi = image[y - 1:y + 2, x - 1:x + 2]
#
#         # Check for out-of-bounds indices
#         if roi.shape[0] == 3 and roi.shape[1] == 3:
#             # Apply dilation to the ROI using the kernel
#             dilated_roi = cv2.dilate(roi, dilation_kernel)
#
#             # Calculate the sharpened pixel value for the center pixel
#             sharpened_pixel = (roi[1, 1].astype(float) - dilated_roi[1, 1].astype(float)) * sharpen_weight + \
#                               dilated_roi[1, 1].astype(float)
#
#             # Ensure pixel values are in the valid 0-255 range
#             sharpened_pixel = np.clip(sharpened_pixel, 0, 255).astype(np.uint8)
#
#             # Set the sharpened pixel in the output image
#             output[y, x] = sharpened_pixel
#
# # Display the output image
# cv2.imshow('Dilation and Sharpening', output)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imwrite(r"C:\Users\rosha\Downloads\dilate_sharpened.jpg",output)
#
import cv2
import numpy as np
import random

# Load the image
image = cv2.imread(r"C:\Users\rosha\Downloads\rainmap-crop (1).jpg")

# Get the height and width of the image
height, width, _ = image.shape

# Create an empty output image
output = np.zeros_like(image)

# Define the kernel for dilation
dilation_kernel = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]], dtype=np.uint8)

# Iterate over the image pixels from left to right
for y in range(height):
    for x in range(width):
        # Generate random sharpening weight between 0.5 and 2
        # Decrease the sharpening weight as you move down the image
        sharpen_weight = random.uniform(0.5, 2) * (1 - y / height)

        # Extract the region of interest (3x3) centered at the current pixel
        roi = image[y - 1:y + 2, x - 1:x + 2]

        # Check for out-of-bounds indices
        if roi.shape[0] == 3 and roi.shape[1] == 3:
            # Apply dilation to the ROI using the kernel
            dilated_roi = cv2.dilate(roi, dilation_kernel)

            # Calculate the sharpened pixel value for the center pixel
            sharpened_pixel = (roi[1, 1].astype(float) - dilated_roi[1, 1].astype(float)) * sharpen_weight + \
                              dilated_roi[1, 1].astype(float)

            # Ensure pixel values are in the valid 0-255 range
            sharpened_pixel = np.clip(sharpened_pixel, 0, 255).astype(np.uint8)

            # Set the sharpened pixel in the output image
            output[y, x] = sharpened_pixel

# Display the output image
cv2.imshow('Dilation and Sharpening', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(r"C:\Users\rosha\Downloads\dilsharpup2down.jpg",output)
