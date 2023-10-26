import imageio
import numpy as np
from math import sqrt
from math import log
import sys
import argparse
import os
import cv2
import random
from defisheye import Defisheye

def get_fish_xn_yn(source_x, source_y, radius, distortion):
    """
    Get normalized x, y pixel coordinates from the original image and return normalized
    x, y pixel coordinates in the destination fished image.
    :param distortion: Amount in which to move pixels from/to center.
    As distortion grows, pixels will be moved further from the center, and vice versa.
    """

    if 1 - distortion * (radius ** 2) == 0:
        return source_x, source_y

    return source_x / (1 - (distortion * (radius ** 2))), source_y / (1 - (distortion * (radius ** 2)))


def fish(img, distortion_coefficient):
    """
    :type img: numpy.ndarray
    :param distortion_coefficient: The amount of distortion to apply.
    :return: numpy.ndarray - the image with applied effect.
    """

    # If input image is only BW or RGB convert it to RGBA
    # So that output 'frame' can be transparent.
    w, h = img.shape[0], img.shape[1]
    if len(img.shape) == 2:
        # Duplicate the one BW channel twice to create Black and White
        # RGB image (For each pixel, the 3 channels have the same value)
        bw_channel = np.copy(img)
        img = np.dstack((img, bw_channel))
        img = np.dstack((img, bw_channel))
    if len(img.shape) == 3 and img.shape[2] == 3:
        print("RGB to RGBA")
        img = np.dstack((img, np.full((w, h), 255)))

    # prepare array for dst image
    dstimg = np.zeros_like(img)

    # floats for calcultions
    w, h = float(w), float(h)

    # easier calcultion if we traverse x, y in dst image
    for x in range(len(dstimg)):
        for y in range(len(dstimg[x])):

            # normalize x and y to be in interval of [-1, 1]
            xnd, ynd = float((2 * x - w) / w), float((2 * y - h) / h)

            # get xn and yn distance from normalized center
            rd = sqrt(xnd ** 2 + ynd ** 2)

            # new normalized pixel coordinates
            xdu, ydu = get_fish_xn_yn(xnd, ynd, rd, distortion_coefficient)

            # convert the normalized distorted xdn and ydn back to image pixels
            xu, yu = int(((xdu + 1) * w) / 2), int(((ydu + 1) * h) / 2)

            # if new pixel is in bounds copy from source pixel to destination pixel
            if 0 <= xu and xu < img.shape[0] and 0 <= yu and yu < img.shape[1]:
                dstimg[x][y] = img[xu][yu]

    return dstimg.astype(np.uint8)


def parse_args(args=sys.argv[1:]):
    """Parse arguments."""

    parser = argparse.ArgumentParser(
        description="Apply fish-eye effect to images.",
        prog='python3 fish.py')
    parser.add_argument("-s","--streakpath",help="path to rain streak")
    parser.add_argument("-i", "--imagepath", help="path to image file."
                                              " If no input is given, the supplied example 'grid.jpg' will be used.",
                        type=str, default="grid.jpg")

    parser.add_argument("-o", "--outputpath", help="file path to write output to."
                                                " format: <path>.<format(jpg,png,etc..)>",
                        type=str, default="fish.png")

    parser.add_argument("-d", "--distortion",
                        help="The distoration coefficient. How much the move pixels from/to the center."
                             " Recommended values are between -1 and 1."
                             " The bigger the distortion, the further pixels will be moved outwars from the center (fisheye)."
                             " The Smaller the distortion, the closer pixels will be move inwards toward the center (rectilinear)."
                             " For example, to reverse the fisheye effect with --distoration 0.5,"
                             " You can run with --distortion -0.3."
                             " Note that due to double processing the result will be somewhat distorted.",
                        type=float, default=0.5)

    return parser.parse_args(args)

def intesnity_addition(fisheye_image):
    height, width = fisheye_image.shape[:2]
    output_image = np.zeros((height, width, 3), dtype=np.uint8)
    center_x, center_y = width // 2, height // 2
    for y in range(height):
        for x in range(width):
            # Calculate the distance from the center
            distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

            # Calculate the intensity based on 1/r^2 distribution
            intensity = 1 / (distance ** 2) if distance > 0 else 0

            # Scale the intensity to a suitable range (0-255)
            intensity = min(intensity * 255, 255)

            # Add the intensity to the corresponding pixel in the fisheye image
            pixel = fisheye_image[y, x][:3]  # Extract only the first 3 channels
            output_image[y, x] = pixel + intensity
    return output_image
# def defish_image(intensity_added_image):
#
#     # Load the fisheye image
#     fisheye_image = intensity_added_image
#
#     # Define the camera matrix (K) and distortion coefficients (D) for the fisheye lens.
#     K = np.array([[600, 0, fisheye_image.shape[1] / 2], [0, 600, fisheye_image.shape[0] / 2], [0, 0, 1]])
#     D = np.array([0.1, 0.2, 0.0, 0.0])
#
#     # Undistort the fisheye image
#     undistorted_image = cv2.fisheye.undistortImage(fisheye_image, K, D, Knew=K)
#
#     # Display the undistorted image
#     cv2.imshow('Undistorted Image', undistorted_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     return undistorted_image
def flat_intensity_addition(image):

    # Load the flat image
    flat_image = image

    # Get image dimensions
    height, width = flat_image.shape[:2]

    # Create a blank image of the same size
    output_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Define the center of the image
    center_x, center_y = width // 2, height // 2

    # Iterate over each pixel in the output image
    for x in range(width):
        for y in range(height):
            # Calculate the distance from the center
            distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

            # Calculate the intensity based on 1/r^2 distribution
            # intensity = 1 / (distance ** 2) if distance > 0 else 0

            # Scale the intensity to a suitable range (0-255)
            # intensity = min(intensity * 255, 255)

            # Add the intensity to the corresponding pixel in the flat image
            # if distance == 0:
            #     output_image[y, x] = [255, 255, 255]  # White (or any desired color)
            # else:
            #     output_image[y, x] = flat_image[y, x] + intensity
            if distance == 0:
                output_image[y, x] = flat_image[y, x]
            else:
                intensity = 1 / (distance ** 2)
                intensity = log(distance)
                output_image[y, x] = np.clip(flat_image[y, x] + intensity, 0, 255).astype(np.uint8)
    return output_image
    # Display the output image
    cv2.imshow('Flat Image with Added Pixels', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(r"C:\Users\rosha\Downloads\flatintaddedlog.jpg",output_image)


def get_undistorted_xn_yn(fish_x, fish_y, radius, distortion):
    """
    Get normalized x, y pixel coordinates in the fisheye image and return normalized
    x, y pixel coordinates in the destination undistorted image.
    :param distortion: Amount of fisheye distortion applied.
    """
    if 1 - distortion * (radius ** 2) == 0:
        return fish_x, fish_y

    return fish_x * (1 - (distortion * (radius ** 2))), fish_y * (1 - (distortion * (radius ** 2)))

def defish(img, distortion_coefficient):
    """
    :type img: numpy.ndarray
    :param distortion_coefficient: The amount of distortion to reverse.
    :return: numpy.ndarray - the undistorted (defished) image.
    """
    # If input image is only BW or RGB, convert it to RGBA.
    w, h = img.shape[0], img.shape[1]
    if len(img.shape) == 2:
        bw_channel = np.copy(img)
        img = np.dstack((img, bw_channel))
        img = np.dstack((img, bw_channel))
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = np.dstack((img, np.full((w, h), 255)))

    # Prepare an array for the destination image.
    dstimg = np.zeros_like(img)

    # Floats for calculations.
    w, h = float(w), float(h)

    # Easier calculation if we traverse x, y in the destination image.
    for x in range(len(dstimg)):
        for y in range(len(dstimg[x])):
            # Normalize x and y to be in the interval of [-1, 1].
            xnd, ynd = float((2 * x - w) / w), float((2 * y - h) / h)

            # Get xn and yn distance from the normalized center.
            rd = sqrt(xnd ** 2 + ynd ** 2)

            # New normalized pixel coordinates for the undistorted image.
            xdu, ydu = get_undistorted_xn_yn(xnd, ynd, rd, distortion_coefficient)

            # Convert the normalized undistorted xdu and ydu back to image pixels.
            xu, yu = int(((xdu + 1) * w) / 2), int(((ydu + 1) * h) / 2)

            # If the new pixel is in bounds, copy from the source pixel to the destination pixel.
            if 0 <= xu and xu < img.shape[0] and 0 <= yu and yu < img.shape[1]:
                dstimg[x][y] = img[xu][yu]

    return dstimg.astype(np.uint8)
def streak_process(image):
    height, width, _ = image.shape
    print("inside strwak process function")
    print(f"height {height} width {width}")
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
    return output
if __name__ == "__main__":
    args = parse_args()
    try:
        imgobj = imageio.v2.imread(args.imagepath)
        streak = imageio.v2.imread(args.streakpath)
    except Exception as e:
        print(e)
        sys.exit(1)
    if os.path.exists(args.outputpath):
        ans = input(
            args.outputpath + " exists. File will be overridden. Continue? y/n: ")
        if ans.lower() != 'y':
            print("exiting")
            sys.exit(0)
    flat_intensity_addition = flat_intensity_addition(imgobj)
    height, width, _ = imgobj.shape
    dil_sharp_rain_streak = streak_process(streak)
    resized_streak = cv2.resize(dil_sharp_rain_streak, (width, height))
    cv2.imshow("rain",dil_sharp_rain_streak)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    final_img = imgobj + resized_streak
    cv2.imshow("final",final_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #print("Processing Input image to fish eye image")
    #fisheye_image = fish(imgobj, args.distortion)
    #print("Adding intensity to fish eye image")
    #intensity_added_image = intesnity_addition(fisheye_image)
    # print("Defishing the image")
    # defisheye_image = defish(intensity_added_image,args.distortion)
    # defisheye_image = Defisheye(intensity_added_image,dtype = "linear", format = "fullframe",fov = 180,pfov = 120)
    # print(type(defisheye_image))
    # defisheye_image.convert(outfile=r"C:\Users\rosha\Downloads\defish.jpg")
   # imageio.imwrite(r"C:\Users\rosha\Downloads\fisheyeimage.jpg", fisheye_image, format='png')
   #cv2.imwrite("intensityaddedfisheyeimage",intensity_added_image)
