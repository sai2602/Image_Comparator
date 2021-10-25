import cv2
import imutils
from os import listdir, remove
from os.path import join, exists
from time import time, strftime, gmtime

CONTOUR_AREA = 400.0
THRESHOLD_AREA = 2000.0

# This value has to be changed to the specific root directory where the images are saved
data_path = "C:\\Users\\saipa\\Desktop\\Dataset\\c23"


def draw_color_mask(img, borders, color=(0, 0, 0)):
    h = img.shape[0]
    w = img.shape[1]

    x_min = int(borders[0] * w / 100)
    x_max = w - int(borders[2] * w / 100)
    y_min = int(borders[1] * h / 100)
    y_max = h - int(borders[3] * h / 100)

    img = cv2.rectangle(img, (0, 0), (x_min, h), color, -1)
    img = cv2.rectangle(img, (0, 0), (w, y_min), color, -1)
    img = cv2.rectangle(img, (x_max, 0), (w, h), color, -1)
    img = cv2.rectangle(img, (0, y_max), (w, h), color, -1)

    return img


def preprocess_image_change_detection(img, gaussian_blur_radius_list=None, black_mask=(5, 10, 5, 0)):
    gray = img.copy()
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if gaussian_blur_radius_list is not None:
        for radius in gaussian_blur_radius_list:
            gray = cv2.GaussianBlur(gray, (radius, radius), 0)

    gray = draw_color_mask(gray, black_mask)

    return gray


def compare_frames_change_detection(prev_frame, next_frame, min_contour_area):
    frame_delta = cv2.absdiff(prev_frame, next_frame)
    thresh = cv2.threshold(frame_delta, 45, 255, cv2.THRESH_BINARY)[1]

    thresh = cv2.dilate(thresh, None, iterations=2)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(cnts)

    score = 0
    res_cnts = []
    for c in cnts:
        if cv2.contourArea(c) < min_contour_area:
            continue

        res_cnts.append(c)
        score += cv2.contourArea(c)

    return score, res_cnts, thresh


if __name__ == "__main__":
    start_time = time()

    # Check for validity of the directory
    image_list = []
    if exists(data_path):
        image_list = listdir(data_path)
        print("Valid directory provided!!!\n")
    else:
        print("Invalid Directory")
        raise AssertionError("Invalid search path provided!!!\n")

    # Make sure the directory is not empty
    assert len(image_list) != 0, "No images found in directory!!!\n"

    ###########################################################################################################
    # ######################################### DATA PRE-PROCESSING ######################################### #
    ###########################################################################################################

    # Load all the images from the directory and apply the necessary pre-processing steps and save to a list

    print("Working on data pre-processing\n")
    processed_images = []
    for each_file in image_list:
        image_path = join(data_path, each_file)
        raw_image = cv2.imread(image_path)
        processed_image = preprocess_image_change_detection(raw_image)
        processed_images.append(processed_image)

    print("Images prepared and ready to use\n")

    ###########################################################################################################
    # ########################################### IMAGE COMPARATOR ########################################## #
    ###########################################################################################################

    # Take an image from the list and compare it to the remaining images in the list
    # Indices of images similar to the current reference image are accumulated in a list
    # The next valid image is then chosen as the reference image and the process is repeated

    print("Performing image comparison with the pre-processed images\n")

    similar_image_list = []

    if len(processed_images) > 1:
        reference_image_index = 0
        last_image_index = len(processed_images) - 1
        averaged_contour_area = 0

        while reference_image_index != last_image_index:

            if reference_image_index not in similar_image_list:
                for current_image_index in range(reference_image_index+1, last_image_index+1):
                    if current_image_index not in similar_image_list:
                        total_area, contour_count, dilated_contour_map = \
                            compare_frames_change_detection(processed_images[reference_image_index],
                                                            processed_images[current_image_index], CONTOUR_AREA)

                        averaged_contour_area = 0 if len(contour_count) == 0 else total_area/len(contour_count)
                        if averaged_contour_area <= THRESHOLD_AREA:
                            similar_image_list.append(current_image_index)

            reference_image_index += 1

    else:
        print("Not many images to compare!!!\n")

    ###########################################################################################################
    # ######################################### DATA POST_PROCESSING ######################################## #
    ###########################################################################################################

    # IMAGE COMPARATOR section created a list of indices signifying images that look similar
    # This list is looped through and corresponding images are deleted

    print("Deleting duplicate images")
    for delete_image_index in similar_image_list:
        delete_image = join(data_path, image_list[delete_image_index])
        if exists(delete_image):
            remove(delete_image)

    print(strftime("%H:%M:%S", gmtime(time() - start_time)))
