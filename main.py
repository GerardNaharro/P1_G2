import cv2
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import convolve2d
from typing import Tuple
import glob
from skimage import io, exposure, filters
from skimage.morphology import skeletonize, opening, closing, square
from scipy.ndimage import label as ndi_label
from utils import sample_filepath


def read_img_to_grayscale(filename):
    img = cv2.imread(sample_filepath(filename), cv2.COLOR_RGB2BGR)  # Read the image.
    # Resizing the image for compatibility
    img = cv2.resize(img, (800, 450))
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def read_img(filename):
    img = cv2.imread(sample_filepath(filename), cv2.COLOR_RGB2BGR)  # Read the image.
    # Resizing the image for compatibility
    return cv2.resize(img, (800, 450))


def apply_clahe(img, clipLim=2.0, tileSize=(10, 10)):
    # CLAHE
    # clipLimit -> Threshold for contrast limiting
    clahe_filter = cv2.createCLAHE(clipLim, tileSize)
    return clahe_filter.apply(img)


def new_detection_roi(img, coordenadas_predichas, medBlurSize=5, roi_x=0, roi_y=0, roi_width=100, roi_height=100):
    # Read the image and transform it to grayscale
    original_img = read_img(img)
    cv2.imshow("ORIGINAL IMAGE", original_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    gray = read_img_to_grayscale(img)
    cv2.imshow("GRAYSCALE IMAGE", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Apply median blur
    gray = cv2.medianBlur(gray, medBlurSize)
    cv2.imshow("MEDIAN BLUR", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Get the Region Of Interest (ROI)
    roi = gray[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

    # Apply sobel in both X and Y axis
    sobelx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=1)
    sobely = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=1)

    # Transform negative values in positive values
    sobelx_abs = cv2.convertScaleAbs(sobelx)
    sobely_abs = cv2.convertScaleAbs(sobely)

    # Combine X and Y sobel images
    sobel_combined = cv2.addWeighted(sobelx_abs, 0.5, sobely_abs, 0.5, 0)

    # Binarize the image
    _, binary_image = cv2.threshold(sobel_combined, 15, 255, cv2.THRESH_BINARY)
    cv2.imshow("BINARY IMAGE", binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Opening of the image (eroding + dilating)
    kernel = np.ones((1, 1), np.uint8)
    binary_image = cv2.erode(binary_image, kernel)
    kernel = np.ones((1, 1), np.uint8)
    binary_image = cv2.dilate(binary_image, kernel)
    cv2.imshow("DILATED IMAGE", binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Find countours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        contours[i][:, :, 1] += roi_y

    # Draw countours in the image
    contour_image = original_img.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    cv2.imshow("COUNTOUR IMAGES", contour_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Filter based on the area
    min_contour_area = 10.0
    max_contour_area = 350.0
    filtered_contours = [contour for contour in contours if (
                cv2.contourArea(contour) >= min_contour_area and cv2.contourArea(contour) <= max_contour_area)]

    for contour in filtered_contours:
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        coordenadas_predichas.append([cX, cY])

    # Draw countourns
    contour_image = original_img.copy()
    cv2.drawContours(contour_image, filtered_contours, -1, (0, 255, 0), 2)

    # Show result image
    cv2.imshow('Filtered Contours', contour_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return len(filtered_contours)


# -------------------------------------------------------------------------------------------------


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
IMAGE AVERAGING :
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def create_background_model(image_paths):
    # Read and accumulate images to create a background model
    sum_image = None
    for image_path in image_paths:
        img = cv2.imread(image_path, 0)
        if img is None:
            print(f"Error: Image at {image_path} not found.")
            continue
        if sum_image is None:
            sum_image = np.zeros_like(img, dtype=np.float32)
        sum_image += img
    return (sum_image / len(image_paths)).astype(np.uint8)


def identify_people(current_image, background_model, coordenadas_predichas):
    # Subtract the background
    fg_mask = cv2.absdiff(current_image, background_model)

    # Apply threshold to get binary image
    _, thresh = cv2.threshold(fg_mask, 80, 255, cv2.THRESH_BINARY)

    # Find contours which will be the people in the image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Process contours
    people_count = 0
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Filter out small contours that are not people
            people_count += 1
            # Optionally draw bounding boxes around people
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(current_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            M = cv2.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            coordenadas_predichas.append([cX, cY])

    # Redimension for display the image
    width = int(current_image.shape[1] * 50 / 100)
    height = int(current_image.shape[0] * 50 / 100)
    fg_mask_resized = cv2.resize(fg_mask, (width, height), interpolation=cv2.INTER_AREA)
    resized_image = cv2.resize(current_image, (width, height), interpolation=cv2.INTER_AREA)
    resized_thresh = cv2.resize(thresh, (width, height), interpolation=cv2.INTER_AREA)

    # display mask
    cv2.imshow('mask', fg_mask_resized)

    # display thresholded image
    cv2.imshow('thresholding', resized_thresh)

    # Display the processed image
    cv2.imshow('Identified People', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # print(f"Number of people identified: {people_count}")
    return people_count


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
MORPHOLOGICAL THINING :
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def morphological_thining(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # Scaling the image to convert the objects bigger
    image = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray_8bit = (gray * 255).astype('uint8')  # Convertir a 8 bits, sino salta error

    # Adaptative thresholding
    thresh = cv2.adaptiveThreshold(gray_8bit, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    thinned = skeletonize(thresh // 255) * 255  # Apply skeletonize
    return thresh, image, gray, thinned


def count_people(thinned):
    labeled_array, num_features = ndi_label(thinned)
    # Adjusting object size for detection
    component_sizes = np.bincount(labeled_array.ravel())
    too_small = component_sizes < 10
    too_large = component_sizes > 50
    too_small_or_large = too_small | too_large
    mask_sizes = too_small_or_large[labeled_array]
    labeled_array[mask_sizes] = 0
    labeled_array, final_count = ndi_label(labeled_array > 0)
    return labeled_array, final_count


def display(image_path):
    _, og_image, gray, thinned = morphological_thining(image_path)
    labeled_array, num_people = count_people(thinned)

    print(f"Estimated number of people: {num_people}")
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    ax = axes.ravel()

    ax[0].imshow(cv2.cvtColor(og_image, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(gray, cmap=plt.cm.gray)
    ax[1].set_title('Gray Image')
    ax[1].axis('off')

    ax[2].imshow(thinned.astype(np.uint8), cmap=plt.cm.gray)
    ax[2].set_title('Thinned Image')
    ax[2].axis('off')

    ax[3].imshow(labeled_array.astype(np.uint8), cmap=plt.cm.nipy_spectral)
    ax[3].set_title('Labeled Image')
    ax[3].axis('off')

    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------------------------------------


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Folder with the images
    carpeta = "samples"

    roi_x = 0  # Upper left X point coordinate
    roi_y = 180  # Upper left Y point coordinate
    roi_width = 800  # Width
    roi_height = 380  # Height

    # CSV file path
    archivo_csv = 'data\datos.csv'

    # Read CSV file with ground truths in a pandas dataframe
    df = pd.read_csv(archivo_csv, header=None, delimiter=';')

    predichas = []
    reales = []

    factor_x = 800 / 1980
    factor_y = 480 / 1080
    umbral = 25
    tp = 0
    fp = 0
    fn = 0
    accuracy = 0
    n_images = 0


    # Loop throught all images in the folder
    for archivo in os.listdir(carpeta):
        n_images += 1
        coordenadas_predichas = []
        coordenadas_reales = []
        num = new_detection_roi(archivo, coordenadas_predichas, 7, roi_x, roi_y, roi_width, roi_height)
        print(f'Number of people detected in the image {archivo}: {num}')
        predichas.append(num)
        indices = df.index[df.iloc[:, 3] == archivo].tolist()
        sum = len(indices)
        print(f'Number of people MANUALLY detected in the image {archivo}: {sum}')
        reales.append((sum))


        coordenadas_reales = df.loc[indices , [1 , 2]].transpose()
        coordenadas_reales.reset_index(drop=True, inplace=True)
        coordenadas_reales = coordenadas_reales.apply(lambda col: col * [factor_x, factor_y]).round().astype(int)

        for i in range(len(coordenadas_predichas)):
            coordenadas_reales.columns = range(len(coordenadas_reales.columns))
            for j in range(coordenadas_reales.shape[1]):
                if coordenadas_predichas[i][0] - coordenadas_reales[j][0] <= umbral and coordenadas_predichas[i][1] - coordenadas_reales[j][1] <= umbral:
                    #found tp
                    tp += 1
                    coordenadas_reales = coordenadas_reales.drop(coordenadas_reales.columns[j], axis=1)

                    break

        fp = len(coordenadas_predichas) - tp
        fn = coordenadas_reales.shape[1]
        accuracy += (tp / (tp + fp + fn))
        print(f"Image {archivo} accuracy = {(tp / (tp + fp + fn))}")
        tp = 0


    # MSE calculation for validation
    mse = 0
    for i in range(len(predichas)):
        mse += (predichas[i] - reales[i]) ** 2
    mse = mse/len(predichas)
    print(f'Mean Squared Error: {mse}')


    # Accuracy calculation for validation
    print(f"Average accuracy: {accuracy / n_images}")



    # -------------------------------------------------------------------------------------------------
    print("-------------------------------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------------------------------")

    # Create a background model from a set of images
    background_model = create_background_model(glob.glob("samples/*.jpg"))

    # display(image_path)

    predichas = []
    reales = []

    factor_x = 800 / 1980
    factor_y = 480 / 1080
    umbral = 25
    tp = 0
    fp = 0
    fn = 0
    accuracy = 0
    n_images = 0

    carpeta = "samples"

    # Loop throught all images in the folder
    for archivo in os.listdir(carpeta):
        n_images += 1
        coordenadas_predichas = []
        coordenadas_reales = []
        current_image = cv2.imread("samples/" + archivo, 0)
        num = identify_people(current_image, background_model, coordenadas_predichas)

        print(f'Number of people detected in the image {archivo}: {num}')
        predichas.append(num)
        indices = df.index[df.iloc[:, 3] == archivo].tolist()
        sum = len(indices)
        print(f'Number of people MANUALLY detected in the image {archivo}: {sum}')
        reales.append((sum))

        coordenadas_reales = df.loc[indices, [1, 2]].transpose()
        coordenadas_reales.reset_index(drop=True, inplace=True)
        # coordenadas_reales = coordenadas_reales.apply(lambda col: col * [factor_x, factor_y]).round().astype(int)

        for i in range(len(coordenadas_predichas)):
            coordenadas_reales.columns = range(len(coordenadas_reales.columns))
            for j in range(coordenadas_reales.shape[1]):
                if coordenadas_predichas[i][0] - coordenadas_reales[j][0] <= umbral and coordenadas_predichas[i][1] - \
                        coordenadas_reales[j][1] <= umbral:
                    # found tp
                    tp += 1
                    coordenadas_reales = coordenadas_reales.drop(coordenadas_reales.columns[j], axis=1)

                    break

        fp = len(coordenadas_predichas) - tp
        fn = coordenadas_reales.shape[1]
        accuracy += (tp / (tp + fp + fn))
        print(f"Image {archivo} accuracy = {(tp / (tp + fp + fn))}")
        tp = 0

    # MSE calculation for validation
    mse = 0
    for i in range(len(predichas)):
        mse += (predichas[i] - reales[i]) ** 2
    mse = mse / len(predichas)
    print(f'Mean Squared Error: {mse}')

    # Accuracy calculation for validation
    print(f"Average accuracy: {accuracy / n_images}")





















