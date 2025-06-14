import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import easyocr
import dask.array as da
import dask.dataframe as dd
import time
import cv2
from PIL import Image

def read_labels_from_txt(file_path):
    """
    Reads a .txt file and creates a DataFrame with the data.
    
    Each line in the .txt file should contain values separated by spaces.
    The resulting DataFrame will have columns named 'file_name' and 'label'.
    
    Parameters:
    file_path (str): The path to the .txt file to read.
    
    Returns:
    pd.DataFrame: A DataFrame containing the data from the .txt file.
    """
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(line.strip().split(' '))
    df = pd.DataFrame(data)
    
    # Rename the columns
    df.columns = ['file_name', 'label']
    
    return df


def get_images_pixels(images_path):
    """
    Converts a pandas Series of image paths to a numpy array of image pixel arrays.
    
    Parameters:
    images_path (pd.Series): A pandas Series containing the paths to the images.
    
    Returns:
    np.array: A numpy array where each element is an array of image pixels.
    """
    start_time = time.time()
    # Read the first image and initialize the list of arrays
    first_image = cv2.imread("data/images/"+images_path.iloc[0], cv2.IMREAD_GRAYSCALE)
    print("First image shape: ", first_image.shape)
    dsize = (350, 500)
    first_image = cv2.resize(first_image, dsize)
    arrays = [first_image]
    # Read remaining images and append to the list
    for path in images_path.iloc[1:]:
        image = cv2.imread("data/images/"+path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, dsize)
        arrays.append(image)
    
    return np.array(arrays)

def get_images_pixels_dask(images_path):
    """
    Converts a pandas Series of image paths to a dask array of image pixel arrays.
    
    Parameters:
    images_path (pd.Series): A pandas Series containing the paths to the images.
    
    Returns:
    dask.array: A dask array where each element is an array of image pixels.
    """
    start_time = time.time()
    # Create empty dask array for first image
    first_image = cv2.imread("data/images/"+images_path.iloc[0], cv2.IMREAD_GRAYSCALE)
    dsize = (500, 380)
    first_image = cv2.resize(first_image, dsize)
    arrays = da.from_array(first_image, chunks='auto')
    arrays = arrays.reshape(1, *arrays.shape)

    # Read remaining images and concatenate to dask array
    for path in images_path.iloc[1:]:
        image = cv2.imread("data/images/"+path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, dsize)
        dask_array = da.from_array(image, chunks='auto')
        dask_array = dask_array.reshape(1, *dask_array.shape)
        arrays = da.concatenate([arrays, dask_array], axis=0)

    return arrays

def add_pixel_array_to_df_dask(df, only_empty=True, writing_freq=0, parquet_name="data/df.parFquet.gzip", images_path='data/images/'):
    """
    Memory-optimized version using dask DataFrame to add pixel arrays to DataFrame.
        
    Parameters:
    df (pd.DataFrame): Input DataFrame with 'file_name' column
    only_empty (bool): Process only rows without pixel arrays if True
    writing_freq (int): Save frequency (0 for no saves)
    parquet_name (str): Output parquet file path
    images_path (str): Images directory path
        
    Returns:
    dask.dataframe: DataFrame with added pixel arrays
    """
    start_time = time.time()
        
    # Convert to dask DataFrame
    ddf = dd.from_pandas(df, npartitions=20)
        
    if 'pixel_array' not in ddf.columns:
        ddf['pixel_array'] = None
            
        def process_image(row):
            image = cv2.imread(images_path + row['file_name'], cv2.IMREAD_GRAYSCALE)
            row['pixel_array'] = str(image.tolist())
            return row
            
        ddf = ddf.apply(process_image, axis=1, meta=ddf)
            
        print("Time to calculate :", time.time() - start_time)

        start_time= time.time()
        ddf.compute()
        print("Time to compute :", time.time() - start_time)

        start_time= time.time()
        ddf.to_parquet('df.igOne.parquet',
            write_metadata_file=True,
            )
        print("Total time to parquet: ", time.time() - start_time)

def add_image_shape_to_df(df, only_empty=True, writing_freq=0, parquet_name="data/df.parquet.gzip", images_path='data/images/'):
    """
    """
    start_time = time.time()

    if 'width' not in df.columns:
        df['width'] = [None for _ in range(len(df))]

    if 'height' not in df.columns:
        df['height'] = [None for _ in range(len(df))]
    

    iteration_count = 0
    for index, row in df.iterrows():
        # if only_empty is True, only process the rows with empty pixel arrays
        if only_empty and row['width'] is not None:
            continue

        # read the image and convert it to a list of pixel values
        image = cv2.imread(images_path + row['file_name'], cv2.IMREAD_GRAYSCALE)
        (h, w) = image.shape[:2]
        # update the DataFrame with the pixel shape
        df.at[index, 'width'] = w
        df.at[index, 'height'] = h

        # Free up memory by deleting the image variable
        del image

        iteration_count += 1
        if writing_freq and iteration_count % writing_freq == 0:
            df.to_parquet(parquet_name, compression='gzip')
            print("On iteration ", iteration_count)
            print("Time elapsed: ", time.time() - start_time)
            print('-----------------------------------\n')

    if writing_freq:
        df.to_parquet(parquet_name, compression='gzip')

    print("Total time elapsed: ", time.time() - start_time)
    print('-----------------------------------\n')

    return df

def get_images_pixels(images_path):
    """
    Converts a pandas Series of image paths to a numpy array of image pixel arrays.
    
    Parameters:
    images_path (pd.Series): A pandas Series containing the paths to the images.
    
    Returns:
    np.array: A numpy array where each element is an array of image pixels.
    """
    start_time = time.time()
    # Read the first image and initialize the list of arrays
    first_image = cv2.imread("data/images/"+images_path.iloc[0], cv2.IMREAD_GRAYSCALE)
    print("First image shape: ", first_image.shape)
    dsize = (380, 500)
    first_image = cv2.resize(first_image, dsize)
    arrays = [first_image]
    # Read remaining images and append to the list
    for path in images_path.iloc[1:]:
        image = cv2.imread("data/images/"+path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, dsize)
        arrays.append(image)
    
    return np.array(arrays)

def get_images_pixels_dask(images_path, writing_freq=0):
    """
    Converts a pandas Series of image paths to a dask array of image pixel arrays.
    
    Parameters:
    images_path (pd.Series): A pandas Series containing the paths to the images.
    
    Returns:
    dask.array: A dask array where each element is an array of image pixels.
    """
    start_time = time.time()
    # Create empty dask array for first image
    first_image = cv2.imread("data/images/"+images_path.iloc[0], cv2.IMREAD_GRAYSCALE)
    dsize = (500, 350)
    first_image = cv2.resize(first_image, dsize)
    arrays = da.from_array(first_image, chunks='auto')
    arrays = arrays.reshape(1, *arrays.shape)
    iteration_count = 0

    # Read remaining images and concatenate to dask array
    for path in images_path.iloc[1:]:
        image = cv2.imread("data/images/"+path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, dsize)
        dask_array = da.from_array(image, chunks='auto')
        dask_array = dask_array.reshape(1, *dask_array.shape)
        arrays = da.concatenate([arrays, dask_array], axis=0)
        iteration_count += 1
        if writing_freq and iteration_count % writing_freq == 0:
            #df.to_parquet(parquet_name, compression='gzip')
            print("On iteration ", iteration_count)
            print("Time elapsed: ", time.time() - start_time)
            print('-----------------------------------\n')
            return arrays
    
    return arrays

def add_pixel_array_to_df_dask(df, only_empty=True, writing_freq=0, parquet_name="data/df.parFquet.gzip", images_path='data/images/'):
    """
    Memory-optimized version using dask DataFrame to add pixel arrays to DataFrame.
        
    Parameters:
    df (pd.DataFrame): Input DataFrame with 'file_name' column
    only_empty (bool): Process only rows without pixel arrays if True
    writing_freq (int): Save frequency (0 for no saves)
    parquet_name (str): Output parquet file path
    images_path (str): Images directory path
        
    Returns:
    dask.dataframe: DataFrame with added pixel arrays
    """
    start_time = time.time()
        
    # Convert to dask DataFrame
    ddf = dd.from_pandas(df, npartitions=20)
        
    if 'pixel_array' not in ddf.columns:
        ddf['pixel_array'] = None
            
        def process_image(row):
            image = cv2.imread(images_path + row['file_name'], cv2.IMREAD_GRAYSCALE)
            row['pixel_array'] = str(image.tolist())
            return row
            
        ddf = ddf.apply(process_image, axis=1, meta=ddf)
            
        print("Time to calculate :", time.time() - start_time)

        start_time= time.time()
        ddf.compute()
        print("Time to compute :", time.time() - start_time)

        start_time= time.time()
        ddf.to_parquet('df.igOne.parquet',
            write_metadata_file=True,
            )
        print("Total time to parquet: ", time.time() - start_time)

def add_pixel_array_to_df(df, only_empty=True, writing_freq=0, parquet_name="data/df.parquet.gzip", images_path='data/images/'):
    """
    Adds the pixel array of images to a DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to which the pixel arrays will be added. 
                       It should contain a 'file_name' column with the path to the images.

    only_empty (bool): If True, only process rows with empty 'pixel_array' values.

    writing_freq (int): The frequency at which the DataFrame will be written to a .csv file (number of images).
                        If 0, the DataFrame will not be written to a file.
    
    parquet_name (str): The name of the parquet file to write the DataFrame to. Only used if writing_freq > 0.

    images_path (str): The path to the folder containing the images.

    Returns:
    pd.DataFrame: The DataFrame with the pixel arrays added.
    """
    start_time = time.time()
    # if the column does not exist, create it with NaN values
    if 'pixel_array' not in df.columns:
        df['pixel_array'] = [[] for _ in range(len(df))]

    iteration_count = 0
    for index, row in df.iterrows():
        # if only_empty is True, only process the rows with empty pixel arrays
        if only_empty and len(row['pixel_array']) > 0:
            continue

        # read the image and convert it to a list of pixel values
        image = cv2.imread(images_path + row['file_name'], cv2.IMREAD_GRAYSCALE)
        
        # update the DataFrame with the pixel shape
        df.at[index, 'pixel_array'] = image.tolist()

        # Free up memory by deleting the image variable
        del image

        iteration_count += 1
        if writing_freq and iteration_count % writing_freq == 0:
            df.to_parquet(parquet_name, compression='gzip')
            print("On iteration ", iteration_count)
            print("Time elapsed: ", time.time() - start_time)
            print('-----------------------------------\n')

    if writing_freq:
        df.to_parquet(parquet_name, compression='gzip')

    print("Total time elapsed: ", time.time() - start_time)
    print('-----------------------------------\n')

    return df

def add_easyocr_text_to_df(df, only_empty=True, writing_freq=0, parquet_name="data/df.parquet.gzip", image_path='data/images/'):
    """
    Adds text extracted by EasyOCR to a DataFrame.
    This function uses GPU acceleration to speed up the process.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to which the EasyOCR text will be added. 
                       It should contain a 'file_name' column with the path to the images.

    only_empty (bool): If True, only process rows with empty 'easyocr_text' values.

    image_path (str): The path to the folder containing the images.

    writing_freq (int): The frequency at which the DataFrame will be written to a .csv file (number of images).
                        If 0, the DataFrame will not be written to a file.
    
    parquet_name (str): The name of the parquet file to write the DataFrame to. Only used if writing_freq > 0.

    Returns:
    pd.DataFrame: The DataFrame with the EasyOCR text added.
    """
    start_time = time.time()
    #if the column does not exist, create it with empty strings
    if 'easyocr_text' not in df.columns:
        df['easyocr_text'] = df.apply(lambda x: [], axis=1)

    #we instantiate the easyocr reader
    reader = easyocr.Reader(['en'], gpu=True) 

    #we iterate over the rows of the dataframe
    iteration_count = 0
    for index, row in df.iterrows() :
        #if only_empty is True, only process the rows with empty strings
        #this is useful if we want to continue processing a dataframe that was already processed
        #and we want to avoid processing the same rows again (for example, if the process was interrupted)
        
        if only_empty and len(row['easyocr_text']) > 0:
            continue
        
        result = reader.readtext(image_path+row['file_name'], detail=0)

        #we clear the list and add the new values (useful if we want to reprocess the same dataframe)
        row['easyocr_text'].clear()
        row['easyocr_text'].extend(result)

        iteration_count += 1
        if writing_freq and iteration_count % writing_freq == 0:
            df.to_parquet(parquet_name,compression='gzip')
            print("On iteration ", iteration_count)
            print("Time elapsed: ", time.time() - start_time)
            print('-----------------------------------\n')           

    
    if writing_freq :
            df.to_parquet(parquet_name,compression='gzip')

    print("Total time elapsed: ", time.time() - start_time)
    print('-----------------------------------\n')           


    return df

def detect_faces_open_cv(image_file_path):
    """
    Detects faces in an image using OpenCV's pre-trained Haar Cascade classifier.
    
    Parameters:
    image_path (str): The path to the image file.
    
    Returns:
    bool: True if a face is detected, False otherwise.
    """

    # Load the pre-trained Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Read the image
    image = cv2.imread(image_file_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)
    
    # Return True if at least one face is detected, otherwise False
    return len(faces) > 0
#we add a label attribute to the function so we can use it as a column name in the dataframe
detect_faces_open_cv.label='open_cv_face'


def add_face_detection_to_df(df, face_detection_func, only_empty=True, writing_freq=0, parquet_name="data/df.parquet.gzip", image_path='data/images/'):
    """
    Adds a column to the DataFrame indicating whether a face is detected in the image using the specified fade 
    detection function.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to which the face detection results will be added.
                       It should contain a 'file_name' column with the path to the images.

    face_detection_func (function): The function to use for face detection.

    only_empty (bool): If True, only process rows with empty 'face_cv' values.

    writing_freq (int): The frequency at which the DataFrame will be written to a .csv file (number of images). 
                        If 0, the DataFrame will not be written to a file.

    parquet_name (str): The name of the parquet file to write the DataFrame to. Only used if writing_freq > 0.
    
    image_path (str): The path to the folder containing the images.
    
    Returns:
    pd.DataFrame: The DataFrame with the face detection results added.
    """
    #if the column does not exist, create it with empty None values
    col_label=face_detection_func.label
    if col_label not in df.columns:
        df[col_label] = None

    iteration_count = 0
    start_time = time.time()

    for index, row in df.iterrows() :

        if only_empty and row[col_label] is not None:
            continue

        df.at[index, col_label] = detect_faces_open_cv(image_path + row['file_name'])

        iteration_count += 1
        if writing_freq and iteration_count % writing_freq == 0:
            df.to_parquet(parquet_name,compression='gzip')
            print("On iteration ", iteration_count)
            print("Time elapsed: ", time.time() - start_time)
            print('-----------------------------------\n')

    if writing_freq :
            df.to_parquet(parquet_name,compression='gzip')

    print("Total time elapsed: ", time.time() - start_time)
    print('-----------------------------------\n') 

    return df

def process_images_by_label(df):
    """
    Iterates over a range from 0 to 12 and calls the get_images_pixels function
    using as parameter only the rows where the 'label' column is equal to the current iteration.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the image file names and labels.
    """
    labels = ["letter", "form", "email", "handwritten", "advertisement", "scientific report", "scientific publication", "specification", "file folder", "news article", "budget", "invoice", "présentation", "questionnaire", "resume", "memo"]

    for i in labels:
        label_df = df[df['label'] == str(i)]
        if not label_df.empty:
            print(f"Processing label {i} with {len(label_df)} images.")
            images_pixels = get_images_pixels_dask(label_df['file_name'])
            print(f"Processed label {i} with shape: {images_pixels.shape}")
            np.savez_compressed(f"data/processed_ifiltered_images_label_{i}.npz", images_pixels)

def create_tf_directory(df, data_dir='data/images'):
    """
    Creates a directory structure for TensorFlow data.
    """
    if not os.path.exists('data/tf_data'):
        os.makedirs('data/tf_data')

    for i in range(16):
        if not os.path.exists(f'data/tf_data/{i}'):
            os.makedirs(f'data/tf_data/{i}')
    
    for idx, row in df.iterrows():
            label_dir = f"data/tf_data/{row['label']}"
            # Create symlink to original image
            src = f"../../../{data_dir}/{row['file_name']}"
            print(src)
            only_file_name = row['file_name'].split('/')[-1]
            only_file_name = only_file_name.replace('.tif', '.png')
            dst = f"{label_dir}/{only_file_name}"
            if not os.path.exists(dst):
                os.symlink(src, dst)

def create_tf_directory_and_format(df, data_dir='data/images'):
    """
    Creates a directory structure for TensorFlow data.
    """
    if not os.path.exists('data/tf_data'):
        os.makedirs('data/tf_data')

    for i in range(16):
        if not os.path.exists(f'data/tf_data/{i}'):
            os.makedirs(f'data/tf_data/{i}')
    
    for idx, row in df.iterrows():
            label_dir = f"data/tf_data/{row['label']}"
            # Create symlink to original image
            src = f"{data_dir}/{row['file_name']}"
            only_file_name = row['file_name'].split('/')[-1]
            only_file_name = only_file_name.replace('.tif', '.png')
            dst = f"{label_dir}/{only_file_name}"
            try:
                img = Image.open(src)
                img.save(dst, "png")
            except:
                print(f"Error processing {src}")
                continue

def plot_training_history(model_files):
    """
    Plots the training history for each model.
    
    Parameters:
    model_files (dict): A dictionary where the keys are the model names and the values are the file paths.
    """
    # Read the CSV files for each model
    data_frames = {model: pd.read_csv(file) for model, file in model_files.items()}
    columns_to_plot = ['accuracy', 'loss', 'val_accuracy', 'val_loss']

    # Plot each column for all models
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i, col in enumerate(columns_to_plot):
        ax = axes[i]
        for model, df in data_frames.items():
            if col in df.columns:
                ax.plot(df[col], label=model)
        ax.set_title(f'{col.capitalize()} Comparison')
        ax.set_xlabel('Epochs')
        ax.set_ylabel(col.capitalize())
        ax.legend()
        ax.grid()

    plt.tight_layout()
    plt.show()