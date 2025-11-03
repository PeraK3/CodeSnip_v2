import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import skimage
import cv2
import joblib


from skimage import io, color, exposure, segmentation, measure
from skimage.filters import threshold_otsu, sobel
from skimage.color import label2rgb
from scipy import ndimage as ndi
from scipy import io
from statistics import mode, StatisticsError
from PIL import Image



# gelgeniesetup.py

def make_histogram_V4(image_path): # Alter intensities WORKS!!!!
    """
    Load image, normalize to 0–1, and plot histogram with rescaled intensities
    """
    # Read the image from file
    image = io.imread(image_path)

    # Convert to grayscale if RGB
    if image.ndim == 3:
        image = color.rgb2gray(image)

    # Rescale intensities to [0, 1]
    image_rescaled = exposure.rescale_intensity(image, in_range='image', out_range=(0, 1))

    # Compute histogram
    hist, bins = np.histogram(image_rescaled, bins=256, range=(0, 1))

    # Plot image + histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    ax1.imshow(image_rescaled, cmap=plt.cm.gray, interpolation='nearest')
    ax1.axis('off')
    ax2.plot(bins[:-1], hist, lw=2)
    ax2.set_title('histogram of grey values (rescaled 0–1)')
    plt.show()

    return hist

from skimage import io, color, exposure
import numpy as np
import matplotlib.pyplot as plt

def make_histogram_V5(image_input):
    """
    Load image (from file path or numpy array), normalize to 0–1,
    and plot histogram with rescaled intensities.
    """

    # Handle both file path and numpy array
    if isinstance(image_input, str):
        image = io.imread(image_input)
    elif isinstance(image_input, np.ndarray):
        image = image_input
    else:
        raise TypeError("Expected file path (str) or numpy array")

    # Convert to grayscale if RGB
    if image.ndim == 3:
        image = color.rgb2gray(image)

    # Rescale intensities to [0, 1]
    image_rescaled = exposure.rescale_intensity(image, in_range='image', out_range=(0, 1))

    # Compute histogram
    hist, bins = np.histogram(image_rescaled, bins=256, range=(0, 1))

    # Plot image + histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    ax1.imshow(image_rescaled, cmap=plt.cm.gray, interpolation='nearest')
    ax1.axis('off')
    ax2.plot(bins[:-1], hist, lw=2)
    ax2.set_title('histogram of grey values (rescaled 0–1)')
    plt.show()

    return hist


def auto_thresholds(image, method="otsu"):
    """
    Automatically determine background and foreground thresholds.
    
    Parameters
    ----------
    image : ndarray
        Grayscale gel image.
    method : str
        'otsu' (default) or 'percentile'
        
    Returns
    -------
    sure_bg : float
        Background threshold
    sure_fg : float
        Foreground threshold
    """
    if method == "otsu":
        t = threshold_otsu(image)
        sure_bg = t * 0.5   # looser background
        sure_fg = t * 1.2   # stricter foreground
    elif method == "percentile":
        sure_bg = np.percentile(image, 10)
        sure_fg = np.percentile(image, 95)
    else:
        raise ValueError("Unknown method: choose 'otsu' or 'percentile'")
    
    return sure_bg, sure_fg

def w_seg_f_b(image, sure_fg, sure_bg, verbose=False):
    elevation_map = sobel(image)

    markers = np.zeros_like(image, dtype=np.int32)
    markers[image < sure_bg] = 1
    markers[image > sure_fg] = 2

    segmentation_result = segmentation.watershed(elevation_map, markers)
    segmentation_result = ndi.binary_fill_holes(segmentation_result - 1)

    labeled_bands, _ = ndi.label(segmentation_result)

    props = measure.regionprops(labeled_bands, intensity_image=image)
    props_table = measure.regionprops_table(
        labeled_bands,
        intensity_image=image,
        properties=("label", "bbox", "area", "mean_intensity", "max_intensity", "centroid")
    )

    if verbose:
        overlay = color.label2rgb(labeled_bands, image=image, bg_label=0)
        plt.figure()
        plt.title("Region-based Segmentation")
        plt.imshow(overlay)
        plt.show()

        print(f"✅ Found {len(props)} bands")
        for p in props:
            print(f"Band {p.label}: area={p.area}, bbox={p.bbox}, mean_intensity={p.mean_intensity:.2f}")

    return labeled_bands, props, props_table

def generate_csv_V4(labels, img):
    props_table = measure.regionprops_table(
        labels,
        intensity_image=img,
        properties=("label", "centroid", "bbox", "area", "mean_intensity", "centroid"),
    )
    df = pd.DataFrame(props_table)
    return df

def analyze_image(img_raw, verbose=True):
    make_histogram_V4(img_raw)  # img_raw is now a full path string

    img = io.imread(img_raw, as_gray=True)  # loads properly now

    sure_bg, sure_fg = auto_thresholds(img, method="otsu")
    labels, props, props_table = w_seg_f_b(img, sure_fg, sure_bg, verbose)

    print(f"Labels shape: {labels.shape}, unique: {np.unique(labels)}")

    df = generate_csv_V4(labels, img)  # save temporary, will be renamed in loop
    return df

from skimage import io, color
import numpy as np

def analyze_image_V2(img_raw, verbose=True):
    # Handle both file paths and arrays
    if isinstance(img_raw, str):
        # Case 1: path string
        make_histogram_V5(img_raw)
        img = io.imread(img_raw, as_gray=True)

    elif isinstance(img_raw, np.ndarray):
        # Case 2: numpy array (already loaded)
        make_histogram_V5(img_raw)  # <-- if this expects a file path, may need adjustment

        img = img_raw
        if img.ndim == 3:  # If RGB, convert to grayscale
            img = color.rgb2gray(img)

    else:
        raise TypeError("img_raw must be either a file path (str) or a NumPy array")

    # Thresholding + segmentation
    sure_bg, sure_fg = auto_thresholds(img, method="otsu")
    labels, props, props_table = w_seg_f_b(img, sure_fg, sure_bg, verbose)

    print(f"Labels shape: {labels.shape}, unique: {np.unique(labels)}")

    # Generate dataframe
    df = generate_csv_V4(labels, img)
    return df


# data_clean

def remove_redundancy(df, min_threshold):
    df_filtered = df[df["area"] >= min_threshold].copy()
    return df_filtered

def find_parameters(df_filtered):
    mean_area = df_filtered["area"].mean()
    median_area = df_filtered["area"].median()
    std_area = df_filtered["area"].std()
    try:
        # statistics.mode might fail if all values are unique, so we use histogram fallback
        mode_area = mode(df_filtered["area"])
    except StatisticsError:
        # Use the bin with the highest frequency as fallback
        counts, bins = np.histogram(df_filtered["area"], bins="auto")
        mode_area = bins[np.argmax(counts)]
    return mean_area, median_area, mode_area, std_area

def classify_bands(df_filtered, parameter):
    labels = []
    tolerance = 0.6  # 60% tolerance

    for area in df_filtered["area"]:
        ratio = area / parameter
    
        if abs(ratio - 1) <= tolerance:
            labels.append("band")
        elif abs(ratio - round(ratio)) <= tolerance and ratio > 1:
            labels.append("likely fused band")
        else:
            labels.append("uncertain")

    df_filtered["label"] = labels

    return df_filtered

# testcode

def watershed_seg_V4(image, sure_fg, sure_bg, verbose=False):
    elevation_map = sobel(image)

    # Force int markers
    markers = np.zeros_like(image, dtype=np.int32)
    markers[image < sure_bg] = 1
    markers[image > sure_fg] = 2

    segmentation = skimage.segmentation.watershed(elevation_map, markers)
    segmentation = ndi.binary_fill_holes(segmentation - 1)
    labeled_bands, _ = ndi.label(segmentation)

    if verbose:
        overlay = label2rgb(labeled_bands, image=image, bg_label=0)
        plt.figure()
        plt.title("Region-based Segmentation")
        plt.imshow(overlay)

    return labeled_bands

def adjust_image(image_input, alpha=1.0, beta=0):
    # If it's a path (string), read from disk
    if isinstance(image_input, str):
        img = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
    
    # If it's already a numpy array, use it directly
    elif isinstance(image_input, np.ndarray):
        img = image_input
    
    # If it's a Streamlit UploadedFile
    else:
        file_bytes = np.asarray(bytearray(image_input.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Image could not be loaded. Check the input type.")

    # Apply contrast (alpha) and brightness (beta)
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted

def resize_to_average(image_file, ref_size=None):
    """
    Resize an image while preserving aspect ratio.
    If ref_size is given, both images will share the same final size.
    No padding or background fill is applied.
    """
    image = Image.open(image_file).convert("RGB")

    if ref_size is not None:
        image = image.resize(ref_size, Image.LANCZOS)
    else:
        w, h = image.size
        avg_dim = int((w + h) / 2)
        image.thumbnail((avg_dim, avg_dim), Image.LANCZOS)

    return np.array(image)

def auto_convert_columns(df):
    """
    Convert numeric-like columns to numeric values while keeping categorical columns as text.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    numeric_cols = []
    categorical_cols = []

    for col in df.columns:
        # Skip columns that are clearly categorical
        if col.lower() in ["label", "type", "band id", "sample", "source_file"]:
            categorical_cols.append(col)
            continue

        try:
            converted = pd.to_numeric(df[col], errors="coerce")
            numeric_ratio = converted.notna().mean()
            if numeric_ratio >= 0.6:
                df[col] = converted
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)
        except Exception:
            categorical_cols.append(col)

    # Drop empty columns or rows
    df = df.dropna(axis=1, how="all").dropna(how="all")

    return df, numeric_cols, categorical_cols



def predict_bandwise_results(csv_input, model_path="rf_model.pkl", fill_missing="zero"):
    """
    Predicts gene editing outcome (Wild Type / Knock In / Knock Out) for each band (row)
    using a trained Random Forest model.
    """
    import joblib
    import pandas as pd
    import numpy as np

    # Load input
    if isinstance(csv_input, str):
        df = pd.read_csv(csv_input)
    elif isinstance(csv_input, pd.DataFrame):
        df = csv_input.copy()
    else:
        raise ValueError("csv_input must be a file path or pandas DataFrame")

    # Load model
    model = joblib.load(model_path)

    # Prepare numeric features
    df = df.replace("missing", 0)
    if fill_missing == "zero":
        df = df.fillna(0)

    X = df.select_dtypes(include=[np.number])

    # Align features
    model_features = getattr(model, "feature_names_in_", None)
    if model_features is not None:
        for m in model_features:
            if m not in X.columns:
                X[m] = 0
        X = X[model_features]

    # Predict each band
    preds = model.predict(X)
    probs = model.predict_proba(X)

    df["Predicted_Class"] = preds
    for i, cls in enumerate(model.classes_):
        df[f"Prob_{cls}"] = probs[:, i]

    # Summary
    summary = df["Predicted_Class"].value_counts().to_dict()

    return {
        "full_results": df,
        "summary": summary,
        "classes": list(model.classes_)
    }


print("All functions ran sucessfully!")