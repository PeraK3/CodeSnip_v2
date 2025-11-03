import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import codesnip_v1
import image_editor2

from skimage import io, color
from PIL import Image
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


st.title(":green[Gene Editing Analyzer]")

# ===============================
# IMAGE UPLOAD SECTION
# ===============================
col_img_ctrl, col_img_exp = st.columns(2)
col_ctrl, col_exp = st.columns(2)

with col_img_ctrl:
    img_raw_ctrl = st.file_uploader("Upload your CONTROL image", ["jpg", "jpeg", "png"])

with col_img_exp:
    img_raw_exp = st.file_uploader("Upload your EXPERIMENTAL image", ["jpg", "jpeg", "png"])



# --- Resize both uploaded images to same average size ---
if img_raw_ctrl and img_raw_exp:
    # Open both to measure sizes
    img_ctrl = Image.open(img_raw_ctrl)
    img_exp = Image.open(img_raw_exp)

    w_ctrl, h_ctrl = img_ctrl.size
    w_exp, h_exp = img_exp.size

    # Average width/height
    avg_w = int((w_ctrl + w_exp) / 2)
    avg_h = int((h_ctrl + h_exp) / 2)

    # Resize both using your provided function
    resized_ctrl = image_editor2.resize_to_average(img_raw_ctrl, ref_size=(avg_w, avg_h))
    resized_exp = image_editor2.resize_to_average(img_raw_exp, ref_size=(avg_w, avg_h))

    # Replace uploaded objects with resized versions
    img_raw_ctrl = Image.fromarray(resized_ctrl)
    img_raw_exp = Image.fromarray(resized_exp)



# ===============================
# CONTROL IMAGE SECTION
# ===============================
with col_ctrl:
    if img_raw_ctrl is not None:
        st.divider()

        img_array_ctrl = np.array(img_raw_ctrl)
        df_ctrl = codesnip_v1.analyze_image_V2(img_array_ctrl)

        # --- Convert to grayscale safely
        gray_ctrl = img_array_ctrl
        if gray_ctrl.ndim == 3 and gray_ctrl.shape[2] == 3:
            gray_ctrl = color.rgb2gray(gray_ctrl)

        bg_ctrl, fg_ctrl = codesnip_v1.auto_thresholds(gray_ctrl)
        labels_ctrl = codesnip_v1.watershed_seg_V4(image=gray_ctrl, sure_fg=fg_ctrl, sure_bg=bg_ctrl)

        fig_ctrl, ax_ctrl = plt.subplots()
        im_ctrl = ax_ctrl.imshow(labels_ctrl, cmap="nipy_spectral")
        ax_ctrl.set_title("Identifiable bands (Control)")
        fig_ctrl.colorbar(im_ctrl, ax=ax_ctrl)

        col3_ctrl, col4_ctrl = st.columns(2)
        with col3_ctrl:
            st.image(img_array_ctrl, caption="Control Image")
        with col4_ctrl:
            st.pyplot(fig_ctrl)

        col5_ctrl, col6_ctrl = st.columns([0.3, 0.7])
        with col5_ctrl:
            alpha_ctrl = st.slider("Adjust contrast (Control)", -1.0, 5.0, 1.0)
            beta_ctrl = st.slider("Adjust brightness (Control)", -50, 50, 0)

        img_tweak_ctrl = image_editor2.adjust_image(img_array_ctrl, alpha_ctrl, beta_ctrl)

        # --- Convert to grayscale safely (after tweak)
        gray_tweak_ctrl = img_tweak_ctrl
        if gray_tweak_ctrl.ndim == 3 and gray_tweak_ctrl.shape[2] == 3:
            gray_tweak_ctrl = color.rgb2gray(gray_tweak_ctrl)

        bg_ctrl_tweak, fg_ctrl_tweak = codesnip_v1.auto_thresholds(gray_tweak_ctrl)
        labels_ctrl_tweak = codesnip_v1.watershed_seg_V4(image=gray_tweak_ctrl, sure_fg=fg_ctrl_tweak, sure_bg=bg_ctrl_tweak)

        fig_tweak_ctrl, ax_tweak_ctrl = plt.subplots()
        im_tweak_ctrl = ax_tweak_ctrl.imshow(labels_ctrl_tweak, cmap="nipy_spectral")
        ax_tweak_ctrl.set_title("Identifiable bands (Adjusted Control)")

        with col6_ctrl:
            st.image(img_tweak_ctrl, caption="Adjusted Control Image")
            st.pyplot(fig_tweak_ctrl)

        st.dataframe(df_ctrl)

        st.divider()
        col1_ctrl, col2_ctrl = st.columns(2)
        with col1_ctrl:
            tolerance_ctrl = st.slider("Band labelling tolerance (Control)", 0.0, 1.0, 0.30)
        with col2_ctrl:
            min_threshold_ctrl = st.slider("Delete small bands (Control)", 0, 200, 10)

        df_filtered_ctrl = codesnip_v1.remove_redundancy(df_ctrl, min_threshold_ctrl)
        mean_ctrl, median_ctrl, mode_ctrl, std_ctrl = codesnip_v1.find_parameters(df_filtered_ctrl)

        parameters_dict_ctrl = {
            "Mean": float(mean_ctrl),
            "Median": float(median_ctrl),
            "Mode": float(mode_ctrl)
        }

        height_ctrl, width_ctrl = img_tweak_ctrl.shape[:2]
        df_filtered_ctrl["Image Height"] = height_ctrl
        df_filtered_ctrl["centroid_cart-0"] = height_ctrl - df_filtered_ctrl["centroid-0"]

        selected_key_ctrl = st.selectbox("Pick a parameter (Control):", parameters_dict_ctrl)
        selected_param_ctrl = parameters_dict_ctrl[selected_key_ctrl]

        df_filtered_ctrl = codesnip_v1.classify_bands(df_filtered_ctrl, selected_param_ctrl)

        st.divider()
        st.dataframe(df_filtered_ctrl)
        st.scatter_chart(df_filtered_ctrl, x="centroid-1", y="centroid_cart-0",
                         x_label="X axis", y_label="Y axis", color="label", size="area")

# ===============================
# EXPERIMENTAL IMAGE SECTION
# ===============================
with col_exp:
    if img_raw_exp is not None:
        st.divider()

        img_array_exp = np.array(img_raw_exp)
        df_exp = codesnip_v1.analyze_image_V2(img_array_exp)

        # --- Convert to grayscale safely
        gray_exp = img_array_exp
        if gray_exp.ndim == 3 and gray_exp.shape[2] == 3:
            gray_exp = color.rgb2gray(gray_exp)

        bg_exp, fg_exp = codesnip_v1.auto_thresholds(gray_exp)
        labels_exp = codesnip_v1.watershed_seg_V4(image=gray_exp, sure_fg=fg_exp, sure_bg=bg_exp)

        fig_exp, ax_exp = plt.subplots()
        im_exp = ax_exp.imshow(labels_exp, cmap="nipy_spectral")
        ax_exp.set_title("Identifiable bands (Experimental)")
        fig_exp.colorbar(im_exp, ax=ax_exp)

        col3_exp, col4_exp = st.columns(2)
        with col3_exp:
            st.image(img_array_exp, caption="Experimental Image")
        with col4_exp:
            st.pyplot(fig_exp)

        col5_exp, col6_exp = st.columns([0.3, 0.7])
        with col5_exp:
            alpha_exp = st.slider("Adjust contrast (Experimental)", -1.0, 5.0, 1.0)
            beta_exp = st.slider("Adjust brightness (Experimental)", -50, 50, 0)

        img_tweak_exp = image_editor2.adjust_image(img_array_exp, alpha_exp, beta_exp)

        # --- Convert to grayscale safely (after tweak)
        gray_tweak_exp = img_tweak_exp
        if gray_tweak_exp.ndim == 3 and gray_tweak_exp.shape[2] == 3:
            gray_tweak_exp = color.rgb2gray(gray_tweak_exp)

        bg_exp_tweak, fg_exp_tweak = codesnip_v1.auto_thresholds(gray_tweak_exp)
        labels_exp_tweak = codesnip_v1.watershed_seg_V4(image=gray_tweak_exp, sure_fg=fg_exp_tweak, sure_bg=bg_exp_tweak)

        fig_tweak_exp, ax_tweak_exp = plt.subplots()
        im_tweak_exp = ax_tweak_exp.imshow(labels_exp_tweak, cmap="nipy_spectral")
        ax_tweak_exp.set_title("Identifiable bands (Adjusted Experimental)")

        with col6_exp:
            st.image(img_tweak_exp, caption="Adjusted Experimental Image")
            st.pyplot(fig_tweak_exp)

        st.dataframe(df_exp)

        st.divider()
        col1_exp, col2_exp = st.columns(2)
        with col1_exp:
            tolerance_exp = st.slider("Band labelling tolerance (Experimental)", 0.0, 1.0, 0.30)
        with col2_exp:
            min_threshold_exp = st.slider("Delete small bands (Experimental)", 0, 200, 10)

        df_filtered_exp = codesnip_v1.remove_redundancy(df_exp, min_threshold_exp)
        mean_exp, median_exp, mode_exp, std_exp = codesnip_v1.find_parameters(df_filtered_exp)

        parameters_dict_exp = {
            "Mean": float(mean_exp),
            "Median": float(median_exp),
            "Mode": float(mode_exp)
        }

        height_exp, width_exp = img_tweak_exp.shape[:2]
        df_filtered_exp["Image Height"] = height_exp
        df_filtered_exp["centroid_cart-0"] = height_exp - df_filtered_exp["centroid-0"]

        selected_key_exp = st.selectbox("Pick a parameter (Experimental):", parameters_dict_exp)
        selected_param_exp = parameters_dict_exp[selected_key_exp]

        df_filtered_exp = codesnip_v1.classify_bands(df_filtered_exp, selected_param_exp)

        st.divider()
        st.dataframe(df_filtered_exp)
        st.scatter_chart(df_filtered_exp, x="centroid-1", y="centroid_cart-0",
                         x_label="X axis", y_label="Y axis", color="label", size="area")


# ===============================
# COMBINED DATA SECTION (DISTANCE MATRIX METHOD)
# ===============================
st.divider()
if "df_filtered_ctrl" in locals() and "df_filtered_exp" in locals():
    st.markdown("### 🧩 Combined Analysis (Distance-Based Matching)")

    # Extract centroids
    ctrl_coords = df_filtered_ctrl[["centroid-1", "centroid_cart-0"]].to_numpy()
    exp_coords = df_filtered_exp[["centroid-1", "centroid_cart-0"]].to_numpy()

    # Compute distance matrix
    distance_matrix = cdist(ctrl_coords, exp_coords, metric="euclidean")

    # Solve optimal assignment
    row_ind, col_ind = linear_sum_assignment(distance_matrix)

    # Define threshold for valid matches (pixels)
    max_distance = st.slider("Max allowed centroid distance for match (px)", 5, 150, 50)

    # Prepare matched and unmatched entries
    matched_ctrl_idx = []
    matched_exp_idx = []
    matched_pairs = []

    for i, j in zip(row_ind, col_ind):
        dist = distance_matrix[i, j]
        if dist <= max_distance:
            matched_ctrl_idx.append(i)
            matched_exp_idx.append(j)
            matched_pairs.append((i, j, dist))

    # Build matched DataFrame
    matched_ctrl = df_filtered_ctrl.iloc[matched_ctrl_idx].reset_index(drop=True)
    matched_exp = df_filtered_exp.iloc[matched_exp_idx].reset_index(drop=True)
    combined_df = pd.concat(
        [matched_ctrl.add_suffix("_Control"), matched_exp.add_suffix("_Experimental")],
        axis=1
    )

    # Label unmatched bands
    unmatched_ctrl = set(range(len(df_filtered_ctrl))) - set(matched_ctrl_idx)
    unmatched_exp = set(range(len(df_filtered_exp))) - set(matched_exp_idx)

    # Add rows for missing control bands
    for idx in unmatched_ctrl:
        ctrl_row = df_filtered_ctrl.iloc[[idx]].add_suffix("_Control")
        empty_exp = pd.DataFrame(columns=[c + "_Experimental" for c in df_filtered_exp.columns])
        combined_df = pd.concat([combined_df, pd.concat([ctrl_row, empty_exp], axis=1)], ignore_index=True)

    # Add rows for missing experimental bands
    for idx in unmatched_exp:
        exp_row = df_filtered_exp.iloc[[idx]].add_suffix("_Experimental")
        empty_ctrl = pd.DataFrame(columns=[c + "_Control" for c in df_filtered_ctrl.columns])
        combined_df = pd.concat([combined_df, pd.concat([empty_ctrl, exp_row], axis=1)], ignore_index=True)

    # Fill NaNs
    combined_df = combined_df.fillna("missing")

    # Add Band Status
    combined_df["Band_Status"] = combined_df.apply(
        lambda r: "Matched"
        if (r["label_Control"] != "missing") and (r["label_Experimental"] != "missing")
        else ("Missing in Control" if r["label_Control"] == "missing" else "Missing in Experimental"),
        axis=1
    )

    combined_df = combined_df.replace("missing", np.nan)

    # Optionally delete uncertain/fused
    is_del_redundant = st.checkbox("Delete Uncertains and Fused Bands?")
    if is_del_redundant:
        for col in ["label_Control", "label_Experimental"]:
            if col in combined_df.columns:
                combined_df = combined_df[
                    ~combined_df[col].isin(["likely fused band", "uncertain"])
                ]

    # ===============================
    # AUTO-CONVERT TEXT TO NUMERIC (from image_editor2)
    # ===============================
    combined_df, num_cols, cat_cols = image_editor2.auto_convert_columns(combined_df)

    # Display result
    st.dataframe(combined_df)

# ===============================
# MODEL PREDICTION SECTION
# ===============================
if st.button("Run the Model"):
    import joblib

    st.markdown("### 🤖 Model Prediction Section")

    # Load your trained model
    try:
        model = joblib.load("rf_model.pkl")  # <-- change filename if different
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    # Copy dataframe for prediction
    df_pred_input = combined_df.copy()

    # Ensure same columns as during training
    expected_features = model.feature_names_in_ if hasattr(model, "feature_names_in_") else None

    if expected_features is not None:
        for col in expected_features:
            if col not in df_pred_input.columns:
                # Create placeholder values for missing columns
                if col.lower().startswith("source") or "file" in col.lower():
                    df_pred_input[col] = "unknown"
                else:
                    df_pred_input[col] = 0  # neutral numeric filler

        # Drop extra columns not used by model
        df_pred_input = df_pred_input[[c for c in expected_features]]

    # Convert object columns to categorical codes or numerics
    for col in df_pred_input.select_dtypes(include=["object"]).columns:
        df_pred_input[col] = df_pred_input[col].astype("category").cat.codes

    # Run prediction
    try:
        preds = model.predict(df_pred_input)
        combined_df["Predicted_Class"] = preds

        # Optional: display prediction probabilities
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df_pred_input)
            combined_df["Confidence"] = probs.max(axis=1)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # Display results
    st.success("✅ Model prediction completed successfully.")
    st.dataframe(combined_df[["Band_Status", "Predicted_Class", "Confidence"]])


