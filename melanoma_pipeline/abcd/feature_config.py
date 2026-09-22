"""The frozen V5 feature set's name/order list ONLY -- copied from
pipeline_v5/decision_model.py::V5_CONFIG. This package deliberately does
NOT consolidate the rest of that module (`load_frozen_pipeline`,
`v5_predict_from_row`, `frozen_model.pkl`): the multimodal pipeline in
melanoma_pipeline/ trains its own CatBoost classifier on these feature
VALUES, it does not use V5's own frozen logistic-regression classifier or
its 0.25 threshold. Only the feature name list/order is a real dependency
here.

Order matters for reproducibility/debugging even though this package's own
`features.py` builds a name-keyed dict internally: this list is what the
original V5 model was fit on, and is preserved here unchanged as the
canonical reference order for these 17 features.
"""

V5_BASE_FEATURES = ["A_value", "B_circularity", "C_value", "D_px", "confidence", "lesion_fraction"]
V5_NEW_FEATURES = ["color_entropy", "lab_a_std", "lab_b_std", "red_fraction", "bluegray_fraction",
                    "dark_fraction", "skin_contrast", "solidity", "turning_angle_std",
                    "eccentricity", "D_px_normalized"]
V5_ALL_FEATURES = V5_BASE_FEATURES + V5_NEW_FEATURES  # 17 features, in the original's exact fitting order
