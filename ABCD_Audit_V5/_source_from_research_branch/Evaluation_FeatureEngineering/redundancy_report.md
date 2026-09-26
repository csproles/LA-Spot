# Redundancy / quality-control report

## Near-zero-variance features (std ~ 0, flagged, not usable)

(none)


## High-missingness features (>5% NaN)

(none)


## Highly redundant pairs (|Spearman rho| > 0.90)

- B_circularity <-> isoperimetric_ratio: rho=1.0
- solidity <-> convexity_deficit: rho=-1.0
- aspect_ratio <-> eccentricity: rho=1.0
- D_px <-> major_axis_px: rho=0.998
- lesion_fraction <-> D_px_normalized: rho=0.989
- D_px <-> lesion_area_px: rho=0.981
- lesion_area_px <-> minor_axis_px: rho=0.98
- lesion_area_px <-> major_axis_px: rho=0.977
- local_contrast <-> glcm_homogeneity: rho=-0.95
- lab_L_std <-> entropy_intensity: rho=0.945
- D_px <-> minor_axis_px: rho=0.932
- major_axis_px <-> minor_axis_px: rho=0.922
- radial_cv <-> aspect_ratio: rho=0.913
- radial_cv <-> eccentricity: rho=0.913
- local_contrast <-> glcm_contrast: rho=0.908
