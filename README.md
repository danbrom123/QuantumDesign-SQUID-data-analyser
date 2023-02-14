# QuantumDesign SQUID data analyser

A Python script for quick analysis of moment-vs-field (MH) curves of ferromagnetic samples measured using a QuantumDesign (QD) SQUID VSM. The script streamlines the data analysis of MH curves by allowing users to easily remove initial saturation, paramagnetic backgrounds and drifts from MH curves.

This was written to speed up processing MH curves for data analysis for my PhD thesis (there were lots of them!).

## How it works:
- Choose QD SQUID file using interactive file browser
- File is loaded in to pandas dataframe
- MH curve is plotted of raw data
- initial saturation curve is removed
- User has option to remove paramagnetic signal (caused by measurement holder)
  - If chosen a plot showing the removal method is produced
-User has option to remove drift from signal
- User has option to normalise data (for easy comparison with other signals)
- Final plot is produced (this is formatted to thesis/publication quality)

## Examples

file = "/example_data/S2_Cu-OOP-MH@5K_3kOe_course_1500Oe_fine-separate_run.dat"

<p float="left">
  <img src="/images/S2-IP-MH@5K_2T_course_8kOe_fine - RAW DATA.png" width="333" />
  <img src="/images/S2-IP-MH@5K_2T_course_8kOe_fine - PARAMAG_REMOVE.png" width="333" /> 
  <img src="/images/S2-IP-MH@5K_2T_course_8kOe_fine - FIGURE.png" width="333" />
</p>
<p align="center"><i>Figure 1.  Example analysis for in-plane applied field MH measurment. </i></p>

file = "/example_data/S2-IP-MH@5K_2T_course_8kOe_fine.dat"

<p float="left">
  <img src="/images/S2_Cu-OOP-MH@5K_3kOe_course_1500Oe - RAW DATA.png" width="333" />
  <img src="/images/S2_Cu-OOP-MH@5K_3kOe_course_1500Oe - PARAMAG_REMOVE.png" width="333" /> 
  <img src="/images/S2_Cu-OOP-MH@5K_3kOe_course_1500Oe - FIGURE.png" width="333" />
</p>
<p align="center"><i>Figure 2.  Example analysis for out-of-plane applied field MH measurment. </i></p>
