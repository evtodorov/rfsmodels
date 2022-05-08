$# LS-DYNA Keyword file created by LS-PrePost(R) V4.7.17 - 08Jul2020
$# Created on Mar-27-2022 (10:50:23)
*KEYWORD
*PART
Upper part
         1         1&mat               0         0         0         0         0
*SECTION_SHELL_TITLE
Section 1 (top)
         1        12    0.8333         1       1.0         0         0         1
&thk1     &thk1     &thk1     &thk1            0.0       0.0       0.0         0
*MAT_PIECEWISE_LINEAR_PLASTICITY_TITLE
Rubber-like linear plastic material
         21.52200E-6      0.02   0.49925       0.0       0.01.00000E21       0.0
       0.0       0.0         0         0       0.0
       0.0       0.1       0.2       0.5       1.0       5.0       0.0       0.0
    0.0002    0.0007     0.001    0.0018     0.003      0.01       0.0       0.0
*PART
Bottom part
         2         2&mat               0         0         0         0         0
*SECTION_SHELL_TITLE
Section 2 (bottom)
         2        12    0.8333         1       1.0         0         0         1
&thk2     &thk2     &thk2     &thk2            0.0       0.0       0.0         0
*MAT_OGDEN_RUBBER_TITLE
Neo-Hookean-like Rubber
         11.52200E-6   0.49925         0         0       0.0       0.0       0.0
    0.0015       0.0       0.0       0.0       0.0       0.0       0.0       0.0
       2.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0
*END
