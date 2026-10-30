# Data included in this release

`frozen/panel_w5.csv` is the de-identified, analysis-ready panel for the six
standard NHANES cycles used in the primary temporal analyses. `frozen/panel_w7.csv`
adds the later-cycle panel used for the reduced-predictor and measurement analyses.
They are included so that the released code can reproduce the reported analyses
without a separate extraction step.

The repository deliberately does **not** redistribute raw NHANES XPT files.
Those public-use source files remain available from the National Center for
Health Statistics (NCHS) NHANES data portal: <https://wwwn.cdc.gov/nchs/nhanes/>.
The frozen panels contain only variables used by this study plus survey design
variables needed for the stated jackknife calculations. They must not be used
to re-identify participants or linked to other data.
