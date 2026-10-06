# Energy dataset used by the October 2026 experiments

`energydata_complete.csv` is the cached UCI Appliances Energy Prediction CSV
used by the frozen Energy protocols. It contains 19,735 observations and 29
columns. This publication copies that cache byte-for-byte; it does not modify
its values or claim byte identity with a newly downloaded UCI export.

Source and attribution: Candanedo, L. (2017). *Appliances Energy Prediction*
[Dataset]. UCI Machine Learning Repository.
[Dataset page](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)
· [DOI: 10.24432/C5VC8G](https://doi.org/10.24432/C5VC8G).
The dataset is distributed under
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
Dataset license and attribution apply to this CSV independently of repository code.

Related paper: Candanedo, L. M., Feldheim, V., and Deramaix, D. (2017).
*Data driven prediction models of energy use of appliances in a low-energy house.*
Energy and Buildings, 140, 81–97.

SHA-256: `d5107b1732a324021038393e112ac738c038a37a5441f8af7dc5e64757378e0e`.

The scripts exclude `date`, `Appliances`, `rv1`, and `rv2` from the predictors,
leaving 25 features, including `lights`. Labels and preprocessing are fitted
inside each training split; the CSV itself remains unchanged.
