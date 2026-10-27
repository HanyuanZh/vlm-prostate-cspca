# Data

The processed data used in the paper are available at
https://drive.google.com/drive/folders/12EuUeEnfmpzD5WPda4lcMLWiCg7Y9W6H?usp=sharing

The original dataset is described below. `--data-root` must contain:

```
<data-root>/
  prostate/
    t2/<patient_id>_<study_id>.png
    dwi/<patient_id>_<study_id>.png
    adc/<patient_id>_<study_id>.png
    marksheet.csv
```

## PI-CAI

Public training data and marksheet of the PI-CAI challenge
(<https://pi-cai.grand-challenge.org/>; Saha et al., *Lancet Oncol* 2024): 1,500
biparametric MRI examinations from 1,476 patients.

* One 256 × 256 grayscale PNG per examination and sequence (T2W, high b-value DWI,
  ADC), named `<patient_id>_<study_id>.png`, in `t2/`, `dwi/` and `adc/`.
* `marksheet.csv` is the PI-CAI marksheet; the columns used are `patient_id`,
  `study_id`, `psa`, `prostate_volume` and `case_csPCa` (reference label, `YES`/`NO`).

> **To be added:** the script that extracts the 2-D PNGs from the PI-CAI volumes.
