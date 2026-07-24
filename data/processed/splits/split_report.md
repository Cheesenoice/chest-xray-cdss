# Dataset Split Summary Report

- **Num Classes:** 4
- **Random Seed:** 42
- **Patient Leakage:** ZERO (verified via GroupShuffleSplit & Assertions)
- **MD5 Hash Leakage:** ZERO (verified via MD5 set intersections)

## Image & Patient Distribution

| Split | normal | bacterial_pneumonia | viral_pneumonia | tuberculosis | Total Images | Unique Patients |
|---|---|---|---|---|---|---|
| **Train** | 1325 | 1966 | 1060 | 235 | **4586** | 2740 |
| **Validation** | 270 | 411 | 226 | 59 | **966** | 587 |
| **Internal Test** | 310 | 383 | 199 | 42 | **934** | 588 |
| **External Test** | 240 | 0 | 0 | 174 | **414** | 138 |
