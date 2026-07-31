# Dataset Split Summary Report

- **Num Classes:** 3
- **Random Seed:** 42
- **Patient Leakage:** ZERO (verified via GroupShuffleSplit & Assertions)
- **MD5 Hash Leakage:** ZERO (verified via MD5 set intersections)

## Image & Patient Distribution

| Split | normal | bacterial_pneumonia | viral_pneumonia | Total Images | Unique Patients |
|---|---|---|---|---|---|
| **Train** | 1360 | 2029 | 1053 | **4442** | 2561 |
| **Validation** | 290 | 377 | 229 | **896** | 549 |
| **Internal Test** | 335 | 354 | 203 | **892** | 549 |
| **External Test** | 0 | 0 | 0 | **0** | 0 |
