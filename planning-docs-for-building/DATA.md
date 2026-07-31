# DATA.md — Kế hoạch xử lý dữ liệu & Đặc tả Data Pipeline

> File này bổ sung cho `CLAUDE.md`. Nó đặc tả TOÀN BỘ pipeline dữ liệu cho dự án
> "Explainable Deep Learning Clinical Decision Support System for Chest X-ray Screening".
> Claude Code PHẢI đọc kỹ trước khi viết bất kỳ code xử lý dữ liệu nào, và bám sát các
> ràng buộc khoa học ở đây (chống rò rỉ dữ liệu là ưu tiên tuyệt đối).
>
> Nguyên tắc vàng: **thà pipeline chậm mà đúng, còn hơn nhanh mà rò rỉ.** Mọi bước có rủi
> ro rò rỉ hoặc gán nhãn sai phải DỪNG LẠI, in log rõ ràng, và giải thích cho người dùng.

---

## 0. Mục tiêu của pipeline

Biến các bộ ảnh X-quang công khai (định dạng, kích thước, cách đặt tên khác nhau) thành:
1. Một **manifest CSV thống nhất** (mỗi dòng = 1 ảnh) với các cột chuẩn hóa.
2. Các **file split CSV** (train / val / test / external) chia ở **mức bệnh nhân / mức
   nguồn**, tuyệt đối không rò rỉ.
3. Một **DataLoader** PyTorch nạp ảnh + augmentation, sẵn sàng cho training.
4. Một **báo cáo thống kê dữ liệu** (số ảnh mỗi lớp mỗi split, kích thước ảnh, kiểm tra
   trùng lặp, kiểm tra rò rỉ) để đưa vào paper và slide bảo vệ.

Pipeline phải **tái lập được**: chạy lại với cùng seed cho ra cùng split.

---

## 1. Các bộ dữ liệu (đã verify — nhưng vẫn kiểm tra lại slug trước khi tải)

> Slug Kaggle có thể đổi. Trước khi tải, verify sự tồn tại; nếu lỗi, tìm bộ tương đương và
> báo người dùng.

| # | Bộ dữ liệu | Slug Kaggle | Nội dung | License / Ghi chú |
|---|-----------|-------------|----------|-------------------|
| A | Kermany Chest X-Ray (Pneumonia) | `paultimothymooney/chest-xray-pneumonia` | ~5.863 ảnh, Normal / Pneumonia (bacterial & viral suy từ tên file). **Trẻ em, Quảng Châu.** | CC BY 4.0. Phải trích dẫn Kermany et al. 2018. |
| B | TB Chest X-ray Database (Rahman) | `tawsifurrahman/tuberculosis-tb-chest-xray-dataset` | Gộp Montgomery, Shenzhen, Belarus, NIAID; phổ biến có ~700 TB + ~3.500 Normal. **Người lớn.** | Trích dẫn Rahman et al. "Reliable TB Detection". |
| C | Pulmonary Chest X-ray Abnormalities | `kmader/pulmonary-chest-xray-abnormalities` | Shenzhen (662: 336 TB / 326 normal) + Montgomery (138: 58 TB / 80 normal), kèm mask phổi. | Ghi nguồn NLM/NIH + Shenzhen No.3 Hospital; trích dẫn Jaeger 2014 & Candemir 2014. Lưu ý: Shenzhen có lẫn một ít ảnh nhi. |
| D | (Tùy chọn) 4-class gộp sẵn | `jtiptj/chest-xray-pneumoniacovid19tuberculosis` | ~7.135 ảnh, 4 lớp Normal / Pneumonia / COVID-19 / TB. | Tiện nếu muốn bỏ qua công gộp thủ công. |

**Phân vai mặc định (khuyến nghị):**
- **Kermany (A)** → nguồn train/val/test cho các lớp `Normal`, `Bacterial pneumonia`,
  `Viral pneumonia`. Chia ở mức bệnh nhân (xem mục 4).
- **Shenzhen (trong C)** → nguồn train/val cho lớp `Tuberculosis`.
- **Montgomery (trong C)** → **giữ lại HOÀN TOÀN làm external test cho TB** (held-out
  source). KHÔNG dùng để train.
- **(Tùy chọn) một nguồn viêm phổi khác** (ví dụ subset viral pneumonia của bộ D, hoặc bộ
  RSNA) → external test cho viêm phổi, nếu còn thời gian.

Cho phép chạy 2 cấu hình qua config:
- `num_classes: 3` → chỉ Kermany (Normal / Bacterial / Viral) — **benchmark sạch nhất,
  đáng tin nhất, dùng làm kết quả chính**.
- `num_classes: 4` → thêm TB — **giàu hơn cho sản phẩm demo**, kèm ghi chú domain confound.

---

## 2. Quy tắc đặt tên file (để gán nhãn & suy ID bệnh nhân)

Đây là chi tiết kỹ thuật then chốt — pipeline dựa vào đây để gán nhãn và chống rò rỉ.

**Kermany (A):**
- Ảnh viêm phổi: `person{N}_bacteria_{M}.jpeg` hoặc `person{N}_virus_{M}.jpeg`.
  → Nhãn bacterial/viral suy từ chuỗi `bacteria`/`virus`.
  → **`patient_id = "person{N}"`** — nhóm theo đây để chia mức bệnh nhân.
- Ảnh normal: dạng `IM-xxxx-....jpeg` hoặc `NORMAL2-IM-xxxx-....jpeg` (không có `person`).
  → Với ảnh normal không truy được patient rõ ràng, coi mỗi file là một đơn vị riêng
    NHƯNG vẫn kiểm tra trùng lặp; ghi chú giới hạn này trong paper.

**Shenzhen (trong C):** tên dạng `CHNCXR_{id}_{label}.png`, trong đó `label`:
`0` = normal, `1` = TB. → nhãn suy từ hậu tố; `patient_id` suy từ `{id}`.

**Montgomery (trong C):** tên dạng `MCUCXR_{id}_{label}.png`, `0` = normal, `1` = TB.

→ Viết hàm parser riêng cho từng nguồn, KHÔNG giả định một quy tắc chung. Nếu gặp tên
file không khớp mẫu, log cảnh báo và bỏ qua (không đoán bừa).

---

## 3. Giai đoạn 1 — Tải dữ liệu (`src/data/download.py`)

Trách nhiệm: tải các bộ A, B/C về thư mục `data/raw/` một cách idempotent (đã có thì bỏ
qua). Không commit dữ liệu (thêm `data/` vào `.gitignore`).

Hỗ trợ 2 chế độ:
- **kagglehub** (ưu tiên, ít cấu hình):
  ```python
  import kagglehub
  path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
  ```
- **Kaggle API / CLI** (cần `~/.kaggle/kaggle.json`):
  ```bash
  kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data/raw --unzip
  kaggle datasets download -d kmader/pulmonary-chest-xray-abnormalities -p data/raw --unzip
  ```
- **Chế độ Kaggle Notebook:** nếu chạy trên Kaggle, dữ liệu đã mount ở `/kaggle/input/`;
  chỉ cần trỏ đường dẫn, không tải lại. Pipeline nên tự phát hiện môi trường.

Ghi log: bộ nào đã tải, đường dẫn, dung lượng. Kiểm tra checksum/đếm file cơ bản sau tải.

---

## 4. Giai đoạn 2 — Chuẩn hóa & gán nhãn (`src/data/prepare.py`)

Trách nhiệm: quét toàn bộ ảnh từ `data/raw/`, tạo **manifest thống nhất**
`data/processed/manifest.csv` với schema cố định:

| Cột | Ý nghĩa |
|-----|---------|
| `filepath` | đường dẫn tuyệt đối tới ảnh |
| `source` | nguồn: `kermany` / `shenzhen` / `montgomery` / ... |
| `label` | nhãn chuẩn hóa: `normal` / `bacterial_pneumonia` / `viral_pneumonia` / `tuberculosis` |
| `patient_id` | ID bệnh nhân suy từ tên file (hoặc `source + hash` nếu không có) |
| `domain` | `pediatric` / `adult` (để phân tích confound) |
| `width`, `height` | kích thước ảnh gốc |
| `md5` | mã băm nội dung ảnh (để phát hiện trùng lặp) |

Các bước:
1. Duyệt từng nguồn bằng parser riêng (mục 2), gán `label`, `patient_id`, `source`,
   `domain`.
2. Đọc kích thước ảnh; loại ảnh hỏng/không đọc được (log lại).
3. Tính `md5` nội dung ảnh.
4. **Khử trùng lặp:** nếu nhiều ảnh cùng `md5`, giữ 1, loại phần còn lại; log số lượng đã
   loại. (Ảnh trùng giữa các split là nguồn rò rỉ điển hình.)
5. Xuất `manifest.csv` + in bảng thống kê (số ảnh mỗi `label` mỗi `source`).

Ràng buộc: KHÔNG gán nhãn bằng cách đoán; chỉ gán khi tên file khớp mẫu đã biết.

---

## 5. Giai đoạn 3 — Chia dữ liệu (`src/data/split.py`) — PHẦN QUAN TRỌNG NHẤT

Trách nhiệm: từ `manifest.csv`, tạo các file split CSV trong `data/processed/splits/`:
`train.csv`, `val.csv`, `test.csv`, và `external_test.csv`.

**Nguyên tắc bất di bất dịch:**
1. **Chia ở MỨC BỆNH NHÂN, không ở mức ảnh.** Dùng `GroupShuffleSplit` /
   `StratifiedGroupKFold` (scikit-learn) với `groups = patient_id`. Không một `patient_id`
   nào được xuất hiện ở hai split.
2. **External test = nguồn giữ lại hoàn toàn.** Toàn bộ ảnh `source == montgomery` đi vào
   `external_test.csv` và KHÔNG bao giờ vào train/val/test. Đây là held-out source để đo
   tổng quát hóa trung thực.
3. **Test set nội bộ chỉ đụng một lần** ở cuối; không dùng để chọn model.
4. Tỉ lệ mặc định (trong config): train 70% / val 15% / test 15% (tính trên các nguồn
   không phải external), phân tầng theo `label` ở mức nhóm nếu có thể.
5. Cố định `seed` từ config để tái lập.

**Sau khi chia, BẮT BUỘC chạy các kiểm tra rò rỉ (assert, dừng nếu sai):**
- Giao của tập `patient_id` giữa train/val/test phải RỖNG.
- Giao của tập `md5` giữa mọi cặp split phải RỖNG.
- `external_test` không chứa `source` nào trùng với nguồn train của cùng lớp.
- In bảng: số ảnh & số bệnh nhân mỗi lớp mỗi split; cảnh báo nếu lớp nào < ngưỡng tối
  thiểu.

**Lưu ý riêng cho lớp normal của Kermany** (không có patient_id rõ): xử lý theo mức file
nhưng vẫn khử trùng md5; ghi rõ đây là giới hạn trong paper.

Xuất kèm `data/processed/splits/split_report.md` tóm tắt để dán vào paper.

---

## 6. Giai đoạn 4 — Tiền xử lý & Dataset (`src/datasets.py`)

Trách nhiệm: lớp `Dataset` PyTorch đọc từ file split CSV.

**Tiền xử lý ảnh:**
- Đọc ảnh, chuyển về **RGB 3 kênh** (một số ảnh là grayscale/16-bit — chuẩn hóa về 8-bit
  3 kênh để dùng backbone ImageNet).
- Resize về **224×224** (cấu hình được).
- Chuẩn hóa theo **mean/std ImageNet** (`[0.485,0.456,0.406]`/`[0.229,0.224,0.225]`).

**Augmentation (chỉ áp dụng cho TRAIN, dùng `albumentations`):**
- Được phép: lật ngang (horizontal flip), xoay nhẹ (±10–15°), thay đổi độ sáng/tương phản
  vừa phải, dịch/zoom nhẹ.
- **CẤM** các phép làm méo/biến dạng bệnh lý hoặc phi thực tế: lật dọc (giải phẫu ngực
  không đối xứng trên–dưới), méo đàn hồi mạnh, đảo màu, cắt xén làm mất vùng phổi.
- Val/test/external: KHÔNG augmentation (chỉ resize + normalize).

**Xử lý mất cân bằng lớp:** cung cấp tùy chọn `WeightedRandomSampler` hoặc class weights
trong loss (bật/tắt qua config). Log phân bố lớp để minh bạch.

---

## 7. Xử lý domain confound (bắt buộc ghi nhận)

Kermany = **nhi** (trẻ em), TB (Shenzhen/Montgomery) = **người lớn**. Ở cấu hình 4 lớp,
model có thể tách TB khỏi viêm phổi nhờ đặc điểm tuổi/giải phẫu thay vì bệnh lý.

Pipeline phải:
- Gắn cột `domain` (`pediatric`/`adult`) trong manifest để phân tích sau.
- Cho phép phân tích chéo: xem model có nhầm dựa trên domain không (ví dụ báo cáo hiệu năng
  tách theo domain).
- Không "sửa" dữ liệu để giấu vấn đề. Thay vào đó, external validation (Montgomery giữ lại)
  và benchmark 3 lớp sạch (chỉ Kermany) là hai cách trình bày trung thực.

---

## 8. Kiểm tra chất lượng & sanity checks (`src/data/checks.py` hoặc trong split.py)

Chạy tự động, in ra `results/data_quality.md`:
- Tổng số ảnh, số ảnh mỗi lớp, mỗi nguồn, mỗi split.
- Số ảnh trùng md5 đã loại.
- Phân bố kích thước ảnh (min/median/max).
- Kiểm tra rò rỉ patient/md5 (assert như mục 5).
- Vài ảnh mẫu mỗi lớp (lưu thành lưới hình để mắt người kiểm tra nhãn có hợp lý không).
- Cảnh báo lớp thiểu số nghiêm trọng (ví dụ TB rất ít so với normal).

---

## 9. Cấu hình dữ liệu (khối `data` trong `configs/default.yaml`)

```yaml
data:
  num_classes: 4                 # 3 (chỉ Kermany) hoặc 4 (thêm TB)
  raw_dir: data/raw
  processed_dir: data/processed
  image_size: 224
  splits:
    train: 0.70
    val: 0.15
    test: 0.15
  external_sources: [montgomery] # nguồn giữ lại làm external test
  seed: 42
  dedup: true
  imbalance_strategy: weighted_sampler   # none | weighted_sampler | class_weights
  augmentation:
    horizontal_flip: true
    rotate_deg: 12
    brightness_contrast: 0.2
```

Mọi hằng số dữ liệu nằm ở đây, KHÔNG hardcode trong code.

---

## 10. Sản phẩm đầu ra của pipeline (artifacts)

- `data/processed/manifest.csv` — manifest thống nhất.
- `data/processed/splits/{train,val,test,external_test}.csv` — các split.
- `data/processed/splits/split_report.md` — tóm tắt split (cho paper).
- `results/data_quality.md` + lưới ảnh mẫu — báo cáo chất lượng.
- Log đầy đủ mỗi lần chạy (số liệu, seed, thời gian).

---

## 11. Thứ tự chạy (một lệnh tái lập)

```bash
python -m src.data.download        # tải về data/raw (idempotent)
python -m src.data.prepare         # -> manifest.csv (+ khử trùng, gán nhãn)
python -m src.data.split           # -> splits/*.csv (+ kiểm tra rò rỉ) 
python -m src.data.checks          # -> results/data_quality.md
```

Lý tưởng: gói lại thành `python -m src.data.build_all` chạy tuần tự cả 4 bước với cùng seed.

---

## 12. Ràng buộc khoa học & đạo đức (nhắc lại — cứng)

- **Chia mức bệnh nhân/nguồn**, không mức ảnh. Assert rò rỉ, dừng nếu vi phạm.
- **Test/external chỉ đụng một lần**; không dùng để tune.
- **Khử trùng lặp** trước khi chia.
- **Chỉ dùng dữ liệu công khai đúng license**; ghi license mỗi nguồn.
- **Trích dẫn paper gốc**: Kermany et al. 2018 (bộ A); Jaeger et al. 2014 & Candemir et
  al. 2014 (Shenzhen/Montgomery); Rahman et al. (bộ B). Ghi attribution NLM/NIH + Shenzhen
  No.3 Hospital khi công bố.
- **Không dữ liệu định danh bệnh nhân**; các bộ trên đã de-identified.
- Ghi rõ giới hạn: nguồn hạn chế, domain confound nhi/người lớn, chưa validate lâm sàng.

---

## 13. Definition of Done cho data pipeline

- [ ] `download.py` tải đủ A + C (và B nếu dùng), idempotent, chạy được cả local lẫn Kaggle.
- [ ] `prepare.py` sinh `manifest.csv` đúng schema, gán nhãn chuẩn, khử trùng, gắn domain.
- [ ] `split.py` chia mức bệnh nhân, giữ Montgomery làm external, PASS mọi assert rò rỉ.
- [ ] `checks.py` xuất báo cáo chất lượng + lưới ảnh mẫu.
- [ ] `datasets.py` trả DataLoader đúng (augment chỉ ở train; normalize ImageNet).
- [ ] Chạy lại cùng seed → cùng split (tái lập).
- [ ] `split_report.md` sẵn sàng dán vào phần Methods của paper.
