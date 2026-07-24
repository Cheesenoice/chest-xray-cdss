# CLAUDE.md — Context & Plan cho dự án tốt nghiệp

> File này là bối cảnh chính của project. Claude Code PHẢI đọc kỹ toàn bộ trước khi
> viết bất kỳ dòng code nào. Nó mô tả: người dùng là ai, hoàn cảnh, đề tài đã chốt,
> chuẩn khoa học bắt buộc, kiến trúc kỹ thuật, cấu trúc repo, lịch thực hiện, và các
> bẫy phải tránh. Khi có mâu thuẫn giữa yêu cầu nhất thời và file này, ưu tiên file này
> và hỏi lại người dùng.

---

## 0. TL;DR cho Claude Code

Ta xây một **hệ thống hỗ trợ sàng lọc & quyết định lâm sàng từ ảnh X-quang ngực bằng
Deep Learning CÓ GIẢI THÍCH (Explainable AI)**. Đây vừa là **sản phẩm demo** để bảo vệ
trước hội đồng tốt nghiệp, vừa là **nghiên cứu thực nghiệm** để viết một paper/preprint.

Trọng tâm nghiên cứu KHÔNG phải "phát minh model mới", mà là một **benchmark có thể tái
lập (reproducible), có giải thích (explainable), và có external validation trung thực**.
Đây chính là research gap khả thi trong 1 tháng và là thứ reviewer y tế đánh giá cao.

Deliverable cuối gồm: (1) code repo sạch, chạy được, tái lập được; (2) model đạt metrics
tốt; (3) web app demo (upload ảnh → chẩn đoán → heatmap Grad-CAM → báo cáo); (4) một
preprint + bản thảo paper.

---

## 1. Bối cảnh người dùng (đọc để hiểu ràng buộc)

- Đây là **đồ án tốt nghiệp đại học**, làm năm 2026, thời đại AI agent.
- Người dùng **vừa học vừa làm**; phần lớn code và logic do AI (Claude Code) hỗ trợ sinh
  ra. => Code phải **rõ ràng, có comment giải thích, dễ đọc để người dùng học được**.
  Không viết code "khôn lỏi" khó hiểu. Ưu tiên dễ hiểu hơn là tối ưu cực đoan.
- **Thời gian: đúng 1 THÁNG (4 tuần).** Mọi quyết định kỹ thuật phải ưu tiên tính khả thi
  trong khung này. Không chọn hướng đòi hỏi kiến trúc mới từ đầu, không 3D volumetric
  segmentation, không pipeline EHR lớn.
- **Tài nguyên tính toán:** Kaggle GPU miễn phí (T4/P100) + server GPU mạnh miễn phí của
  trường. Dataset phải đủ nhỏ để train được trên các tài nguyên này trong vài phút đến
  vài giờ.
- **Mục tiêu kép:**
  1. Có **sản phẩm demo cực tốt + thông số chứng minh** để nộp và bảo vệ hội đồng.
  2. Có **paper/preprint chuẩn chỉnh** để xin học bổng thạc sĩ ở Trung Quốc (CSC) hoặc
     Hàn Quốc (GKS). Không cần venue rank cao — có output công khai, hợp pháp, trích dẫn
     được là đủ.
- **Kỳ vọng thực tế về paper:** trong 1 tháng CHẮC CHẮN nộp + preprint được, nhưng
  KHÔNG kịp có acceptance qua bình duyệt. Vì vậy output bắt buộc = **preprint (arXiv/
  medRxiv) + repo công khai + demo online**; nộp hội nghị/tạp chí là "được thì tốt".
- Mọi thứ phải làm **đúng chuẩn nghiên cứu khoa học** — đây là yêu cầu cứng, xem mục 4.

---

## 2. Đề tài đã chốt

**Tên (VN):** Hệ thống hỗ trợ sàng lọc và quyết định lâm sàng từ ảnh X-quang ngực bằng
học sâu có giải thích.

**Tên (EN, dùng cho paper):** *An Explainable Deep Learning Clinical Decision Support
System for Chest X-ray Screening.*

**Bài toán:** phân loại đa lớp ảnh X-quang ngực. Cấu hình mặc định là **4 lớp**:
`Normal` / `Bacterial pneumonia` / `Viral pneumonia` / `Tuberculosis (TB)`.

Pipeline phải **linh hoạt cấu hình số lớp** (chạy được 3 lớp Normal/Bacterial/Viral, hoặc
4 lớp có TB) qua config, vì lý do khoa học ở mục 4.3.

**Câu chuyện ứng dụng (để trả lời hội đồng):** ở tuyến y tế cơ sở thiếu bác sĩ chẩn đoán
hình ảnh, hệ thống sàng lọc phân tầng — gắn cờ ca nghi ngờ nặng (đỏ) để ưu tiên chuyển
tuyến, kèm heatmap giải thích vùng tổn thương để bác sĩ đối chiếu. Đây là "hỗ trợ quyết
định lâm sàng", không thay thế bác sĩ.

---

## 3. Đóng khung nghiên cứu (research framing / gap)

KHÔNG đóng khung là "ta đề xuất model mới". Đóng khung là **một nghiên cứu benchmark
nghiêm túc**, với 4 trụ cột — đây chính là đóng góp và cũng là những thứ paper y tế
thường thiếu:

1. **Reproducibility:** so sánh 3–4 backbone trong CÙNG điều kiện (cùng split, cùng
   augmentation, cùng schedule), báo cáo **trung bình ± độ lệch chuẩn qua ≥3 seed**.
2. **Explainability:** Grad-CAM (và nếu kịp: Grad-CAM++ hoặc Score-CAM) overlay lên ảnh,
   kiểm tra định tính model có "nhìn" đúng vùng phổi tổn thương không.
3. **External validation:** đánh giá trên **nguồn dữ liệu bị giữ lại hoàn toàn** (held-out
   source) mà model chưa thấy khi train, để đo TRUNG THỰC mức tụt hiệu năng. Đây là điểm
   ăn tiền nhất và là câu trả lời cho "model có học vẹt không".
4. **So sánh với phương pháp truyền thống:** một baseline ML cổ điển (đặc trưng thủ công
   HOG/LBP + SVM hoặc Logistic Regression) để thỏa yêu cầu đề tài "so sánh với phương
   pháp truyền thống".

Một can thiệp cải thiện độ chính xác (chọn 1): class-balanced loss / focal loss, hoặc
test-time augmentation, hoặc ensembling — báo cáo delta hiệu năng so với baseline.

---

## 4. CHUẨN KHOA HỌC BẮT BUỘC (đọc kỹ — vi phạm là hỏng cả paper)

### 4.1. Chống rò rỉ dữ liệu (data leakage) — ưu tiên số 1
- **Chia dữ liệu ở MỨC BỆNH NHÂN/NGUỒN, không ở mức ảnh.** Không được để ảnh của cùng một
  bệnh nhân xuất hiện ở cả train và test. Đây là lỗi kinh điển nhất trong AI ảnh y tế và
  reviewer luôn kiểm tra. Nếu dataset không có patient ID, phải ghi rõ giới hạn này và
  chia theo nguồn ảnh (source-level) ở mức có thể.
- Loại bỏ ảnh trùng lặp (duplicate) trước khi chia.

### 4.2. Chia dữ liệu 3 phần & quy trình đánh giá
- Chia **train / validation / test**. Chỉ tune trên validation. **Test set chỉ đụng đến
  MỘT LẦN** ở cuối cùng, không dùng để chọn model.
- Cố định tỉ lệ chia và ghi vào config; log lại số lượng ảnh mỗi lớp mỗi split.

### 4.3. Cảnh báo domain confound (quan trọng cho cấu hình 4 lớp)
- Bộ Kermany là **X-quang trẻ em** (Quảng Châu); các bộ TB (Shenzhen/Montgomery) là
  **người lớn**. Nếu gộp thành 4 lớp, model CÓ THỂ phân biệt TB với viêm phổi dựa trên
  đặc điểm tuổi/giải phẫu thay vì bệnh lý => confound.
- Cách xử lý (bắt buộc ghi vào paper như một limitation + động lực cho external
  validation): (a) vẫn build 4 lớp cho sản phẩm demo, NHƯNG (b) nêu rõ khác biệt domain,
  và (c) thiết kế external validation theo **held-out source** để phơi bày vấn đề trung
  thực. Ngoài ra nên chạy song song một benchmark 3 lớp SẠCH chỉ trên Kermany
  (Normal/Bacterial/Viral) làm kết quả chính đáng tin nhất.

### 4.4. Báo cáo metrics
- Bắt buộc: **Accuracy, Precision, Recall, F1 (macro), AUC (macro / one-vs-rest)**.
- Với dữ liệu mất cân bằng, thêm **balanced accuracy** và **per-class metrics** (đừng chỉ
  báo accuracy tổng).
- Báo cáo **mean ± std qua ≥3 seed** (ví dụ seed 42, 7, 123).
- Báo cáo **khoảng tin cậy 95% (bootstrap)** cho metric chính.
- Vẽ **confusion matrix** và **ROC curve** cho test set.
- Nếu kịp: **calibration** (reliability diagram + ECE).

### 4.5. Tái lập & đạo đức
- Cố định seed toàn bộ (Python, NumPy, PyTorch, cuDNN deterministic khi có thể).
- Ghi rõ phần cứng, số epoch, thời gian train, phiên bản thư viện (`requirements.txt`
  hoặc `environment.yml` pin version).
- Chỉ dùng dữ liệu công khai đúng license; **trích dẫn paper gốc của mỗi dataset**; ghi
  chú license (một số bộ như HAM10000/ISIC là CC BY-NC — không dùng thương mại).
- Xuất **model card** ngắn (mục đích, dữ liệu, giới hạn, không dùng để tự chẩn đoán).
- Tuyên bố rõ: sản phẩm hỗ trợ, KHÔNG thay thế chẩn đoán của bác sĩ.

---

## 5. Dữ liệu (datasets)

> Slug Kaggle có thể thay đổi — Claude Code phải verify lại trên Kaggle trước khi tải, và
> ưu tiên dùng Kaggle API (`kaggle datasets download`) hoặc `kagglehub`.

**Nguồn chính — viêm phổi (trẻ em):**
- Kermany "Chest X-Ray Images (Pneumonia)". Slug tham khảo:
  `paultimothymooney/chest-xray-pneumonia`. ~5,856 ảnh, lớp Normal/Pneumonia; nhãn
  bacterial vs viral suy ra từ tên file (chuỗi "bacteria"/"virus"). License CC BY 4.0.
  Nhẹ, train nhanh.

**Nguồn cho lớp TB (người lớn) — dùng khi cấu hình 4 lớp:**
- TB Chest X-ray Database. Slug tham khảo:
  `tawsifurrahman/tuberculosis-tb-chest-xray-dataset`, hoặc bộ Shenzhen + Montgomery
  (`kmader/pulmonary-chest-xray-abnormalities`). Ảnh có nhãn TB / normal.

**External validation (held-out source — KHÔNG dùng để train):**
- Chọn một nguồn viêm phổi khác Kermany để test khả năng tổng quát hóa (ví dụ một bộ
  chest X-ray pneumonia/covid khác trên Kaggle), HOẶC với TB: train trên Shenzhen và giữ
  lại toàn bộ Montgomery làm external test (hoặc ngược lại). Ghi rõ chiến lược này.

Backup (nếu muốn đổi hướng — đã cân nhắc, chỉ dùng nếu người dùng yêu cầu): HAM10000 da
liễu; APTOS 2019 võng mạc tiểu đường; MIT-BIH ECG loạn nhịp. **Mặc định KHÔNG đổi**, ta
đi X-quang ngực.

---

## 6. Kiến trúc kỹ thuật

**Ngôn ngữ / framework:** Python 3.10+, PyTorch (ưu tiên) + timm (thư viện backbone
pretrained), scikit-learn (baseline truyền thống + metrics), albumentations (augmentation),
pytorch-grad-cam (Grad-CAM), matplotlib/seaborn (biểu đồ).

**Model family (transfer learning, ImageNet pretrained — train nhanh trên GPU free):**
so sánh tối thiểu:
- `ResNet-18` (baseline nhẹ)
- `DenseNet-121` (đã benchmark tốt trên chest X-ray)
- `EfficientNet-B0` (hiệu quả tham số)
- (tùy chọn) `Swin-Tiny` hoặc `ViT-Small` để có góc Transformer thỏa yêu cầu đề tài.

Ảnh resize 224×224, chuẩn hóa theo mean/std ImageNet. Augmentation vừa phải (flip ngang,
xoay nhẹ, thay đổi độ sáng/tương phản) — KHÔNG augmentation làm méo bệnh lý.

**Training config (đưa vào file config, không hardcode):** optimizer AdamW, lr ~1e-4 với
fine-tuning, scheduler cosine hoặc ReduceLROnPlateau, early stopping theo val macro-F1,
mixed precision (AMP) để nhanh hơn. Xử lý mất cân bằng bằng class weights hoặc
WeightedRandomSampler.

**Explainability:** Grad-CAM trên layer conv cuối; xuất ảnh overlay heatmap. Lưu vài ví
dụ đúng và sai để đưa vào paper.

**Baseline truyền thống:** trích đặc trưng HOG hoặc LBP → SVM/LogReg (scikit-learn). Đánh
giá cùng test set để so sánh.

**Sản phẩm demo (web app):**
- Backend: FastAPI phục vụ inference + sinh Grad-CAM (hoặc gộp luôn trong Streamlit/Gradio
  cho nhanh).
- Frontend: **Streamlit** hoặc **Gradio** (ưu tiên tốc độ) — luồng: upload ảnh → predict
  → hiện % từng lớp → overlay heatmap → cảnh báo phân tầng (đỏ/vàng/xanh) → nút xuất báo
  cáo PDF (reportlab/fpdf) → (tùy chọn) dashboard lịch sử ca.
- Deploy: **Hugging Face Spaces** (miễn phí) để có link chạy live cho hội đồng bấm thử.
  Chuẩn bị cả cách chạy local `README`.

---

## 7. Cấu trúc repo đề xuất

```
chest-xray-cdss/
├── CLAUDE.md                 # file này
├── README.md                 # mô tả, cách cài, cách chạy, kết quả, link demo
├── requirements.txt          # pin version
├── configs/
│   └── default.yaml          # số lớp, đường dẫn data, hyperparams, seed list
├── data/                     # KHÔNG commit dữ liệu; chỉ script tải + .gitignore
│   └── README.md
├── src/
│   ├── data/
│   │   ├── download.py       # tải dataset qua kaggle API
│   │   ├── prepare.py        # gộp nguồn, gán nhãn, loại trùng
│   │   └── split.py          # chia patient/source-level, xuất csv split
│   ├── datasets.py           # Dataset/DataLoader + augmentation
│   ├── models.py             # factory backbone (timm)
│   ├── train.py              # vòng train, log, early stopping, lưu checkpoint
│   ├── evaluate.py           # metrics, CI bootstrap, confusion matrix, ROC
│   ├── explain.py            # Grad-CAM overlay
│   ├── baseline_classical.py # HOG/LBP + SVM
│   └── utils.py              # seed, logging, reproducibility helpers
├── experiments/
│   └── run_all.py            # chạy nhiều backbone × nhiều seed, gom kết quả
├── results/                  # bảng metrics, hình, log (commit được)
├── app/
│   ├── app.py                # Streamlit/Gradio demo
│   └── report.py             # sinh PDF báo cáo
├── notebooks/                # EDA, thử nghiệm nhanh
├── model_card.md
└── paper/
    ├── outline.md            # khung IMRaD (xem mục 10)
    └── figures/
```

Nguyên tắc: **không commit dữ liệu và checkpoint nặng** (dùng `.gitignore`); commit config,
code, kết quả bảng/hình, log.

---

## 8. Thông số mục tiêu (để biết thế nào là "tốt")

Dựa trên literature (dùng làm mốc, không phóng đại):
- Viêm phổi nhị phân/đa lớp trên Kermany: accuracy ~90–95%, AUC ~0.95–0.97.
- Nếu chia đúng mức bệnh nhân + external validation, **kỳ vọng con số nội bộ cao và con
  số external THẤP hơn** (ví dụ AUC nội bộ ~0.96, external ~0.85–0.90). ĐÂY LÀ KẾT QUẢ
  TRUNG THỰC VÀ ĐÁNG KHEN — không được "chỉnh" cho đẹp.
- Nếu best model < 85% accuracy hoặc AUC < 0.90 ở nội bộ => nghi có lỗi (leakage,
  augmentation sai, nhãn sai) => debug, KHÔNG đổi đề tài.

Cảnh báo: các con số ~99% trong nhiều paper thường do split rò rỉ / ảnh trùng. Không đặt
mục tiêu đó và không claim nếu chia đúng.

---

## 9. Lịch 4 tuần (định hướng cho cách làm việc)

**Tuần 1 — Data + baseline + repo hygiene.**
Dựng repo, cấu hình, seed. Tải Kermady (+ TB). EDA. Chia train/val/test ĐÚNG chuẩn mục 4.
Viết Dataset/DataLoader. Train xong một baseline DenseNet-121 chạy end-to-end ra kết quả
đầu tiên. Khởi tạo Git + README + requirements.

**Tuần 2 — Benchmark + tuning + explainability.**
Train 3–4 backbone × 3 seed cùng điều kiện. Log đủ metrics + std. Thêm Grad-CAM. Viết
baseline truyền thống (HOG/LBP+SVM). Bắt đầu soạn liên hệ giáo sư song song.

**Tuần 3 — External validation + demo product.**
Chạy best model trên nguồn held-out, đo mức tụt. Thử 1 can thiệp cải thiện, báo cáo delta.
Build web app (upload→predict→Grad-CAM→report), deploy Hugging Face Spaces. Quay video demo.

**Tuần 4 — Viết + preprint + nộp.**
Viết paper IMRaD (AI hỗ trợ, nhưng NGƯỜI KIỂM CHỨNG mọi số liệu & trích dẫn). Post preprint
arXiv/medRxiv. Nộp workshop/tạp chí nếu kịp. Freeze + release repo (tag + seed + env). Làm
slide bảo vệ + bảng thông số kỹ thuật.

---

## 10. Khung paper (IMRaD) để tham chiếu

- **Title / Abstract:** nêu bài toán, đóng góp (benchmark tái lập + XAI + external
  validation), con số chính.
- **Introduction:** bối cảnh AI cho chest X-ray, khoảng trống (over-reliance vào 1 dataset,
  thiếu external validation, thiếu giải thích), đóng góp của ta.
- **Related work:** ngắn gọn, các CNN/Transformer cho chest X-ray.
- **Methods:** dataset + license + chiến lược chia (nhấn mạnh mức bệnh nhân/nguồn), các
  backbone, cấu hình train, Grad-CAM, baseline truyền thống, metrics + CI.
- **Results:** bảng đầy đủ (mean±std, CI), confusion matrix, ROC, Grad-CAM figures,
  external validation, delta của can thiệp cải thiện.
- **Discussion:** ý nghĩa lâm sàng, mức tụt external (thảo luận trung thực), so với
  benchmark công bố.
- **Limitations:** domain confound trẻ em/người lớn, dữ liệu 1–2 nguồn, chưa thử nghiệm
  lâm sàng thật.
- **Conclusion + Reproducibility statement** (link code, seed, env).

Venue: preprint arXiv (eess.IV/cs.CV) hoặc medRxiv (bắt buộc). Nộp thêm (tùy chọn):
workshop vệ tinh MICCAI (vd AMAI), IEEE ICHI/Healthcom, hoặc tạp chí OA có index
DOAJ/Scopus. **Tránh tạp chí predatory** (kiểm tra DOAJ/Scopus/WoS, dùng Think-Check-
Submit; nghi ngờ mọi lời hứa "nhận trong vài ngày").

---

## 11. Cách Claude Code nên làm việc (conventions)

- **Làm tăng dần, chạy được từng bước.** Ưu tiên có một pipeline end-to-end chạy ra kết
  quả sớm (dù nhỏ), rồi mới mở rộng. Không viết một khối khổng lồ rồi mới chạy.
- **Test trên tập nhỏ trước** (vài trăm ảnh / 1–2 epoch) để bắt lỗi nhanh, rồi mới chạy đủ.
- **Code dễ đọc, có docstring + comment tiếng Việt/Anh** để người dùng học được. Giải
  thích "tại sao", không chỉ "làm gì".
- **Mọi hằng số/hyperparam vào `configs/`**, không hardcode rải rác.
- **Seed & reproducibility là mặc định**, không phải tính năng thêm.
- **In log rõ ràng:** kích thước mỗi split, số ảnh mỗi lớp, metric mỗi epoch.
- Khi một bước có rủi ro (dễ leakage, dễ sai nhãn), **dừng lại giải thích cho người dùng**
  trước khi chạy tiếp.
- Trước khi tải data hay dùng slug Kaggle, **verify slug còn tồn tại**; nếu API cần token,
  hướng dẫn người dùng đặt `kaggle.json`.
- Ưu tiên thư viện phổ biến ổn định (timm, pytorch-grad-cam, albumentations) hơn tự viết
  lại từ đầu.

---

## 12. Bẫy phải tránh (nhắc lại, ngắn gọn)

- Chia dữ liệu ở mức ảnh (rò rỉ) → luôn chia mức bệnh nhân/nguồn.
- Dùng test set để tune → chỉ đụng test một lần.
- Ảnh trùng lặp giữa các split.
- Gộp trẻ em (viêm phổi) + người lớn (TB) mà không nêu confound.
- Claim accuracy ~99% kiểu leaky.
- 3D BraTS segmentation / MIMIC credentialed / federated learning phức tạp — KHÔNG làm
  trong 1 tháng.
- Tạp chí predatory.
- Chỉ báo accuracy tổng trên dữ liệu mất cân bằng.

---

## 13. Định nghĩa "xong" (Definition of Done)

- [ ] Pipeline tái lập: clone repo + đặt data + chạy 1 lệnh → ra kết quả khớp.
- [ ] Bảng metrics đầy đủ (mean±std, CI) + confusion matrix + ROC + Grad-CAM figures.
- [ ] External validation có số liệu và thảo luận trung thực.
- [ ] Baseline truyền thống để so sánh.
- [ ] Web app demo chạy được + deploy Hugging Face Spaces + video demo.
- [ ] Model card + reproducibility statement.
- [ ] Preprint sẵn sàng post + bản thảo paper IMRaD.
- [ ] Slide bảo vệ + bảng thông số kỹ thuật cho hội đồng.
