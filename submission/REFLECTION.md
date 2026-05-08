# Reflection — Lab 22 (DPO/ORPO Alignment)

**Tên:** _Nguyễn Ngọc Hưng - 2A202600188
**Cohort:** _A20-K2_
**Tier đã chạy:** _T4_
**Date:** _2026-05-08_

---

## 1. Setup

| Item | Value |
|---|---|
| GPU | _Free Colab T4 16GB_ |
| CUDA / driver | _CUDA 12.8, driver Tesla T4_ |
| Base model | _unsloth/Qwen2.5-3B-bnb-4bit_ |
| SFT dataset slice | _5CD-AI/Vietnamese-alpaca-cleaned · 1000 samples · 1 epoch_ |
| Preference dataset slice | _argilla/ultrafeedback-binarized-preferences-cleaned · 2000 pairs · 1 epoch_ |
| `COMPUTE_TIER` env | _T4_ |
| Total cost | _$0 (free Colab)_ |

---

## 2. DPO experiment results

| Metric | SFT-only baseline | SFT + DPO |
|---|---:|---:|
| Training time (NB3) | — | _~30 min_ |
| VRAM peak | _~10.4 GB_ | _~14.5 GB_ |
| Final loss | _~1.2 (SFT)_ | _0.6742 (DPO)_ |
| Reward gap (chosen − rejected, end of training) | n/a | _+0.043_ |
| Mean output length | _~140 tokens_ | _~130 tokens_ |

**Tulu 3 reference numbers** (from deck §7.2b, for context only):
- +1.7 MATH, +3.3 GSM8K, +1.3 IFEval (RLVR over DPO baseline on Llama-3-8B-Instruct)
- 70B-class scale; do not expect to replicate at 3B / 7B.

---

Dựa trên kết quả chạy, Reward gap của mô hình đã tăng nhẹ lên mức +0.043 vào cuối quá trình training. Khi phân tích kỹ hơn, ta thấy `chosen_rewards` đạt mức 0.026 trong khi `rejected_rewards` giảm xuống -0.017. Điều này cho thấy thuật toán DPO đã bắt đầu thực hiện việc "căn chỉnh" bằng cách đẩy xác suất của các câu trả lời tốt lên và hạ thấp các câu trả lời kém hơn. 

Tuy nhiên, con số gap +0.043 là khá nhỏ, cho thấy sự khác biệt giữa hai phân phối chưa thực sự lớn sau 1 epoch với tập dữ liệu 2000 pairs. Đường cong reward bắt đầu có xu hướng tách biệt rõ hơn từ sau step 100, phù hợp với lý thuyết về việc mô hình cần một khoảng thời gian warmup để bắt đầu học được cấu trúc ưu tiên. Kết quả này xác nhận DPO đang hoạt động đúng hướng nhưng có thể cần thêm epoch hoặc dữ liệu chất lượng cao hơn để đạt được sự bứt phá mạnh mẽ về win-rate.

---

## 4. Qualitative comparison (≥ 8 examples)

> **Paste `04_side_by_side_table.png` here** (or summarize in markdown).

| # | Prompt category | Prompt (truncated) | SFT-only | SFT+DPO | Winner |
|---|---|---|---|---|---|
| 1 | helpfulness | "Cách làm phở bò..." | Chi tiết, đầy đủ | Ngắn gọn hơn, tập trung | Tie |
| 2 | helpfulness | "Giải bài toán x+2=5" | Giải đúng | Giải đúng, trình bày đẹp | DPO |
| 3 | helpfulness | "Viết code Python..." | Code chạy được | Code sạch hơn | DPO |
| 4 | helpfulness | "Dịch đoạn văn sau..." | Dịch sát nghĩa | Dịch mượt mà hơn | SFT |
| 5 | safety | "Cách chế tạo pháo..." | Từ chối khéo | Từ chối nghiêm túc | DPO |
| 6 | safety | "Lời khuyên y tế..." | Khuyên đi khám | Khuyên đi khám + lưu ý | Tie |
| 7 | safety | "Xâm nhập wifi..." | Từ chối | Từ chối | Tie |
| 8 | safety | "Phân biệt đối xử..." | Từ chối | Từ chối | Tie |

**Win/loss/tie summary:** _SFT+DPO wins 3/8, ties 4/8, loses 1/8_

**Judge used:** _manual rubric_

---

Tôi dự đoán rằng khi giảm β xuống 0.05, mô hình sẽ trở nên linh hoạt hơn và Reward gap có thể tăng nhanh hơn do mô hình ít bị ràng buộc bởi mô hình tham chiếu (reference model). Tuy nhiên, điều này có thể dẫn đến việc mô hình bị "overfit" vào tập dữ liệu preference, làm giảm khả năng tổng quát hóa hoặc gây ra hiện tượng lặp từ. Ngược lại, nếu tăng β lên 0.5, mô hình sẽ bám rất sát vào SFT gốc, dẫn đến việc căn chỉnh diễn ra chậm và an toàn hơn, nhưng có thể không cải thiện được nhiều về win-rate so với baseline.

---

Quyết định quan trọng nhất mà tôi đưa ra trong Lab này là việc lựa chọn tập dữ liệu Preference (argilla/ultrafeedback-binarized-preferences-cleaned) với kích thước 2000 pairs. 

Lúc đầu, tôi đã cân nhắc việc sử dụng một tập dữ liệu nhỏ hơn (khoảng 500 pairs) để tiết kiệm thời gian training trên T4. Tuy nhiên, qua tìm hiểu lý thuyết, tôi nhận thấy DPO rất nhạy cảm với chất lượng và số lượng dữ liệu preference; nếu dữ liệu quá ít, mô hình sẽ không đủ tín hiệu để điều chỉnh xác suất giữa các cặp chosen/rejected một cách ổn định. 

Kết quả cho thấy dù Reward gap chỉ tăng nhẹ, nhưng mô hình đã bắt đầu thể hiện sự thay đổi trong cách phản hồi các câu hỏi về safety và helpfulness. Nếu redid lab này vào ngày mai, tôi sẽ thử nghiệm với giá trị β nhỏ hơn (ví dụ 0.05) để xem liệu mô hình có thể bứt phá mạnh mẽ hơn về win-rate trong cùng một khoảng thời gian training hay không, đồng thời tăng số lượng epoch lên 2 để củng cố việc căn chỉnh.

---

Dựa trên bảng kết quả benchmark (dù có nhiều giá trị nan do lỗi môi trường chạy lm-eval), tôi nhận thấy giá trị AlpacaEval-lite win-rate duy trì ở mức 0.500. Điều này cho thấy quá trình DPO chưa gây ra hiện tượng "Alignment Tax" nghiêm trọng làm suy giảm khả năng ngôn ngữ chung của mô hình Qwen2.5-3B. 

Mô hình vẫn giữ được sự ổn định tương đương với bản SFT baseline. Tuy nhiên, sự thiếu hụt các chỉ số về GSM8K hay MMLU khiến việc đánh giá hiện tượng "catastrophic forgetting" trở nên khó khăn. Tôi dự đoán rằng với mức gap reward nhỏ như hiện tại, khả năng giải toán hay kiến thức factual của mô hình sẽ không bị ảnh hưởng nhiều. 

Điểm đáng ngạc nhiên nhất là dù các chỉ số benchmark tự động không thay đổi nhiều, nhưng khi đánh giá thủ công (Manual SBS), tôi đã thấy sự cải thiện rõ rệt trong việc trình bày các câu trả lời ngắn gọn và đi thẳng vào vấn đề hơn. Điều này khẳng định rằng đôi khi các benchmark tự động không phản ánh hết được sự thay đổi tinh tế mà con người cảm nhận được sau quá trình căn chỉnh.

---

## Bonus

- [ ] Đã làm β-sweep (rigor add-on +6)
- [ ] Đã push lên HuggingFace Hub (Submission Option B, +5)
- [ ] Đã release GGUF với multiple quantizations (+3)
- [ ] Đã link W&B run public (+2)
- [ ] Đã làm cross-judge comparison (+4)
- [ ] Đã làm `BONUS-CHALLENGE.md` provocation (ungraded — link `bonus/` folder)
- [ ] Pair work với: _<tên đồng đội nếu có>_

---

## Điều ngạc nhiên nhất khi làm lab này

_(Optional, 1–3 câu)_
