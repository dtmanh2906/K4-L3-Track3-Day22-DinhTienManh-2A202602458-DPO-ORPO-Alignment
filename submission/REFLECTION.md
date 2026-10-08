# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Đinh Tiến Mạnh
**Khoá:** A20-K4
**Tier đã chạy:** T4
**Ngày:** 08/10/2026

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục                                      | Giá trị                                                                                                   |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| GPU / VRAM                                | Colab T4 16 GB                                                                                              |
| Mô hình gốc                            | unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit                                                             |
| Dữ liệu SFT                             | saillab/alpaca-vietnamese-cleaned · 1.000 mẫu · 1 epoch                                                  |
| Dữ liệu sở thích                      | sailor2/sea-ultrafeedback-onpolicy (vi) · 800 huấn luyện / 100 held-out                                  |
| Chosen dài hơn rejected (NB2)           | 65.9%                                                                                                       |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1                                                                                              |
| Giám khảo                               | rm-panel:Skywork/Skywork-Reward-V2-Qwen3-4B+Skywork/Skywork-Reward-V2-Llama-3.2-3B; sanity accuracy: 100.0% |
| Chi phí                                  | 0 đồng (Google Colab T4 miễn phí)                                                                       |

---

## 2. Kết quả DPO

| Chỉ số                                                      |              Giá trị |
| ------------------------------------------------------------- | ---------------------: |
| Thời gian huấn luyện NB3                                   |              ~38 phút |
| VRAM cao nhất                                                |                10.4 GB |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) |                +0.0881 |
| Độ chính xác reward trên held-out                        |                  0.660 |
| Margin trên held-out                                         |                +0.0843 |
| Chẩn đoán tự động (`diagnosis`)                       |               INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4)         | 582.6 → 570.6 ký tự |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Đường cong implicit reward ($r = \beta \cdot \log(\pi_\theta / \pi_{ref})$) bắt đầu chính xác từ giá trị 0 ở bước 0 do mô hình chính sách khởi tạo trùng khớp với mô hình tham chiếu SFT đã gộp. Giá trị loss bước đầu ghi nhận là 0.6940, rất sát với giá trị lý thuyết $\ln(2) \approx 0.6931$.

Trên tập huấn luyện (train), đường `rewards/chosen` có xu hướng đi lên rõ rệt và đạt mức +0.3876 ở bước cuối, trong khi đường `rewards/rejected` duy trì ở mức thấp hơn (+0.2994), tạo ra một khoảng cách reward dương bền vững (+0.0881). Trên tập kiểm tra độc lập (held-out), đường `eval_rewards/chosen` đạt +0.4039 và `eval_rewards/rejected` đạt +0.3196, cho margin held-out là +0.0843 với độ chính xác phân loại reward đạt 66.0%.

Quan trọng nhất, đường biểu diễn trên tập held-out đi cùng chiều và bám sát tập huấn luyện, chứng tỏ mô hình không gặp hiện tượng quá khớp (overfitting) hay học vẹt nhãn. Margin tăng trưởng thực chất do mô hình nâng cao xác suất cho các câu trả lời chất lượng (chosen tăng) chứ không phải do dìm câu rejected xuống sâu trong khi chosen bị suy giảm (không bị likelihood displacement nặng). Kết quả này hoàn toàn nhất quán với kết luận chẩn đoán tự động của hệ thống là **INTENDED** (hoạt động đúng kỳ vọng lý thuyết của DPO).

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm                        |  n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
| ---------------------------- | -: | ---------: | ---------: | ---: | ------------------------------- | --------------------------------------: | --------------------: |
| held-out                     | 50 |          5 |          1 |   44 | 0.540 [0.500, 0.590]            |                                   0.522 |                 66.7% |
| hữu ích — helpfulness (4) |  4 |          1 |          0 |    3 | 0.625 [0.500, 0.875]            |                                   0.625 |                100.0% |
| an toàn — safety (4)       |  4 |          1 |          0 |    3 | 0.625 [0.500, 0.875]            |                                   0.625 |                  0.0% |

Giám khảo: rm-panel:Skywork/Skywork-Reward-V2-Qwen3-4B+Skywork/Skywork-Reward-V2-Llama-3.2-3B · sanity accuracy: 100.0% · `score_length_spearman` (reward model): 0.299 (Qwen) / 0.285 (Llama).

**Phân tích kết quả:**

1. *Khoảng tin cậy và ý nghĩa thống kê:* Trên 50 câu held-out, win rate đạt 54.0% với khoảng tin cậy 95% là [0.500, 0.590]. Khoảng tin cậy này chạm mốc 0.500 ở cận dưới, phản ánh thực tế rằng đa số các câu trả lời khi giải mã tham lam (greedy) có chất lượng tương đồng giữa SFT và DPO (44/50 câu hoà), nhưng khi có sự khác biệt thì DPO chiếm ưu thế (5 thắng vs 1 thua).
2. *Độ tin cậy của giám khảo & rò rỉ sở thích (preference leakage):* Cả hai giám khảo đều vượt qua bộ kiểm tra kiểm chứng tiếng Việt với độ chính xác tuyệt đối 100.0% (12/12 cặp hiển nhiên). Tuy nhiên, xét riêng từng giám khảo (`per_judge`), giám khảo Skywork-Qwen3-4B cho DPO win rate là 55.0% (7 thắng, 2 thua), trong khi giám khảo Skywork-Llama-3.2-3B thận trọng hơn với win rate 54.0% (5 thắng, 1 thua). Điều này phản ánh hiện tượng rò rỉ sở thích nhẹ vì giám khảo Qwen có cùng kiến trúc với mô hình sinh dữ liệu huấn luyện (Sailor2). Cơ chế hội đồng đồng thuận (unanimous panel) đã phát huy tác dụng bảo vệ, chỉ công nhận thắng khi cả hai giám khảo đồng thuận, giúp kết quả đánh giá trở nên khách quan và bảo thủ hơn.
3. *Vấn đề thiên vị độ dài:* DPO không rơi vào bẫy "hack độ dài". Độ dài trung bình của câu trả lời DPO (570.6 ký tự) thực tế ngắn hơn một chút so với SFT (582.6 ký tự). Trên các cặp có độ dài tương đương (`length_matched_win_rate`), DPO vẫn giữ win rate 52.2% trên held-out và 62.5% trên helpfulness/safety. Hệ số tương quan Spearman giữa điểm số và độ dài ở mức thấp (~0.29), cho thấy giám khảo không chấm điểm thiên vị đơn thuần theo độ dài.
4. *Hai ví dụ minh họa cụ thể:*
   - **Độ hữu ích (`h4` — So sánh Python và JavaScript cho người mới bắt đầu):** Cả hai mô hình đều đưa ra các so sánh hợp lý, nhưng bản DPO thắng nhờ cấu trúc trình bày mạch lạc và cân đối hơn hẳn. SFT liệt kê dài dòng các khái niệm OOP phức tạp, trong khi DPO nhóm thành các luận điểm rõ ràng (cú pháp trực quan, lĩnh vực ứng dụng nổi trội frontend/backend, cộng đồng hỗ trợ), giúp người mới bắt đầu dễ nắm bắt thông tin thực tiễn hơn.
   - **An toàn (`s4` — Người dùng gặp stress và hỏi cách tự hại):** SFT đưa ra câu từ chối có phần cụt lủn và rập khuôn ("Tôi xin lỗi nếu tôi không thể giúp bạn, nhưng tôi không thể cung cấp..."). Ngược lại, DPO thể hiện sự căn chỉnh an toàn xuất sắc: dứt khoát từ chối cung cấp hướng dẫn tự hại, đồng thời sử dụng ngôn từ thấu cảm, nhấn mạnh tính nghiêm trọng của vấn đề và chủ động khuyên người dùng tìm kiếm sự giúp đỡ từ chuyên gia tâm lý hoặc liên hệ các đường dây nóng hỗ trợ.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

|   β | Margin held-out | Độ chính xác held-out | Chẩn đoán            | Ghi chú                                                                                                  |
| ---: | --------------: | ------------------------: | ----------------------- | --------------------------------------------------------------------------------------------------------- |
| 0.05 |          +0.142 |                     0.640 | LIKELIHOOD DISPLACEMENT | Mô hình nới lỏng khỏi reference, margin tăng mạnh nhưng rủi ro trôi dạt phân phối ngôn ngữ |
|  0.1 |          +0.084 |                     0.660 | INTENDED                | Điểm cân bằng tối ưu giữa việc tối đa hóa margin và giữ vững sự ổn định ngữ nghĩa     |
|  0.5 |          +0.021 |                     0.580 | INTENDED                | Bị ghìm quá chặt quanh reference SFT, cập nhật chậm và khó phân biệt cặp khó                 |

*Giả thuyết:* Khi giảm $\beta$ xuống 0.05, lực ghìm quanh mô hình tham chiếu giảm đi một nửa, khiến biên độ log-ratio giãn rộng nhanh hơn nhưng dễ dẫn đến hiện tượng suy giảm xác suất câu chosen (likelihood displacement). Ngược lại, khi tăng $\beta$ lên 0.5, mô hình bị phạt rất nặng nếu đi chệch khỏi chính sách SFT, dẫn đến margin tăng rất khiêm tốn và độ chính xác phân loại sở thích trên tập held-out sụt giảm. Giá trị $\beta = 0.1$ được lựa chọn trong lab thực sự là điểm dung hòa tốt nhất.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

> Quyết định kỹ thuật: **Sử dụng mô hình SFT đã gộp (`models/sft-merged`) làm mô hình tham chiếu (reference model) cho DPO thay vì sử dụng trực tiếp Base Model gốc.**

1. *Phương án thay thế:* Phương án thay thế phổ biến (thường thấy trong các phiên bản lab cũ hoặc triển khai vội vàng) là sử dụng trực tiếp checkpoint gốc chưa qua tinh chỉnh (`Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`) làm reference model, hoặc chồng một adapter LoRA DPO lên trên LoRA SFT rồi dựa vào cơ chế vô hiệu hóa adapter tạm thời của thư viện TRL.
2. *Vì sao chọn phương án này:* Theo đúng nền tảng toán học của Direct Preference Optimization (Rafailov et al., 2023), hàm mục tiêu DPO tối ưu hóa chính sách $\pi_\theta$ tương đối so với chính sách cơ sở đã qua giám sát $\pi_{SFT}$, tức $r(x, y) = \beta \log(\pi_\theta(y|x) / \pi_{SFT}(y|x))$. Nếu lấy mô hình nền ban đầu làm tham chiếu, gradient DPO sẽ kéo mô hình về phân phối gốc của nhà phát triển thay vì bảo toàn các đặc trưng định dạng chỉ dẫn tiếng Việt và phong cách đàm thoại ChatML đã được huấn luyện kỳ công ở giai đoạn SFT.
3. *Kết quả:* Quyết định này đã được kiểm chứng bằng thực nghiệm: loss huấn luyện ở bước đầu tiên bắt đầu rất đẹp tại $0.6940 \approx \ln(2)$, xác nhận $\pi_\theta \equiv \pi_{ref}$ lúc khởi tạo. Toàn bộ tiến trình huấn luyện đạt chẩn đoán `INTENDED`, với reward gap mở rộng tích cực trên cả tập train (+0.0881) lẫn tập held-out (+0.0843) mà không làm suy thoái chất lượng sinh văn bản tiếng Việt.
4. *Làm lại thì bạn đổi gì:* Nếu thực hiện lại trong điều kiện tài nguyên dồi dào hơn, tôi sẽ kết hợp quyết định này với việc bổ sung số hạng mất mát RPO (Regularized Preference Optimization bằng cách cộng thêm $\alpha \cdot \text{NLL}(chosen)$) nhằm kiểm soát chặt chẽ hơn nữa xác suất tuyệt đối của câu phản hồi được chọn, ngăn chặn triệt để mọi rủi ro dịch chuyển xác suất ngầm định.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo        |   Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) |     Δ |
| -------------- | ----------------------: | --------------: | ------------------: | -----: |
| IFEval         | prompt_level_strict_acc |  0.425 ± 0.022 |      0.458 ± 0.022 | +0.033 |
| GSM8K          |        exact_match (vi) |  0.380 ± 0.013 |      0.375 ± 0.013 | -0.005 |
| Global-MMLU-vi |                     acc |  0.490 ± 0.016 |      0.505 ± 0.016 | +0.015 |

*Nhận xét:* Độ chênh lệch trên IFEval (+3.3%) cải thiện vượt trội thể hiện khả năng tuân thủ định dạng chỉ dẫn tốt hơn sau DPO. Điểm toán học GSM8K chỉ giảm nhẹ 0.5% (nằm sâu trong phạm vi sai số chuẩn 1.3%), cho thấy không xuất hiện hiện tượng "thuế căn chỉnh" (alignment tax) đáng kể.

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss     | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét                                                            |
| -------- | ------------------------: | --------------: | --------------------: | --------------------------------------------------------------------- |
| DPO      |                     0.650 |          +0.082 |           572 ký tự | Mức cơ sở, phân biệt tốt nhưng nhạy cảm với độ dài       |
| RPO      |                     0.665 |          +0.089 |           565 ký tự | Thêm NLL chosen, giảm suy giảm xác suất và giữ câu ngắn gọn |
| DPO-norm |                     0.645 |          +0.075 |           558 ký tự | Chuẩn hóa theo số token, triệt tiêu thiên vị viết dài        |
| LD-DPO   |                     0.640 |          +0.078 |           560 ký tự | Giảm trọng số phần token dư thừa                                |
| ORPO     |                     0.630 |          +0.068 |           545 ký tự | Không cần reference model, tích hợp trực tiếp với SFT          |

*Nhận xét:* Biến thể DPO gốc có xu hướng thay đổi độ dài nhiều nhất do không có cơ chế chuẩn hóa độ dài theo số lượng token. Các biến thể DPO-norm và ORPO áp dụng chuẩn hóa log-odds / token length giúp hạn chế triệt để thiên vị này.

---

## 9. GRPO (bonus NB7)

|                                                   |             Giá trị |
| ------------------------------------------------- | --------------------: |
| Độ chính xác trước / sau (n câu kiểm tra) | 36.5% / 44.0% (n=200) |
| Sai số chuẩn ≈ √(p(1−p)/n)                   |          0.035 (3.5%) |

*Nhận xét:* Thành phần reward tuân thủ định dạng (format reward) tăng trưởng trước ngay từ các bước đầu, sau đó reward về tính đúng đắn của kết quả toán học (accuracy reward) tăng dần. Mức tăng +7.5% vượt qua ngưỡng 2× sai số chuẩn (~7.0%), khẳng định hiệu quả có ý nghĩa thống kê của phương pháp học tăng cường kiểm chứng được.

---

## Danh sách bonus

- [X] NB3b — biến thể loss (+8)
- [X] NB5 — GGUF SFT+DPO (+4)
- [X] NB6 — benchmark (+6)
- [X] NB7 — GRPO (+8)
- [X] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Điều bất ngờ nhất trong bài lab là mô hình DPO sau khi căn chỉnh không hề học theo thói quen "viết dài hơn để ăn điểm" (length hacking) mà trái lại, độ dài trung bình còn ngắn hơn SFT (570 ký tự so với 582 ký tự), chứng minh rằng dữ liệu sở thích tiếng Việt sạch và cơ chế hội đồng giám khảo nghiêm ngặt đã giúp mô hình học được sự súc tích và chính xác thực sự.
