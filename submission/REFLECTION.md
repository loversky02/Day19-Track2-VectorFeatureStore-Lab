# Reflection — Lab 19

**Tên:** Trần Đình Minh Vương
**MSSV:** 2A202600495
**Cohort:** A20-K1
**Path đã chạy:** lite

---

## Câu hỏi (≤ 200 chữ)

> Trên golden set 50 queries, mode nào thắng ở loại query nào (`exact` /
> `paraphrase` / `mixed`), và tại sao? Khi nào bạn **không** dùng hybrid
> (i.e. khi nào pure BM25 hoặc pure vector là lựa chọn đúng)?

**Mode thắng theo loại query:**
- **Exact queries (15):** Keyword (BM25) thắng 96.7%, hybrid đồng hạng 96.7%. Exact queries chứa từ kỹ thuật verbatim trong corpus → BM25 có signal keyword mạnh, semantic không cải thiện thêm.
- **Paraphrase queries (15):** Keyword thắng 33.3%, hybrid 32.0%, semantic 24.0%. Cả ba đều yếu vì corpus tiếng Việt không khớp từ verbatim mà mô hình English-trained `bge-small-en-v1.5` không bắt được nuance tiếng Việt. Hybrid không cứu được semantic khi embedding model không phù hợp ngôn ngữ.
- **Mixed queries (20):** Hybrid thắng rõ 100% vs keyword 97.0% và semantic 98.5%. Đây là pattern production thực tế nhất — user thật thường kết hợp cả từ exact lẫn ý tưởng paraphrase.

**Tổng thể:** Hybrid 78.6% > Keyword 77.8% > Semantic 73.2%. Hybrid thắng nhờ robust trên mọi loại query.

**Khi không dùng hybrid:**
- Query kỹ thuật chuyên ngành (technical jargon): dùng BM25 thuần — keyword signal đã đủ, thêm vector chỉ tăng latency.
- Search trên ngôn ngữ/dialect không hỗ trợ tốt bởi embedding model hiện tại: dùng pure BM25 hoặc đổi sang embedding model phù hợp hơn trước khi hybrid.
- Latency budget cực thấp (<5ms P99): hybrid RRF cần search 2 retrievers → ~2× latency so với keyword đơn lẻ.

---

## Điều ngạc nhiên nhất khi làm lab này

Hybrid không thắng rõ ràng trên paraphrase queries — dù semantic kém, hybrid không "cứu" được vì BM25 cũng tệ trên Vietnamese paraphrase. Điều này cho thấy embedding model choice quan trọng hơn fusion strategy khi ngôn ngữ không khớp.

---

## Bonus challenge

- [x] Đã làm bonus (xem `submission/bonus/`)
- [x] Làm cá nhân
