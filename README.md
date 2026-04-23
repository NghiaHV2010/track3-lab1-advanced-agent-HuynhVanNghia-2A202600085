# Lab 16 — Reflexion Agent Scaffold

Repo này cung cấp một khung sườn (scaffold) để xây dựng và đánh giá **Reflexion Agent**.

## 1. Mục tiêu của Repo
- Repo hiện tại đang sử dụng **Mock Data** (`mock_runtime.py`) để giả lập phản hồi từ LLM.
- Mục đích giúp học viên hiểu rõ về **flow**, các bước **loop**, cách thức hoạt động của cơ chế phản chiếu (reflection) và cách đánh giá (evaluation) mà không tốn chi phí API ban đầu.

## 2. Nhiệm vụ của Học viên
Học viên cần thực hiện các bước sau để hoàn thành bài lab:
1. **Xây dựng Agent thật**: Thay thế phần mock bằng việc gọi LLM thật (sử dụng Local LLM như Ollama, vLLM hoặc các Simple LLM API như OpenAI, Gemini).
2. **Chạy Benchmark thực tế**: Chạy đánh giá trên ít nhất **100 mẫu dữ liệu thật** từ bộ dataset **HotpotQA**.
3. **Định dạng báo cáo**: Kết quả chạy phải đảm bảo xuất ra file report (`report.json` và `report.md`) có cùng định dạng (format) với code gốc để có thể chạy được công cụ chấm điểm tự động.
4. **Tính toán Token thực tế**: Thay vì dùng số ước tính, học viên phải cài đặt logic tính toán lượng token tiêu thụ thực tế từ phản hồi của API.

## 3. Cách chạy Lab (Scaffold)
```bash
# Cài đặt môi trường
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Chạy benchmark với mock mode (không tốn API)
python run_benchmark.py --mode mock --dataset data/hotpot_mini.json --out-dir outputs/sample_run

# Chạy benchmark với OpenAI API
# Yêu cầu: export/set OPENAI_API_KEY trước khi chạy
python run_benchmark.py --mode openai --model gpt-4o-mini --dataset data/hotpot_mini.json --out-dir outputs/openai_run

# Chuẩn bị HotpotQA 100+ mẫu (mặc định 110 mẫu) từ Hugging Face
python scripts/prepare_hotpotqa_subset.py --count 110 --out data/hotpot_110.json

# Chạy benchmark full với dataset 100+ mẫu
python run_benchmark.py --mode openai --model gpt-4o-mini --dataset data/hotpot_110.json --out-dir outputs/openai_full_110

# Chạy chấm điểm tự động
python autograde.py --report-path outputs/sample_run/report.json
```

## 4. Tiêu chí chấm điểm (Rubric)
- **80% số điểm (80 điểm)**: Hoàn thiện đúng và đủ luồng (flow) cho Reflexion Agent, chạy thành công với LLM thật và dataset thật.
- **20% số điểm (20 điểm)**: Thực hiện thêm ít nhất một trong các phần **Bonus** được nhắc đến trong mã nguồn (ví dụ: `structured_evaluator`, `reflection_memory`, `adaptive_max_attempts`, `memory_compression`, v.v. - xem chi tiết tại `autograde.py`).

## Thành phần mã nguồn
- `src/reflexion_lab/schemas.py`: Định nghĩa các kiểu dữ liệu trace, record.
- `src/reflexion_lab/prompts.py`: Nơi chứa các template prompt cho Actor, Evaluator và Reflector.
- `src/reflexion_lab/mock_runtime.py`: (Cần thay thế) Logic giả lập phản hồi LLM.
- `src/reflexion_lab/agents.py`: Cấu trúc chính của ReAct và Reflexion Agent.
- `src/reflexion_lab/reporting.py`: Logic xuất báo cáo benchmark.
- `run_benchmark.py`: Script chính để chạy đánh giá.
- `autograde.py`: Công cụ hỗ trợ chấm điểm nhanh dựa trên report.
