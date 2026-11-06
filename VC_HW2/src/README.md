# 需要完成的 `src/` 程式

請自行實作 `encoder.py`、`decoder.py`，可以另加 `codec.py` 等 helper；所有執行相依程式都要放在 `src/`。

- Q1 (`full_search`)：固定 16×16、Full Search ±15、ME **MAD-only**，mode=1。
- Q2 (`four_step_search`)：固定 16×16、4SS ±7、ME **MAD-only**，mode=1。
- Q3 (`four_step_rdo`)：4SS ±7，**ME 只用 MAD**；兩種 16×16 / 4×8×8 候選先完整 encode + reconstruct，再用 **`SSD+64×macroblock bits`** 選 mode。

仍沿用 HW1 8×8 transform 和固定 entropy coding。Q1/Q2/Q3 每個 16×16 區域的 bitstream 都依序有 **mode、MVD(s)、四個 8×8 residual blocks**，順序依 mode=1 或 mode=0 而定。

**Bit-level 讀寫的參考程式：** `../provided/exp_golomb_demo.py`；**完整 bitstream syntax 檢查器：** `../provided/evaluator.py`。不要直接複製教師實作或硬編碼已知輸出。
