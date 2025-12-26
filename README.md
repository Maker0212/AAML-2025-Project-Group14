## Spec
https://nycu-caslab.github.io/AAML2025/project/final_project.html

## 簡報連結
https://docs.google.com/presentation/d/1pG32kf4aLiwhChGHzTRK-Aw9QL3A0cGAZ8KdbxTefHY/edit?usp=drivesdk

# AAML 2025 Final Project: Wav2Letter Acceleration with CFU & TPU

本專案是國立陽明交通大學 AAML (Advanced Applied Machine Learning) 2025 課程的期末專題。目標是在 RISC-V 軟核心處理器上，透過設計客製化功能單元 (CFU) 和張量處理單元 (TPU) 硬體加速器，來加速 **Wav2Letter** 語音辨識模型的推論速度。

專案基於 [CFU-Playground](https://github.com/google/CFU-Playground) 框架開發。

## 專案簡介

本專案針對 `wav2letter_pruned_int8.tflite` 模型進行優化。我們實作了一個整合在 CPU 管線中的 CFU，以及一個專門處理矩陣乘法的 4x4 Systolic Array TPU。透過將卷積運算 (Convolution) 卸載 (Offload) 到硬體加速器，大幅提升了模型的推論效能。

## 硬體架構 (Hardware Architecture)

硬體設計位於 `cfu.v` 及 `hw/` 目錄下，主要包含以下組件：

### 1. Custom Function Unit (CFU)
CFU (`cfu.v`) 作為 CPU 與 TPU 之間的橋樑，並提供額外的指令集來加速特定運算。主要功能包括：
*   **指令解碼**：解析 RISC-V 自定義指令 (`funct3`, `funct7`)。
*   **TPU 控制介面**：負責設定 TPU 的參數 (M, N, K)、啟動運算、以及檢查忙碌狀態。
*   **資料搬運**：控制 Global Buffer (SRAM) 的讀寫，將權重 (Weights) 和輸入特徵 (Inputs) 載入緩衝區。
*   **後處理加速**：包含 SIMD 點積運算 (Dot Product)、量化 (Quantization)、累加器操作 (Accumulator) 及偏權值相加 (Bias Addition)。
*   **轉置單元 (Transpose Unit)**：硬體支援矩陣轉置操作，以符合 TPU 的資料排列需求。

### 2. Tensor Processing Unit (TPU)
TPU (`hw/TPU.v`) 是一個基於脈動陣列 (Systolic Array) 的矩陣乘法加速器：
*   **核心架構**：4x4 Processing Element (PE) 陣列。
*   **Tiling 支援**：支援任意大小的矩陣運算 (M, N, K)，硬體內部自動處理 Tiling (分塊)。
*   **雙緩衝 (Double Buffering)**：A/B/C Buffer 設計，允許在運算同時進行資料載入，掩蓋記憶體存取延遲。
*   **Global Buffers**：
    *   **Buffer A (Input)**: 儲存輸入 Activation。
    *   **Buffer B (Weight)**: 儲存權重。
    *   **Buffer C (Output)**: 儲存運算結果 (Accumulators)。

### 3. Processing Element (PE)
PE (`hw/PE.v`) 是運算的基本單元，負責執行乘加運算 (MAC) 並將資料傳遞給鄰近的 PE。

## 軟體架構 (Software Architecture)

軟體部分位於 `src/` 目錄，基於 TensorFlow Lite for Microcontrollers (TFLM)：

*   **TFLite Micro Kernels** (`src/tensorflow/lite/micro/kernels/`):
    *   我們替換了標準的 `Conv2D` 等運算核心，改為呼叫 CFU 指令。
    *   透過 `src/tpu_helper.h` 提供的介面與硬體溝通。
*   **Wav2Letter Application** (`src/wav2letter/`):
    *   包含模型定義 (`wav2letter_pruned_int8.tflite`)。
    *   測試資料 (`test_input.h`, `test_output.h`)。
    *   選單介面 (`wav2letter.cc`)，用於執行黃金測試 (Golden Tests) 驗證正確性。
*   **Software CFU** (`src/software_cfu.cc`):
    *   提供 CFU 指令的軟體模擬版本，用於功能驗證與開發初期的除錯。

## 檔案結構

```text
.
├── cfu.v                   # CFU 頂層模組 (Verilog)
├── hw/                     # 硬體加速器原始碼
│   ├── TPU.v               # TPU 控制器與狀態機
│   ├── systolic_array.v    # 4x4 脈動陣列
│   ├── PE.v                # 處理單元
│   └── global_buffer_bram.v # 內建記憶體 (BRAM)
├── src/                    # 軟體原始碼
│   ├── wav2letter/         # Wav2Letter 應用程式與模型
│   ├── tensorflow/         # TFLite Micro 函式庫與客製化 Kernels
│   ├── software_cfu.cc     # CFU 軟體模擬
│   ├── tpu_helper.h        # TPU 驅動程式介面
│   └── proj_menu.cc        # 專案主選單
├── Makefile                # 建置腳本
└── ...
