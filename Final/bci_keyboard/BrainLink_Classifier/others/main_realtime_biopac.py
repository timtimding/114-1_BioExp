import time
import numpy as np

# === 1. 引用您寫好的大腦 ===
# 假設您上一份程式碼存檔為 "main_integrated.py"
try:
    from Final.bci_keyboard.eeg_classifer_blink import IntegratedSystem
except ImportError:
    print("Cannot find'main_integrated.py', please check the path.")
    exit()

def main():
    # === 2. 初始化系統 ===
    MODEL_PATH = 'bci_system_v1_block_design.pkl'
    
    print("Loading BCI system..")
    try:
        # 初始化 IntegratedSystem
        # fs (採樣率) 必須跟您硬體設定的一樣！(例如 500Hz)
        bci_system = IntegratedSystem(model_path=MODEL_PATH, fs=500)
        print("System successfully loaded!")
    except Exception as e:
        print(f"Fail to initialize: {e}")
        return

    print("Waiting eeg signal...")
    
    # === 3. 真實數據迴圈 (Infinite Loop) ===
    try:
        while True:
            # ------------------------------------------------------------
            # [步驟 A] 從硬體獲取數據
            # 這裡通常會拿到一個 "Chunk" (一小包數據)，例如一次來 10 個點
            # ------------------------------------------------------------
            
            # !!! 請將這裡替換為您真實的硬體讀取代碼 !!!
            # 範例：假設從某個 SDK 讀到了一小段數據 (例如 Fp1-Fp2 的差值)
            # real_data_chunk = my_headset.read_data() 
            
            # (這裡暫時用隨機數模擬，讓程式能跑)
            real_data_chunk = np.random.randn(10) 
            time.sleep(0.02) # 模擬硬體延遲 (500Hz下 10點約需 0.02秒)
            
            # ------------------------------------------------------------
            # [步驟 B] 餵給系統處理
            # IntegratedSystem 設計為逐點處理 (process_sample)
            # ------------------------------------------------------------
            for sample in real_data_chunk:
                # 只要這一行！所有複雜邏輯都在裡面做完了
                blink_ma, blink_st, bci_st = bci_system.process_sample(sample)
                
                # --------------------------------------------------------
                # [步驟 C] 根據結果做應用 (控制遊戲、燈光、UI...)
                # --------------------------------------------------------
                
                # 應用範例 1: 偵測到眨眼時印出警告
                if blink_st == 1:
                    print("Blink detected!")
                
                # 應用範例 2: 顯示當前專注狀態
                # bci_st: 1=專注, 0=放鬆
                if bci_st == 1:
                    print(f"🔥 專注中... (數值: {sample:.2f})")
                else:
                    print(f"☕ 放鬆中... (數值: {sample:.2f})")

    except KeyboardInterrupt:
        print("\nSystem stopped")

if __name__ == "__main__":
    main()