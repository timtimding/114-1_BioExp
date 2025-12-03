import time
import numpy as np
import os

# === 1. 引用您寫好的大腦 ===
# 假設您上一份程式碼存檔為 "main_integrated.py"
try:
    from eeg_classifer_blink import IntegratedSystem
except ImportError:
    print("Cannot find'main_integrated.py', please check the path.")
    exit()

def main():
    # === 2. 初始化系統 ===
    MODEL_PATH = 'bci_system_v1.pkl'
    FS = 512
    print("Loading BCI system, sample rate = {FS}")
    try:
        # 初始化 IntegratedSystem
        # fs (採樣率) 必須跟您硬體設定的一樣！(例如 512Hz)
        bci_system = IntegratedSystem(model_path=MODEL_PATH, fs=FS)
        print("System successfully loaded!")
    except Exception as e:
        print(f"Fail to initialize: {e}")
        return

    # === 初始化檔案讀取變數 ===
    target_part = 1           # 我們現在等待讀取哪一份檔案 (1~8)
    pointer_file = 'data_ready.txt' # 握手用的指標檔
    
    # 程式啟動時，先清除舊的指標檔，以免讀到上次的殘留狀態
    if os.path.exists(pointer_file):
        try:
            os.remove(pointer_file)
            print("Cleaned up old pointer file.")
        except:
            pass
    print("Waiting for MATLAB data stream...")

    try:
        while True:
            # ------------------------------------------------------------
            # [步驟 A] 從硬體(這裡改為檔案)獲取數據
            # ------------------------------------------------------------
            real_data_chunk = [] # 這一輪要處理的數據
            
            try:
                # 1. 檢查指標檔是否存在
                if os.path.exists(pointer_file):
                    with open(pointer_file, 'r') as f:
                        content = f.read().strip()
                    
                    # 確保內容是數字
                    if content.isdigit():
                        ready_part = int(content)
                        
                        # 2. 握手成功：MATLAB 說它剛寫完的檔案 == 我們想讀的檔案
                        if ready_part == target_part:
                            csv_file = f'raw_data_part{target_part}.csv'
                            
                            # 雙重確認 CSV 存在
                            if os.path.exists(csv_file):
                                # 讀取數據 (這就是我們拿到的 Chunk，通常是 64 點)
                                # dtype=int 因為我們在 MATLAB 改成存整數了
                                real_data_chunk = np.loadtxt(csv_file, dtype=int)
                                
                                # 除錯顯示 (可註解掉)
                                # print(f"-> Reading Part {target_part}, Got {len(real_data_chunk)} points")
                                
                                # 3. 準備下一輪
                                target_part += 1
                                if target_part > 8:
                                    target_part = 1
            except Exception as e:
                print(f"File Read Error: {e}")
                # 遇到讀取錯誤稍微停一下，不要崩潰
                time.sleep(0.1) 
                continue

            # 如果這一輪沒有讀到新資料 (MATLAB 還沒寫完)，就休息一下避免 CPU 飆高
            if len(real_data_chunk) == 0:
                time.sleep(0.01)
                continue

            # ------------------------------------------------------------
            # [步驟 B] 餵給系統處理
            # real_data_chunk 是一個 array (例如 64 個點)
            # 我們用迴圈把它們一顆一顆餵進去
            # ------------------------------------------------------------
            for sample in real_data_chunk:
                # process_sample 負責濾波、特徵提取、判斷
                blink_ma, blink_st, bci_st = bci_system.process_sample(sample)
                
                # --------------------------------------------------------
                # [步驟 C] 根據結果做應用
                # --------------------------------------------------------
                
                # 應用範例 1: 偵測到眨眼
                if blink_st == 1:
                    print(f"*** BLINK DETECTED! (Part {target_part-1 if target_part>1 else 8}) ***")
                
                # 應用範例 2: 顯示專注狀態 (為了避免洗版，我們可以每處理完一個檔案顯示一次狀態)
                # 這裡單純示範每一點都判斷
                if bci_st == 1:
                   print("Focus")

            # [選用] 如果想要視覺上看起來像即時流動，可以在這裡加極短的 sleep
            # 但通常為了效能，我們會希望越快處理完越好，等待下一批資料
            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nSystem stopped by user.")

if __name__ == "__main__":
    main()