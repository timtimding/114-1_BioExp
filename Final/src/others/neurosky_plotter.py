import serial
import threading
import time
import collections
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ==========================================
# 設定區
# ==========================================
COM_PORT = 'COM4'
BAUD_RATE = 57600        # NeuroSky 標準
PLOT_WINDOW = 512 * 5    # 畫面上顯示最近幾點 (512Hz * 5秒 = 2560點)
REFRESH_INTERVAL = 30    # 動畫更新間隔 (毫秒)，30ms 約等於 33 FPS

# ==========================================
# 全域變數 (用於執行緒間溝通)
# ==========================================
# 使用 deque (雙向佇列) 來儲存波形數據，當資料滿了會自動擠掉舊的
raw_data_buffer = collections.deque(np.zeros(PLOT_WINDOW), maxlen=PLOT_WINDOW)
attention_value = 0
signal_quality = 200 # 預設為收訊差
is_running = True

# ==========================================
# 執行緒 1: 負責讀取藍芽 (暴力拆包)
# ==========================================
def read_serial_thread():
    global attention_value, signal_quality, is_running
    
    print(f"Connecting to {COM_PORT}...")
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
        print("Serial connected! Reading data...")
    except Exception as e:
        print(f"Connection failed: {e}")
        is_running = False
        return

    buffer = b''
    
    while is_running:
        try:
            if ser.in_waiting > 0:
                buffer += ser.read(ser.in_waiting)
                
                # --- 暴力拆包邏輯 ---
                while len(buffer) >= 3:
                    if buffer[0] != 0xAA or buffer[1] != 0xAA:
                        buffer = buffer[1:]
                        continue
                    
                    payload_len = buffer[2]
                    packet_len = 2 + 1 + payload_len + 1
                    
                    if len(buffer) < packet_len:
                        break
                    
                    payload = buffer[3 : 3 + payload_len]
                    
                    # 解析 Payload
                    i = 0
                    while i < len(payload):
                        code = payload[i]
                        i += 1
                        
                        if code >= 0x80: # 多 Byte 數據
                            v_len = payload[i]
                            i += 1
                            val_bytes = payload[i : i + v_len]
                            
                            # === 抓取 Raw Wave (畫圖用) ===
                            if code == 0x80:
                                val = int.from_bytes(val_bytes, byteorder='big', signed=True)
                                # 加到 Buffer，deque 會自動處理先進先出
                                raw_data_buffer.append(val)
                            
                            i += v_len
                        else: # 單 Byte 數據
                            val = payload[i]
                            
                            # === 抓取訊號品質 ===
                            if code == 0x02:
                                signal_quality = val
                            
                            # === 抓取專注度 ===
                            elif code == 0x04:
                                attention_value = val
                                
                            i += 1
                            
                    buffer = buffer[packet_len:]
            else:
                # 沒資料時稍微休息，避免 CPU 佔用過高
                time.sleep(0.001)
                
        except Exception as e:
            print(f"Error: {e}")
            is_running = False
            ser.close()
            break

# ==========================================
# 主程式: 設定 Matplotlib 動畫
# ==========================================
def main():
    # 1. 啟動讀取執行緒
    t = threading.Thread(target=read_serial_thread)
    t.daemon = True # 設定為守護執行緒，主程式關閉時它也會關閉
    t.start()
    
    # 2. 設定繪圖視窗
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title('NeuroSky Real-time Plotter')
    
    # 初始化線條
    x_data = np.arange(PLOT_WINDOW)
    line, = ax.plot(x_data, np.zeros(PLOT_WINDOW), 'b-', linewidth=0.8)
    
    # 設定標籤與範圍
    ax.set_ylim(-2048, 2048) # Raw Wave 範圍通常在 +-2048 內，眨眼會爆表
    ax.set_xlim(0, PLOT_WINDOW)
    ax.set_title("Raw EEG Signal")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    ax.grid(True)

    # 顯示文字資訊 (專注度、訊號)
    info_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, 
                        fontsize=12, verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 3. 動畫更新函式
    def update(frame):
        if not is_running:
            plt.close(fig)
            return line, info_text

        # 更新波形數據
        # 為了效能，我們直接更新 Y 軸數據即可
        line.set_ydata(raw_data_buffer)
        
        # 動態調整 Y 軸範圍 (如果波動太大或太小) - 可選
        # current_data = list(raw_data_buffer)
        # y_min, y_max = min(current_data), max(current_data)
        # ax.set_ylim(y_min - 100, y_max + 100)

        # 更新文字
        status = "Good" if signal_quality == 0 else f"Poor ({signal_quality})"
        info_text.set_text(f'Attention: {attention_value}\nSignal: {status}')
        
        return line, info_text

    # 4. 啟動動畫 (blit=True 會大幅提升效能)
    ani = FuncAnimation(fig, update, interval=REFRESH_INTERVAL, blit=True)
    
    print("Plotter started. Close the window to stop.")
    plt.show()

    # 視窗關閉後
    global is_running
    is_running = False
    print("Exiting...")

if __name__ == "__main__":
    main()