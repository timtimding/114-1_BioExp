import serial
import time
import struct
from eeg_classifer_blink import BCIEngine

# ==========================================
# 設定區
# ==========================================
# Windows 請看裝置管理員 (例如 'COM3', 'COM4')
# Mac/Linux 通常是 '/dev/tty.MindWaveMobile-SerialPort'
COM_PORT = 'COM4'  
BAUD_RATE = 57600  # NeuroSky 標準鮑率

def read_neurosky_data():
    print(f"Connecting to {COM_PORT} at {BAUD_RATE}...")
    
    try:
        # 開啟 Serial Port
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
        print("Connected! Waiting for data stream...")
    except serial.SerialException as e:
        print(f"Error connecting to port: {e}")
        print("請檢查藍芽是否已配對，且 COM Port 設定正確。")
        return

    buffer = b''
    
    try:
        while True:
            # 1. 讀取資料到 Buffer
            if ser.in_waiting > 0:
                buffer += ser.read(ser.in_waiting)
                # print(buffer)
            
            # 2. 解析封包 (暴力拆包邏輯)
            # 封包結構: [AA] [AA] [Length] [Payload...] [Checksum]
            while len(buffer) >= 3:
                # 檢查同步標頭 (AA AA)
                if buffer[0] != 0xAA or buffer[1] != 0xAA:
                    # 如果不是 AA AA，丟掉第一個 byte，繼續往後找
                    buffer = buffer[1:]
                    continue
                
                # 取得 Payload 長度
                payload_len = buffer[2]
                
                # 檢查 Buffer 是否已包含完整封包
                # 完整長度 = Header(2) + Length(1) + Payload(payload_len) + Checksum(1)
                packet_len = 2 + 1 + payload_len + 1
                
                if len(buffer) < packet_len:
                    # 資料還沒收完，跳出迴圈等下一批資料
                    break
                
                # --- 截取完整封包 ---
                payload = buffer[3 : 3 + payload_len]
                checksum_byte = buffer[3 + payload_len]
                
                # (選擇性) 可以在這裡做 Checksum 檢查
                # payload_sum = sum(payload)
                # calculated_checksum = (~payload_sum) & 0xFF
                # if calculated_checksum != checksum_byte:
                #     print("Checksum error!")
                #     buffer = buffer[packet_len:] # 丟掉壞封包
                #     continue

                # --- 解析 Payload 內容 ---
                i = 0
                while i < len(payload):
                    code = payload[i]
                    i += 1
                    
                    # 判斷是否為「多 Byte」數據 (Code >= 128 / 0x80)
                    if code >= 0x80:
                        # 下一個 byte 是長度
                        if i >= len(payload): break
                        v_len = payload[i]
                        i += 1
                        
                        if i + v_len > len(payload): break
                        val_bytes = payload[i : i + v_len]
                        
                        # === [重點] 抓取 Raw Wave (Code 0x80) ===
                        if code == 0x80:
                            # 原始腦波是 16-bit Big-Endian Signed Integer
                            # 對應 MATLAB: payloadBuffer(3)*256 + payloadBuffer(4)
                            raw_value = int.from_bytes(val_bytes, byteorder='big', signed=True)
                            
                            # 直接印出數值 (這就是你要餵給 AI 的東西)
                            print(raw_value)

                            
                        # === [重點] 抓取 ASIC EEG Power (Code 0x83) (頻譜) ===
                        elif code == 0x83:
                            # 這裡有 Delta, Theta, Alpha... 等 8 個頻帶的數值
                            # 每個頻帶是 3 bytes unsigned int
                            # 可以根據需要解析
                            pass
                            
                        i += v_len # 跳過數值區段
                        
                    else:
                        # 單 Byte 數據 (Code < 128)
                        if i >= len(payload): break
                        val = payload[i]
                        
                        # === 抓取訊號品質 (Code 0x02) ===
                        if code == 0x02:
                            # 0 = Good, 200 = Sensor Off (脫落)
                            if val > 0:
                                print(f"--- Signal Poor: {val} (Check Sensor) ---")
                        
                        # === 抓取專注度 (Code 0x04) ===
                        elif code == 0x04:
                            print(f"--- Attention: {val} ---")
                            
                        # === 抓取眨眼強度 (Code 0x16) ===
                        elif code == 0x16:
                            print(f"--- Blink Strength: {val} ---")
                            
                        i += 1

                # 處理完這個封包，從 Buffer 中移除
                buffer = buffer[packet_len:]

    except KeyboardInterrupt:
        print("\nStopping...")
        ser.close()
        print("Connection closed.")

if __name__ == "__main__":
    read_neurosky_data()