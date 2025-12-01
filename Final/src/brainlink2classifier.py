import serial
import time
import numpy as np
from eeg_classifer_blink import IntegratedSystem # 您的 AI 系統

# === 設定 ===
# 請去裝置管理員看您的 NeuroSky 連在哪個 COM Port
# Windows: 'COM3', 'COM4' 等
# Mac/Linux: '/dev/tty.MindWaveMobile-SerialPort' 等
COM_PORT = 'COM3' 
BAUD_RATE = 57600

def parse_neurosky_stream():
    # 1. 初始化 AI 系統
    MODEL_PATH = 'bci_system_v1.pkl'
    try:
        # 假設您的模型是訓練在 512Hz
        bci_system = IntegratedSystem(model_path=MODEL_PATH, fs=512)
        print("AI System Loaded.")
    except Exception as e:
        print(f"AI Init Failed: {e}")
        return

    # 2. 連接藍芽 Serial Port
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
        print(f"Connected to {COM_PORT}")
    except serial.SerialException:
        print(f"Could not connect to {COM_PORT}. Check bluetooth pairing.")
        return

    print("Starting packet parsing...")
    
    # 用來暫存數據的 Buffer
    buffer = b''
    
    try:
        while True:
            # 讀取 Serial Port 數據
            if ser.in_waiting > 0:
                buffer += ser.read(ser.in_waiting)
                
                # ---------------------------------------------------------
                # 暴力拆包邏輯 (參照 neuroskylab.m 的 read_packets)
                # ---------------------------------------------------------
                while len(buffer) >= 3: # 至少要有 AA AA [Length]
                    # 1. 找同步標頭 AA AA
                    # MATLAB: inds = strfind(A', [170 170])
                    try:
                        idx = buffer.index(b'\xaa\xaa')
                    except ValueError:
                        # 沒找到，保留最後一個 byte (怕剛好切在 AA 之間)，其他丟掉
                        buffer = buffer[-1:]
                        continue
                    
                    # 丟掉 AA AA 之前的垃圾數據
                    if idx > 0:
                        buffer = buffer[idx:]
                    
                    # 確保有足夠長度讀取 Length byte (AA AA [Len])
                    if len(buffer) < 3:
                        break # 資料不夠，等下一輪
                        
                    # 2. 讀取 Payload 長度
                    # MATLAB: value_length = packet(packet_index)
                    payload_len = buffer[2]
                    
                    # 確保 Buffer 裡有完整的封包 (Header(2) + Len(1) + Payload(Len) + Checksum(1))
                    packet_len = 2 + 1 + payload_len + 1
                    
                    if len(buffer) < packet_len:
                        break # 資料還沒收完，等下一輪
                        
                    # 3. 取出完整 Payload
                    payload = buffer[3 : 3+payload_len]
                    checksum_byte = buffer[3+payload_len]
                    
                    # (選擇性) 可以在這裡做 Checksum 校驗，邏輯在 MATLAB 的 checkpacket 函數
                    # checksum = sum(payload)的低8位取反
                    
                    # 4. 解析 Payload 內容
                    # 我們模仿 MATLAB 的 parse_packet 邏輯
                    i = 0
                    while i < len(payload):
                        code = payload[i]
                        i += 1
                        
                        # 判斷是單 byte 還是多 byte 數據
                        # MATLAB: if( datarow_code >= 128 )
                        if code >= 0x80: 
                            # 多 byte 數據，下一個 byte 是長度
                            if i >= len(payload): break
                            v_len = payload[i]
                            i += 1
                            if i + v_len > len(payload): break
                            
                            val_bytes = payload[i : i+v_len]
                            
                            # === 關鍵：抓取 Raw Wave (Code 0x80 / 128) ===
                            if code == 0x80:
                                # MATLAB: payloadBuffer(3)*256+payloadBuffer(4)
                                # 這是 16-bit Big Endian Signed Integer
                                raw_val = int.from_bytes(val_bytes, byteorder='big', signed=True)
                                
                                # === 直接餵給 AI ===
                                blink_ma, blink_st, bci_st = bci_system.process_sample(raw_val)
                                
                                if blink_st == 1:
                                    print(f"*** BLINK! (Val: {raw_val}) ***")
                                # if bci_st == 1:
                                #     print("Focus...")
                                    
                            i += v_len
                        else:
                            # 單 byte 數據 (例如訊號品質 0x02, 電量 0x01)
                            if i >= len(payload): break
                            val = payload[i]
                            
                            # 可以在這裡抓 Signal Quality (Code 0x02)
                            # if code == 0x02:
                            #     print(f"Signal Poor: {val}") # 0 是最好，200 是脫落
                            
                            i += 1

                    # 處理完這個封包，從 buffer 移除
                    buffer = buffer[packet_len:]
                    
    except KeyboardInterrupt:
        print("Stopped.")
        ser.close()

if __name__ == "__main__":
    parse_neurosky_stream()