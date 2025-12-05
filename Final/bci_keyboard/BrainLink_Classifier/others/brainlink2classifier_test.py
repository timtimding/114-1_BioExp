# In this program, I tried to brutally teardown the packets
# received via bluetooth and extract raw-data part, according
# to the protocals in BrainLink developer documentation.

import serial
import time
import numpy as np
from Final.bci_keyboard.eeg_classifer_blink import IntegratedSystem # 您的 AI 系統
from bci_dashboard import BCIDashboard

import threading
from blink_plotter import blinkPlotter

COM_PORT = 'COM4' 
BAUD_RATE = 57600
BLINK_THRESHOLD = -170
FS = 512
MODEL_PATH = 'bci_system_v1.pkl'

def parse_neurosky_stream(dash):
    # Initialize the system
    try:
        bci_system = IntegratedSystem(model_path=MODEL_PATH, blink_threshold=BLINK_THRESHOLD,
                                      fs=FS, timeout=0.8, max_count=3, cooldown=2.0)
        print("AI System Loaded.")
    except Exception as e:
        print(f"AI Init Failed: {e}")
        return

    # Connect to Bluetooth Serial Port
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
        print(f"Connected to {COM_PORT}")
    except serial.SerialException:
        print(f"Could not connect to {COM_PORT}. Check bluetooth pairing.")
        return

    print("Starting packet parsing...")
    
    # Buffer
    buffer = b''

    while dash._running:
        try:
            while True:
                # Read the data in Serial Port
                if ser.in_waiting > 0:
                    buffer += ser.read(ser.in_waiting)
                    
                    # ---------------------------------------------------------
                    # BRUTAL TEARDOWN (according to read_packets() in neuroskylab.m)
                    # ---------------------------------------------------------
                    while len(buffer) >= 3: # 至少要有 AA AA [Length]
                        # 1. Find synchronous header AA AA
                        # MATLAB: inds = strfind(A', [170 170])
                        try:
                            idx = buffer.index(b'\xaa\xaa')
                        except ValueError:
                            # 沒找到，保留最後一個 byte (怕剛好切在 AA 之間)，其他丟掉
                            buffer = buffer[-1:]
                            continue
                        
                        # Dump junks before AA AA
                        if idx > 0:
                            buffer = buffer[idx:]
                        
                        # Length byte (AA AA [Len])
                        if len(buffer) < 3:
                            break # Wait
                            
                        # Get the length of Payload
                        # MATLAB: value_length = packet(packet_index)
                        payload_len = buffer[2]
                        
                        # Make sure that complete packets is in the Buffer
                        # (Header(2) + Len(1) + Payload(Len) + Checksum(1))
                        packet_len = 2 + 1 + payload_len + 1
                        
                        if len(buffer) < packet_len:
                            break # 尚未完全收到
                            
                        # 3. Get complete Payload
                        payload = buffer[3 : 3+payload_len]
                        checksum_byte = buffer[3+payload_len]
                        
                        
                        # 4. Parse Payload
                        # Imitate the logic of parse_packet() in MATLAB
                        i = 0
                        while i < len(payload):
                            code = payload[i]
                            i += 1
                            
                            # single- or multi- byte data
                            # MATLAB: if( datarow_code >= 128 )
                            if code >= 0x80: 
                                # multi-byte data, the next byte is the length
                                if i >= len(payload): break
                                v_len = payload[i]
                                i += 1
                                if i + v_len > len(payload): break
                                
                                val_bytes = payload[i : i+v_len]
                                
                                # === Get Raw Wave (Code 0x80 / 128) ===
                                if code == 0x80:
                                    # MATLAB: payloadBuffer(3)*256+payloadBuffer(4)
                                    # 16-bit Big Endian Signed Integer
                                    raw_val = int.from_bytes(val_bytes, byteorder='big', signed=True)
                                    
                                    blink_ma, blink_st, bci_st, blink_count = bci_system.process_sample(raw_val)
                                    
                                    # 存取底層 Blink Detector 的除錯變數
                                    # try:
                                    #     debug_avg = bci_system.blink_detector.debug_avg
                                    # except:
                                    #     debug_avg = 0 # 若無法取得則補0
                                    # plotter.add_data(debug_avg, blink_st)
                                    
                                    if blink_st == 1:
                                        print(f"*** BLINK! (Val: {raw_val}) ***")
                                    # if bci_st == 1:
                                        # print("Focus...")
                                    # else: print("Relaxed...")
                                    if blink_count > 0:
                                        print(blink_count)
                                    if bci_st:
                                        dash.is_focus = True
                                        dash.is_relax = False
                                    else:
                                        dash.is_focus = False
                                        dash.is_relax = True
                    
                                    # 假設 blink_detected 為 True
                                    if blink_st:
                                        dash.blink_triggered = True
                                i += v_len
                            else:
                                # 單 byte 數據 (例如訊號品質 0x02, 電量 0x01)
                                if i >= len(payload): break
                                val = payload[i]
                                
                                # Signal Quality (Code 0x02) is available here
                                # if code == 0x02:
                                #     print(f"Signal Poor: {val}") # 0 -> best，200 -> not on the head
                                
                                i += 1

                        # Remove this packet
                        buffer = buffer[packet_len:]
        except KeyboardInterrupt:
            print("Stopped.")
            ser.close()
def main():
    # Plot for Classifier & blink detection
    dash = BCIDashboard()
    t = threading.Thread(target=parse_neurosky_stream, args=(dash,))
    
    # 設定為 Daemon (守護執行緒)，這樣主程式關閉時它會自動跟著關閉
    t.daemon = True 
    t.start()
    dash.start()

    # Plot for blink detection
    # # 1. 初始化繪圖器 (設定閾值與範圍)
    # # threshold 設為 -150，Y 軸範圍設為 -400~200
    # plotter = blinkPlotter(buffer_len=1500, threshold=BLINK_THRESHOLD, y_range=(-1000, 1000))
    
    # # 2. 啟動後台處理執行緒
    # t = threading.Thread(target=parse_neurosky_stream, args=(plotter,))
    # t.daemon = True # 設定為 daemon，主程式關閉時它也會關閉
    # t.start()
    
    # # 3. 啟動 GUI (這會阻斷主執行緒)
    # print("Launching Plotter...")
    # plotter.start()
if __name__ == "__main__":
    # parse_neurosky_stream()
    main()