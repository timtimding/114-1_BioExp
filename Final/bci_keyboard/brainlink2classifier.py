import serial
import threading
import time
import struct
import sys

try:
    from eeg_classifer_blink import IntegratedSystem
except ImportError:
    print("[Error] eeg_classifer_blink.py not found.")
    sys.exit(1)

class BrainLink2Classifier:
    def __init__(self, port='COM4', baud=57600, model_path='bci_system_v1.pkl',
                 fs=512, timeout=0.8, max_count=3, cooldown=2.0, threshold=-170):
        self.port = port
        self.baud = baud
        self.model_path = model_path
        
        self.running = False
        self.thread = None
        self.ser = None
        self.bci_system = None

        self.fs = fs
        self.threshold = threshold
        self.timeout = timeout
        self.max_count = max_count
        self.cooldown = cooldown
        
        # === 輸出狀態緩存 ===
        self._current_focus = False  # True=專注, False=放鬆
        self._last_blink_cmd = 0     # 最近一次的眨眼指令 (0, 1, 2, 3...)
        
        # === Callback 函式 ===
        # 格式: func(is_focus: bool, blink_count: int)
        self.callback = None 

    def start(self):
        """啟動監聽執行緒"""
        if self.running:
            print("Driver is already running.")
            return

        # 1. Initializaiton of AI
        print("Loading AI Model...")
        try:
            self.bci_system = IntegratedSystem(
                model_path=self.model_path,
                blink_threshold=self.threshold,
                fs=self.fs,
                timeout=self.timeout,           # 連點判斷時間
                max_count=self.max_count,       # 最大連點數
                cooldown=self.cooldown          # 冷卻時間
            )
        except Exception as e:
            print(f"[Error] AI Init Failed: {e}")
            return

        # 2. Start Serial Port
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            print(f"Connected to {self.port}")
        except serial.SerialException as e:
            print(f"[Error] Serial Open Failed: {e}")
            return

        # 3. 啟動執行緒
        self.running = True
        self.thread = threading.Thread(target=self._process_loop)
        self.thread.daemon = True # 設定為守護執行緒，主程式關閉時自動結束
        self.thread.start()
        print("BrainLink Driver Started.")

    def stop(self):
        """停止監聽"""
        self.running = False
        if self.ser and self.ser.is_open:
            self.ser.close()
        if self.thread:
            self.thread.join(timeout=2.0)
        print("BrainLink Driver Stopped.")

    def set_callback(self, func):
        """
        設定回調函式，當有新狀態時觸發。
        func signature: def my_callback(is_focus, blink_count):
        """
        self.callback = func

    def _process_loop(self):
        """後台處理迴圈 (暴力拆包 + AI 推論)"""
        buffer = b''
        
        while self.running:
            try:
                if self.ser.in_waiting > 0:
                    buffer += self.ser.read(self.ser.in_waiting)
                    
                    # --- 暴力拆包邏輯 ---
                    while len(buffer) >= 3:
                        # 1. 找同步標頭 AA AA
                        try:
                            idx = buffer.index(b'\xaa\xaa')
                        except ValueError:
                            buffer = buffer[-1:] # 沒找到，留最後一個 byte
                            continue
                        
                        if idx > 0: 
                            buffer = buffer[idx:]
                        
                        if len(buffer) < 3: 
                            break
                        
                        # 2. 讀取長度與檢查完整性
                        payload_len = buffer[2]
                        packet_len = 2 + 1 + payload_len + 1 # Header + Len + Payload + Checksum
                        
                        if len(buffer) < packet_len: 
                            break # 資料不夠，等下一輪
                        
                        # 3. 取出 Payload
                        payload = buffer[3 : 3+payload_len]
                        
                        # 4. 解析內容
                        i = 0
                        while i < len(payload):
                            code = payload[i]
                            i += 1
                            
                            if code >= 0x80: # Multi-byte
                                if i >= len(payload): break
                                v_len = payload[i]
                                i += 1
                                if i + v_len > len(payload): break
                                val_bytes = payload[i : i+v_len]
                                
                                # === 關鍵：Raw Wave (0x80) ===
                                if code == 0x80:
                                    raw_val = int.from_bytes(val_bytes, byteorder='big', signed=True)
                                    
                                    # === AI 處理 ===
                                    # process_sample 回傳: (ma, blink_st, bci_st, blink_seq_count)
                                    _, _, bci_st, seq_cnt = self.bci_system.process_sample(raw_val)
                                    
                                    # 更新狀態
                                    # bci_st: 1=Focus, 0=Relax
                                    new_focus = True if bci_st == 1 else False
                                    
                                    # 觸發 Callback 的條件：
                                    # 1. 眨眼指令產生 (seq_cnt > 0)
                                    # 2. 或者 專注狀態改變 (選擇性，也可以每一幀都回傳)
                                    
                                    if seq_cnt > 0 or new_focus != self._current_focus:
                                        self._current_focus = new_focus
                                        if self.callback:
                                            # 回傳: (專注狀態, 眨眼指令次數)
                                            # seq_cnt 只有在觸發瞬間會是 1, 2, 3，其餘時間是 0
                                            self.callback(self._current_focus, seq_cnt)
                                
                                i += v_len
                            else: # Single-byte
                                if i >= len(payload): break
                                # val = payload[i] 
                                # 這裡可以處理 Signal Quality (Code 0x02)
                                i += 1
                        
                        # 處理完移除封包
                        buffer = buffer[packet_len:]
                else:
                    time.sleep(0.001) # 避免 CPU 100%

            except Exception as e:
                # ==========================================
                # [新增功能] 斷線重連機制
                # ==========================================
                print(f"[Driver Error] Lost connection: {e}")
                print("Trying to reconnect...")
                
                # 1. 先嘗試安全關閉舊連線
                try:
                    if self.ser:
                        self.ser.close()
                except:
                    pass
                
                # 2. 進入重試迴圈 (直到連上或程式被關閉)
                reconnected = False
                while self.running and not reconnected:
                    try:
                        time.sleep(2.0) # Try every 2 seconds
                        
                        # Try restart Serial
                        self.ser = serial.Serial(self.port, self.baud, timeout=1)
                        
                        # 若成功執行到這行，代表連上了
                        print(f"[Driver Info] Successfully reconnect! {self.port}!")
                        buffer = b'' # 清空舊緩衝區，避免資料錯亂
                        reconnected = True
                        
                    except serial.SerialException:
                        print(f"Fail to reconnect... retry in 2 seconds")
                    except Exception as ex:
                        print(f"Unexpected error: {ex}")