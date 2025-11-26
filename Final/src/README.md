# How to use the classifier

```python

def on_data_received(packet):
    """
    當硬體收到一包數據時會呼叫此函數
    packet: 可能是一個包含 50 個 float 的 list
    """
    # 直接用迴圈把這一包 "倒" 進去系統
    for sample in packet:
        # 假設 sample 是單一頻道的數值 (float)
        b_avg, b_state, bci_state = system.process_sample(sample)
        
        # 處理輸出...
        if bci_state == 1:
            print(f"Focus")
# main
system = IntegratedSystem(MODEL_PATH, fs=512)
```
