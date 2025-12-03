# How to use the classifier

```python
def on_data_received(packet):
    # Clarify basic information here
    system = IntegratedSystem(model_path = MODEL_PATH, blink_threshold = BLINK_THRESHOLD, fs = 512)    
    while True:
        # processing data (e.g. parsing packets)
        # put anything you need here
        
        for sample in packet:
            b_avg, b_state, bci_state = system.process_sample(sample)
            if bci_state == 1:    # 1 -> focus, 0 -> Relaxed
                print(f"Focus")
            if b_state == 1:      # 1 -> blink detected
                print("Blink detected!")
```

## About BrainLink2Classifier

This part is implimented by brutally parsing the packets received via Bluetooth (well it's not encoded, HOW?). Only a few pieces of data we need are extracted, and there are a lot more in `neurosky_reader.py`.
A simple example is presented in `bci_main.py`.

## How to find the connection port

1. Pair BrainLink device via Bluetooth.
2. Open `Device Manager -> Ports (COM & LPT)` on your computer.
3. If you see `COM{n}` and `COM{n+1}`, use `COM{n}`.
