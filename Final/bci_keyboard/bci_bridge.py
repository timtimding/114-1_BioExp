"""
BCI Bridge - Connects BrainLink2 EEG headset to T9 Keyboard Flask server

This script:
1. Initializes BrainLink2Classifier
2. Receives focus/blink signals from BCI
3. Sends corresponding states to Flask API
"""

import time
import requests
from src.brainlink2classifier import BrainLink2Classifier

# Flask server configuration
FLASK_URL = 'http://localhost:5000'
API_ENDPOINT = f'{FLASK_URL}/api/input'

# Track current state to avoid redundant calls
current_focus_state = None


def send_state_to_flask(state_num):
    """
    Send state to Flask server
    Args:
        state_num: int (1-5)
    """
    try:
        response = requests.post(f'{API_ENDPOINT}/{state_num}', timeout=1)
        if response.status_code == 200:
            print(f"✓ Sent State {state_num} to Flask")
        else:
            print(f"✗ Flask error: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to Flask server. Is it running?")
    except requests.exceptions.Timeout:
        print("✗ Flask request timeout")
    except Exception as e:
        print(f"✗ Error sending to Flask: {e}")


def bci_callback_handler(is_focus, blink_count):
    """
    Callback function called by BrainLink2Classifier
    
    Args:
        is_focus: bool (True=專注/Focus, False=放鬆/Relax)
        blink_count: int (0=無動作, 1=單擊, 2=雙擊, 3=三擊)
    """
    global current_focus_state
    
    # Display current state
    state_str = "專注 (Focus)" if is_focus else "放鬆 (Relax)"
    print(f"\r狀態: {state_str} | ", end="", flush=True)
    
    # Handle focus/relax state changes
    if is_focus != current_focus_state:
        current_focus_state = is_focus
        
        if is_focus:
            # Focus detected → State 2
            print(f"\n>>> 狀態改變: 專注 → State 2 (Focus - Start scanning)")
            send_state_to_flask(2)
        else:
            # Relax detected → State 1
            print(f"\n>>> 狀態改變: 放鬆 → State 1 (Relax - Stop scanning)")
            send_state_to_flask(1)
        
        print("-" * 50)
    
    # Handle blink commands
    if blink_count > 0:
        print(f"\n>>> 收到眨眼指令: {blink_count} 次！ <<<")
        
        if blink_count == 1:
            # 1 blink → State 3 (Confirm)
            print("執行動作: 確認選擇 → State 3 (Confirm)")
            send_state_to_flask(3)
            
        elif blink_count == 2:
            # 2 blinks → State 4 (Enter predictions)
            print("執行動作: 進入預測模式 → State 4 (Enter Predictions)")
            send_state_to_flask(4)
            
        elif blink_count == 3:
            # 3 blinks → State 5 (Cancel)
            print("執行動作: 取消/返回 → State 5 (Cancel)")
            send_state_to_flask(5)
        
        print("-" * 50)


def main():
    """Main function to start BCI bridge"""
    
    print("=" * 60)
    print("BCI Bridge - Connecting BrainLink2 to T9 Keyboard")
    print("=" * 60)
    
    # Check if Flask server is running
    try:
        response = requests.get(FLASK_URL, timeout=2)
        print("✓ Flask server detected")
    except:
        print("⚠ WARNING: Cannot connect to Flask server!")
        print("  Please start Flask server first:")
        print("  python app_bci.py")
        print("\nContinuing anyway...")
    
    print("\nBCI State Mapping:")
    print("  放鬆 (Relax)  → State 1 (Stop scanning)")
    print("  專注 (Focus)  → State 2 (Start scanning)")
    print("  眨眼 1次      → State 3 (Confirm)")
    print("  眨眼 2次      → State 4 (Enter predictions)")
    print("  眨眼 3次      → State 5 (Cancel)")
    print("=" * 60)
    
    # Initialize BrainLink2 driver
    print("\n初始化 BrainLink2...")
    try:
        driver = BrainLink2Classifier(
            port='COM4',  # Adjust COM port if needed
            model_path='src/bci_system_v1.pkl'
        )
    except Exception as e:
        print(f"✗ 無法初始化 BrainLink2: {e}")
        print("\n可能的解決方案:")
        print("  1. 檢查 COM port 是否正確 (目前設定: COM4)")
        print("  2. 確認設備已連接")
        print("  3. 檢查 model 路徑: src/bci_system_v1.pkl")
        return
    
    # Set callback function
    driver.set_callback(bci_callback_handler)
    
    # Start BCI system
    print("啟動 BCI 系統...")
    driver.start()
    print("✓ BCI Bridge 運行中！")
    print("\n按 Ctrl+C 停止")
    print("=" * 60)
    
    try:
        # Keep running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n程式結束")
        driver.stop()
        print("BCI Bridge 已停止")


if __name__ == "__main__":
    main()