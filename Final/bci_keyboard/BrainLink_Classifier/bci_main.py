import time
from Final.bci_keyboard.brainlink2classifier import BrainLink2Classifier

# ==========================================
# 1. 定義一個函式來接收結果
# ==========================================
def my_bci_handler(is_focus, blink_count):
    """
    這個函式會由 Driver 自動呼叫。
    :param is_focus: bool (True=專注, False=放鬆)
    :param blink_count: int (0=無動作, 1=單擊, 2=雙擊, 3=三擊)
    """
    
    # 顯示目前狀態
    state_str = "🔥 專注" if is_focus else "☕ 放鬆"
    print(f"\r狀態: {state_str} | ", end="")
    
    # 處理眨眼指令
    if blink_count > 0:
        print(f"\n>>> 收到指令: 連續眨眼 {blink_count} 次！ <<<")
        
        if blink_count == 1:
            print("執行動作: 選擇 / 確認")
        elif blink_count == 2:
            print("執行動作: 上一頁 / 雙擊功能")
        elif blink_count == 3:
            print("執行動作: 回首頁 / 三擊功能")
            
        print("-" * 30)

# ==========================================
# 2. 主程式邏輯
# ==========================================
def main():
    # 初始化驅動器
    driver = BrainLink2Classifier(port='COM4', model_path='bci_system_v1.pkl')
    
    # 設定接收函式
    driver.set_callback(my_bci_handler)
    
    # 啟動
    print("系統啟動中...")
    driver.start()
    
    try:
        # 主程式可以做其他事情，例如跑遊戲迴圈、UI更新等
        # 這裡用無窮迴圈模擬主程式持續運行
        while True:
            time.sleep(1) 
            
    except KeyboardInterrupt:
        print("\n程式結束")
        driver.stop()

if __name__ == "__main__":
    main()