import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import threading
import time

class BCIDashboard:
    def __init__(self):
        # === 狀態變數 (外部程式修改這些變數即可) ===
        self.is_focus = False
        self.is_relax = False
        self.blink_triggered = False  # 設為 True 會閃一下，然後自動變回 False
        
        # 內部變數
        self._running = True
        self._blink_counter = 0      # 用來控制眨眼燈號亮多久
        
    def _init_plot(self):
        """初始化繪圖視窗與物件"""
        self.fig, self.ax = plt.subplots(figsize=(6, 3))
        self.fig.canvas.manager.set_window_title('BCI Live Status')
        
        # 隱藏座標軸
        self.ax.set_xlim(0, 3)
        self.ax.set_ylim(0, 1)
        self.ax.axis('off')
        
        # 定義三個燈號 (圓形)
        # 參數: ((x, y), radius, color, alpha)
        # Alpha 0.2 = 暗 (滅), 1.0 = 亮
        self.circle_relax = patches.Circle((0.5, 0.6), 0.3, color='blue', alpha=0.2)
        self.circle_focus = patches.Circle((1.5, 0.6), 0.3, color='green', alpha=0.2)
        self.circle_blink = patches.Circle((2.5, 0.6), 0.3, color='red', alpha=0.2)
        
        self.ax.add_patch(self.circle_relax)
        self.ax.add_patch(self.circle_focus)
        self.ax.add_patch(self.circle_blink)
        
        # 加入文字標籤
        self.ax.text(0.5, 0.1, "RELAX", ha='center', fontsize=12, fontweight='bold')
        self.ax.text(1.5, 0.1, "FOCUS", ha='center', fontsize=12, fontweight='bold')
        self.ax.text(2.5, 0.1, "BLINK", ha='center', fontsize=12, fontweight='bold')
        
    def _update(self, frame):
        """動畫更新函式"""
        if not self._running:
            plt.close(self.fig)
            return
        
        # 1. 更新放鬆燈 (藍色)
        if self.is_relax:
            self.circle_relax.set_alpha(1.0) # 亮
            self.circle_relax.set_edgecolor('black')
        else:
            self.circle_relax.set_alpha(0.1) # 滅
            
        # 2. 更新專注燈 (綠色)
        if self.is_focus:
            self.circle_focus.set_alpha(1.0)
        else:
            self.circle_focus.set_alpha(0.1)
            
        # 3. 更新眨眼燈 (紅色 - 觸發後維持幾幀)
        if self.blink_triggered:
            self.circle_blink.set_alpha(1.0)
            self._blink_counter = 5 # 讓燈亮 5 個 frame (約 150ms)
            self.blink_triggered = False # 重置觸發器
        
        if self._blink_counter > 0:
            self._blink_counter -= 1
        else:
            self.circle_blink.set_alpha(0.1)
            
        return self.circle_relax, self.circle_focus, self.circle_blink

    def start(self):
        """啟動儀表板 (會阻塞主執行緒，請將邏輯放在其他 Thread)"""
        self._init_plot()
        # interval=30ms (約 33 FPS)
        ani = FuncAnimation(self.fig, self._update, interval=30, blit=False)
        plt.show()
        self._running = False # 視窗關閉後設為停止