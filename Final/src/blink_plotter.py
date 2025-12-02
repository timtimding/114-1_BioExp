import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import collections
import numpy as np

class blinkPlotter:
    def __init__(self, buffer_len=1000, threshold=-150, y_range=(-400, 200)):
        """
        初始化繪圖器
        :param buffer_len: X軸顯示的數據點數量
        :param threshold: 眨眼觸發閾值 (將繪製紅色虛線)
        :param y_range: Y軸範圍 (min, max)
        """
        self.buffer_len = buffer_len
        self.threshold = threshold
        self.y_range = y_range
        
        # 數據緩衝區 (儲存移動平均值)
        self.avg_buffer = collections.deque([0.0] * buffer_len, maxlen=buffer_len)
        # 狀態緩衝區 (儲存眨眼觸發狀態 0 or 1，用於視覺提示)
        self.state_buffer = collections.deque([0] * buffer_len, maxlen=buffer_len)
        
        self.is_running = True

    def add_data(self, avg_val, blink_state):
        """
        將最新的移動平均值加入繪圖緩衝區
        :param avg_val: 從 Detector 取得的 debug_avg
        :param blink_state: 當前是否觸發 (0 或 1)
        """
        self.avg_buffer.append(avg_val)
        self.state_buffer.append(blink_state)

    def _init_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.fig.canvas.manager.set_window_title('Blink Detector: Moving Average')
        
        # 設定座標軸
        self.ax.set_xlim(0, self.buffer_len)
        self.ax.set_ylim(self.y_range)
        self.ax.set_title("Moving Average Signal vs Threshold")
        self.ax.grid(True, alpha=0.3)
        
        # 1. 繪製閾值線 (紅色虛線)
        self.ax.axhline(y=self.threshold, color='r', linestyle='--', linewidth=1.5, label='Threshold')
        
        # 2. 繪製移動平均線 (藍色實線)
        self.x_data = np.arange(self.buffer_len)
        self.line_avg, = self.ax.plot(self.x_data, np.zeros(self.buffer_len), 'b-', linewidth=1.2, label='Moving Avg')
        
        # 3. 狀態指示 (當眨眼時，線條變粗或變色，這裡用文字顯示)
        self.status_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes, 
                                        fontsize=12, fontweight='bold',
                                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        self.ax.legend(loc='upper right')

    def _update(self, frame):
        if not self.is_running:
            plt.close(self.fig)
            return
        
        # 更新 Y 軸數據
        self.line_avg.set_ydata(self.avg_buffer)
        
        # 檢查最近的狀態
        current_state = self.state_buffer[-1]
        current_val = self.avg_buffer[-1]
        
        if current_state == 1:
            self.status_text.set_text(f"Status: BLINK DETECTED!\nLevel: {current_val:.1f}")
            self.status_text.set_color('red')
        else:
            self.status_text.set_text(f"Status: Monitoring\nLevel: {current_val:.1f}")
            self.status_text.set_color('green')
            
        return self.line_avg, self.status_text

    def start(self):
        """啟動 GUI (阻斷式)"""
        self._init_plot()
        # interval=30ms (約33FPS)
        ani = FuncAnimation(self.fig, self._update, interval=30, blit=True)
        plt.show()
        self.is_running = False