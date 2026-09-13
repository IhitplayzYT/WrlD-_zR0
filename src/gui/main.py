import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QTableWidget, 
                             QTableWidgetItem, QTextEdit, QSplitter, QFrame,
                             QTabWidget, QComboBox, QSpinBox, QGroupBox, QGridLayout,
                             QScrollArea)
from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import QSizePolicy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from typing import Dict, List, Any
from datetime import datetime
import random


class StockTicker(QWidget):
    """Top ticker showing circulating stock prices"""
    
    def __init__(self, stock_data: Dict[str, float]):
        super().__init__()
        self.stock_data = stock_data
        self.previous_prices = stock_data.copy()
        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.setLayout(self.layout)
        self.labels = {}
        self.update_ticker()
    
    def update_prices(self, new_prices: Dict[str, float]):
        """Update stock prices and calculate changes"""
        self.previous_prices = self.stock_data.copy()
        self.stock_data = new_prices
        self.update_ticker()
    
    def update_ticker(self):
        """Update the ticker display"""
        # Clear existing labels
        for i in reversed(range(self.layout.count())):
            self.layout.itemAt(i).widget().setParent(None)
        
        self.labels = {}
        for symbol, price in self.stock_data.items():
            prev_price = self.previous_prices.get(symbol, price)
            change = price - prev_price
            change_pct = (change / prev_price * 100) if prev_price > 0 else 0
            
            label = QLabel(f"{symbol}: ${price:.2f} {change_pct:+.2f}%")
            label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            
            if change > 0:
                label.setStyleSheet("color: green; padding: 5px;")
            elif change < 0:
                label.setStyleSheet("color: red; padding: 5px;")
            else:
                label.setStyleSheet("color: white; padding: 5px;")
            
            self.layout.addWidget(label)
            self.labels[symbol] = label


class PriceGraph(QWidget):
    """Live price graph with matplotlib"""
    
    def __init__(self, price_history: Dict[str, List[float]]):
        super().__init__()
        self.price_history = price_history
        self.selected_stock = list(price_history.keys())[0] if price_history else "AAPL"
        self.zoom_level = 1.0
        self.black_box_periods = []  # List of (start, end) tuples for black box periods
        self.active_black_box = None  # Currently active black box period (start, end)
        self.historical_black_boxes = []  # Historical black box periods for highlights
        
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        
        # Initial plot
        self.update_graph()
    
    def add_black_box_period(self, start_idx: int, end_idx: int):
        """Add a black box period to highlight"""
        self.active_black_box = (start_idx, end_idx)
        self.update_graph()
    
    def end_black_box_period(self):
        """End the current black box period and move to history"""
        if self.active_black_box:
            self.historical_black_boxes.append(self.active_black_box)
            self.active_black_box = None
            self.update_graph()
    
    def clear_black_box_periods(self):
        """Clear all black box periods"""
        self.black_box_periods = []
        self.active_black_box = None
        self.historical_black_boxes = []
        self.update_graph()
    
    def update_graph(self):
        """Update the price graph"""
        self.figure.clear()
        
        if self.selected_stock not in self.price_history:
            return
        
        prices = self.price_history[self.selected_stock]
        if not prices:
            return
        
        ax = self.figure.add_subplot(111)
        
        # Plot price line
        x = range(len(prices))
        ax.plot(x, prices, 'b-', linewidth=2)
        
        # Color based on direction
        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                ax.plot([i-1, i], [prices[i-1], prices[i]], 'g-', linewidth=2)
            elif prices[i] < prices[i-1]:
                ax.plot([i-1, i], [prices[i-1], prices[i]], 'r-', linewidth=2)
        
        # Highlight active black box period with greyed out region
        if self.active_black_box:
            start, end = self.active_black_box
            if start < len(prices) and end <= len(prices):
                ax.axvspan(start, end, alpha=0.5, color='gray', label='Active Black Box')
        
        # Highlight historical black box periods with small highlights on axes
        for start, end in self.historical_black_boxes:
            if start < len(prices) and end <= len(prices):
                # Add small highlight region on x-axis
                ax.axvspan(start, end, alpha=0.15, color='yellow', ymin=0, ymax=0.05)
                # Add vertical line markers at boundaries
                ax.axvline(x=start, color='orange', linestyle=':', alpha=0.7, linewidth=1)
                ax.axvline(x=end, color='orange', linestyle=':', alpha=0.7, linewidth=1)
        
        ax.set_title(f"{self.selected_stock} Price Chart")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")
        ax.grid(True, alpha=0.3)
        
        # Add legend if there are black box periods
        if self.active_black_box or self.historical_black_boxes:
            legend_elements = []
            if self.active_black_box:
                from matplotlib.patches import Patch
                legend_elements.append(Patch(facecolor='gray', alpha=0.5, label='Active Black Box'))
            if self.historical_black_boxes:
                from matplotlib.patches import Patch
                legend_elements.append(Patch(facecolor='yellow', alpha=0.15, label='Historical Black Box'))
            if legend_elements:
                ax.legend(handles=legend_elements, loc='upper right')
        
        self.canvas.draw()
    
    def set_selected_stock(self, stock: str):
        """Set the selected stock for display"""
        self.selected_stock = stock
        self.update_graph()
    
    def update_price_history(self, price_history: Dict[str, List[float]]):
        """Update price history"""
        self.price_history = price_history
        self.update_graph()


class NewsLog(QWidget):
    """Right sidebar for news and events"""
    
    def __init__(self):
        super().__init__()
        self.entries: List[Dict[str, Any]] = []
        
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        self.label = QLabel("News & Events")
        self.label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.layout.addWidget(self.label)
        
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.layout.addWidget(self.text_area)
    
    def add_entry(self, entry_type: str, message: str, timestamp: datetime = None):
        """Add a news or event entry"""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.entries.append({
            'type': entry_type,
            'message': message,
            'timestamp': timestamp
        })
        
        # Keep only last 100 entries
        if len(self.entries) > 100:
            self.entries = self.entries[-100:]
        
        self.update_display()
    
    def update_display(self):
        """Update the news log display"""
        text = ""
        for entry in reversed(self.entries[-20:]):  # Show last 20 entries
            timestamp = entry['timestamp'].strftime("%H:%M:%S")
            entry_type = entry['type']
            message = entry['message']
            
            # Color code by type
            if entry_type == "Kaiser Election":
                color = "yellow"
            elif entry_type == "Villain Move":
                color = "red"
            elif entry_type == "News":
                color = "cyan"
            else:
                color = "white"
            
            text += f"[{timestamp}] {entry_type}: {message}<br>"
        
        self.text_area.setHtml(text)


class AgentInspector(QWidget):
    """Agent inspection and modification panel"""
    
    def __init__(self, agents: Dict[str, Any]):
        super().__init__()
        self.agents = agents
        self.selected_agent = None
        
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        self.label = QLabel("Agent Inspector")
        self.label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.layout.addWidget(self.label)
        
        self.agent_combo = QComboBox()
        self.agent_combo.addItems(list(agents.keys()))
        self.agent_combo.currentTextChanged.connect(self.select_agent)
        self.layout.addWidget(self.agent_combo)
        
        self.info_area = QTextEdit()
        self.info_area.setReadOnly(True)
        self.layout.addWidget(self.info_area)
        
        if agents:
            self.select_agent(list(agents.keys())[0])
    
    def select_agent(self, agent_id: str):
        """Select an agent for inspection"""
        self.selected_agent = agent_id
        self.update_display()
    
    def update_display(self):
        """Update agent inspection display"""
        if self.selected_agent is None:
            self.info_area.setText("Select an agent to inspect")
            return
        
        if self.selected_agent not in self.agents:
            self.info_area.setText(f"Agent {self.selected_agent} not found")
            return
        
        agent = self.agents[self.selected_agent]
        
        # Display agent properties
        text = f"Agent ID: {self.selected_agent}<br>"
        text += f"Type: {agent.get('agent_type', 'Unknown')}<br>"
        text += f"Cash: ${agent.get('cash', 0):.2f}<br>"
        
        # Display multi-asset holdings
        holdings = agent.get('holdings', {})
        if isinstance(holdings, dict):
            text += "Holdings:<br>"
            for asset, qty in holdings.items():
                text += f"  {asset}: {qty}<br>"
        else:
            text += f"Holdings: {holdings}<br>"
        
        text += f"Total Value: ${agent.get('total_value', 0):.2f}<br>"
        
        # Add villain points if applicable
        if 'villain_points' in agent:
            text += f"Villain Points: {agent['villain_points']}<br>"
        
        self.info_area.setHtml(text)


class AgentPredictionGraph(QWidget):
    """Small graph showing agent predictions"""
    
    def __init__(self, agent_id: str, actual_prices: List[float], predicted_prices: List[float]):
        super().__init__()
        self.agent_id = agent_id
        self.actual_prices = actual_prices
        self.predicted_prices = predicted_prices
        self.accuracy = self.calculate_accuracy()
        
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # Agent label
        self.label = QLabel(f"{agent_id} (Acc: {self.accuracy:.2f})")
        self.label.setFont(QFont("Arial", 8, QFont.Weight.Bold))
        self.layout.addWidget(self.label)
        
        # Create small matplotlib figure
        self.figure = Figure(figsize=(3, 2), dpi=80)
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        
        self.update_graph()
    
    def calculate_accuracy(self) -> float:
        """Calculate prediction accuracy (1 - MAPE)"""
        if len(self.actual_prices) != len(self.predicted_prices) or len(self.actual_prices) == 0:
            return 0.0
        
        errors = []
        for actual, predicted in zip(self.actual_prices, self.predicted_prices):
            if actual != 0:
                errors.append(abs((actual - predicted) / actual))
        
        if not errors:
            return 0.0
        
        mape = sum(errors) / len(errors)
        return max(0.0, 1.0 - mape)
    
    def update_graph(self):
        """Update the prediction graph"""
        self.figure.clear()
        
        if not self.actual_prices or not self.predicted_prices:
            return
        
        ax = self.figure.add_subplot(111)
        
        x = range(len(self.actual_prices))
        
        # Plot actual prices in blue
        ax.plot(x, self.actual_prices, 'b-', linewidth=1, label='Actual')
        
        # Plot predicted prices in red with dashed line
        ax.plot(x, self.predicted_prices, 'r--', linewidth=1, label='Predicted')
        
        ax.set_title(f"{self.agent_id}", fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        
        self.canvas.draw()


class AgentPredictionsPanel(QWidget):
    """Horizontal scrollable panel for agent predictions"""
    
    def __init__(self):
        super().__init__()
        self.agent_predictions: Dict[str, Dict[str, Any]] = {}
        
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        self.label = QLabel("Agent Predictions (Sorted by Accuracy)")
        self.label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.layout.addWidget(self.label)
        
        # Create scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # Container for graphs
        self.graphs_container = QWidget()
        self.graphs_layout = QHBoxLayout()
        self.graphs_container.setLayout(self.graphs_layout)
        self.graphs_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        
        self.scroll_area.setWidget(self.graphs_container)
        self.layout.addWidget(self.scroll_area)
    
    def update_predictions(self, actual_prices: List[float], agent_predictions: Dict[str, List[float]]):
        """Update agent predictions and sort by accuracy"""
        self.agent_predictions = {}
        
        # Calculate accuracy for each agent
        for agent_id, predictions in agent_predictions.items():
            accuracy = self._calculate_accuracy(actual_prices, predictions)
            self.agent_predictions[agent_id] = {
                'predictions': predictions,
                'accuracy': accuracy
            }
        
        # Sort by accuracy (highest first)
        sorted_agents = sorted(self.agent_predictions.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        # Clear existing graphs
        for i in reversed(range(self.graphs_layout.count())):
            self.graphs_layout.itemAt(i).widget().setParent(None)
        
        # Add new graphs in sorted order
        for agent_id, data in sorted_agents:
            graph = AgentPredictionGraph(agent_id, actual_prices, data['predictions'])
            self.graphs_layout.addWidget(graph)
    
    def _calculate_accuracy(self, actual: List[float], predicted: List[float]) -> float:
        """Calculate prediction accuracy (1 - MAPE)"""
        if len(actual) != len(predicted) or len(actual) == 0:
            return 0.0
        
        errors = []
        for a, p in zip(actual, predicted):
            if a != 0:
                errors.append(abs((a - p) / a))
        
        if not errors:
            return 0.0
        
        mape = sum(errors) / len(errors)
        return max(0.0, 1.0 - mape)


class VillainShop(QWidget):
    """Villain points shop for purchasing events with dynamic pricing"""
    
    def __init__(self):
        super().__init__()
        self.shop_items = {
            'market_crash': {
                'base_cost': 100, 
                'impact': 'high', 
                'probability': 0.01,
                'duration_options': [5, 10, 20],
                'timing_options': ['immediate', 'delayed']
            },
            'price_spike': {
                'base_cost': 50, 
                'impact': 'medium', 
                'probability': 0.05,
                'duration_options': [3, 5, 10],
                'timing_options': ['immediate', 'delayed']
            },
            'fake_news': {
                'base_cost': 30, 
                'impact': 'low', 
                'probability': 0.1,
                'duration_options': [2, 5],
                'timing_options': ['immediate']
            },
            'extended_duration': {
                'base_cost': 20, 
                'impact': 'duration', 
                'probability': 0.0,
                'duration_options': [5, 10, 15],
                'timing_options': ['immediate']
            },
            'kaiser_election': {
                'base_cost': 200,
                'impact': 'critical',
                'probability': 0.005,
                'duration_options': [10, 20, 30],
                'timing_options': ['immediate', 'delayed']
            }
        }
        self.villain_points = 0
        self.selected_item = None
        self.selected_duration = 5
        self.selected_timing = 'immediate'
        
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        self.label = QLabel("Villain Shop")
        self.label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.layout.addWidget(self.label)
        
        self.points_label = QLabel(f"Villain Points: {self.villain_points}")
        self.points_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.layout.addWidget(self.points_label)
        
        self.item_combo = QComboBox()
        self.item_combo.addItems(list(self.shop_items.keys()))
        self.item_combo.currentTextChanged.connect(self.select_item)
        self.layout.addWidget(self.item_combo)
        
        self.duration_combo = QComboBox()
        self.layout.addWidget(QLabel("Duration:"))
        self.layout.addWidget(self.duration_combo)
        
        self.timing_combo = QComboBox()
        self.layout.addWidget(QLabel("Timing:"))
        self.layout.addWidget(self.timing_combo)
        
        self.cost_label = QLabel("Cost: 0 pts")
        self.cost_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.layout.addWidget(self.cost_label)
        
        self.purchase_button = QPushButton("Purchase")
        self.purchase_button.clicked.connect(self.purchase_item)
        self.layout.addWidget(self.purchase_button)
        
        if self.shop_items:
            self.select_item(list(self.shop_items.keys())[0])
    
    def set_villain_points(self, points: int):
        """Set available villain points"""
        self.villain_points = points
        self.points_label.setText(f"Villain Points: {self.villain_points}")
        self.update_cost_display()
    
    def calculate_dynamic_cost(self, item_name: str, duration: int = 5, timing: str = 'immediate') -> int:
        """Calculate dynamic cost based on event probability, impact, and options"""
        if item_name not in self.shop_items:
            return 0
        
        item = self.shop_items[item_name]
        base_cost = item['base_cost']
        probability = item['probability']
        
        # Lower probability events cost more
        probability_multiplier = 1.0 / (probability + 0.01) if probability > 0 else 1.0
        
        # Duration multiplier
        duration_multiplier = 1.0 + (duration / 10.0)
        
        # Timing multiplier (delayed costs more due to planning)
        timing_multiplier = 1.5 if timing == 'delayed' else 1.0
        
        # Calculate final cost
        final_cost = int(base_cost * probability_multiplier * duration_multiplier * timing_multiplier)
        
        return final_cost
    
    def select_item(self, item_name: str):
        """Select an item for purchase"""
        self.selected_item = item_name
        if item_name in self.shop_items:
            item = self.shop_items[item_name]
            
            # Update duration combo
            self.duration_combo.clear()
            self.duration_combo.addItems([str(d) for d in item['duration_options']])
            self.duration_combo.currentTextChanged.connect(self.set_duration)
            
            # Update timing combo
            self.timing_combo.clear()
            self.timing_combo.addItems(item['timing_options'])
            self.timing_combo.currentTextChanged.connect(self.set_timing)
            
            self.update_cost_display()
    
    def set_duration(self, duration_str: str):
        """Set duration for selected item"""
        self.selected_duration = int(duration_str)
        self.update_cost_display()
    
    def set_timing(self, timing: str):
        """Set timing for selected item"""
        self.selected_timing = timing
        self.update_cost_display()
    
    def update_cost_display(self):
        """Update cost display"""
        if self.selected_item:
            cost = self.calculate_dynamic_cost(self.selected_item, self.selected_duration, self.selected_timing)
            self.cost_label.setText(f"Cost: {cost} pts")
            
            can_afford = self.villain_points >= cost
            self.purchase_button.setEnabled(can_afford)
    
    def purchase_item(self):
        """Purchase selected item from the shop"""
        if self.selected_item:
            cost = self.calculate_dynamic_cost(self.selected_item, self.selected_duration, self.selected_timing)
            
            if self.villain_points >= cost:
                self.villain_points -= cost
                self.points_label.setText(f"Villain Points: {self.villain_points}")
                self.update_cost_display()
                return True
        return False


class TradingGUI(QMainWindow):
    def __init__(self, simulation_state: Dict[str, Any]):
        super().__init__()
        self.simulation_state = simulation_state
        self.stock_data = simulation_state.get('current_stock_data', {})
        self.price_history = simulation_state.get('price_history', {})
        self.agents = simulation_state.get('agents', {})
        self.news_events = simulation_state.get('news_events', [])
        self.paused = False
        self.update_counter = 0
        self.black_box_active = False
        self.black_box_start_idx = 0
        self.black_box_duration = simulation_state.get('black_box_duration', 5)
        self.black_box_frequency = simulation_state.get('black_box_frequency', 20)
        self.rng_bbox = simulation_state.get('rng_bbox', None)
        
        self.setWindowTitle("RLAIF Trading Simulation")
        self.setGeometry(100, 100, 1400, 900)
        
        self.setup_ui()
        self.setup_timer()
    
    def setup_ui(self):
        """Setup the GUI layout"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Stock ticker at top
        self.stock_ticker = StockTicker(self.stock_data)
        main_layout.addWidget(self.stock_ticker)
        
        # Main content area with splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side - price graph
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_widget.setLayout(left_layout)
        
        # Stock selector
        self.stock_combo = QComboBox()
        self.stock_combo.addItems(list(self.stock_data.keys()))
        self.stock_combo.currentTextChanged.connect(self.change_stock)
        left_layout.addWidget(QLabel("Select Stock:"))
        left_layout.addWidget(self.stock_combo)
        
        self.price_graph = PriceGraph(self.price_history)
        left_layout.addWidget(self.price_graph)
        
        splitter.addWidget(left_widget)
        
        # Right side - sidebar
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_widget.setLayout(right_layout)
        
        # Tab widget for sidebar
        self.tab_widget = QTabWidget()
        
        # News tab
        news_tab = QWidget()
        news_layout = QVBoxLayout()
        news_tab.setLayout(news_layout)
        self.news_log = NewsLog()
        news_layout.addWidget(self.news_log)
        self.tab_widget.addTab(news_tab, "News")
        
        # Agent inspector tab
        agent_tab = QWidget()
        agent_layout = QVBoxLayout()
        agent_tab.setLayout(agent_layout)
        self.agent_inspector = AgentInspector(self.agents)
        agent_layout.addWidget(self.agent_inspector)
        self.tab_widget.addTab(agent_tab, "Agents")
        
        # Agent predictions tab
        predictions_tab = QWidget()
        predictions_layout = QVBoxLayout()
        predictions_tab.setLayout(predictions_layout)
        self.agent_predictions_panel = AgentPredictionsPanel()
        predictions_layout.addWidget(self.agent_predictions_panel)
        self.tab_widget.addTab(predictions_tab, "Predictions")
        
        # Villain shop tab
        shop_tab = QWidget()
        shop_layout = QVBoxLayout()
        shop_tab.setLayout(shop_layout)
        self.villain_shop = VillainShop()
        shop_layout.addWidget(self.villain_shop)
        self.tab_widget.addTab(shop_tab, "Shop")
        
        right_layout.addWidget(self.tab_widget)
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.toggle_pause)
        control_layout.addWidget(self.pause_button)
        
        self.quit_button = QPushButton("Quit")
        self.quit_button.clicked.connect(self.close)
        control_layout.addWidget(self.quit_button)
        
        right_layout.addLayout(control_layout)
        
        splitter.addWidget(right_widget)
        
        # Set splitter proportions
        splitter.setSizes([900, 900])
        
        main_layout.addWidget(splitter)
    
    def setup_timer(self):
        """Setup update timer"""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation_data)
        self.timer.start(100)  # Update every 100ms
    
    def change_stock(self, stock_name: str):
        """Change selected stock"""
        self.price_graph.set_selected_stock(stock_name)
    
    def toggle_pause(self):
        """Toggle pause state"""
        self.paused = not self.paused
        self.pause_button.setText("Resume" if self.paused else "Pause")
    
    def update_simulation_data(self):
        """Update simulation data from state"""
        if self.paused:
            return
            
        self.update_counter += 1
        
        # Handle black box events
        if self.rng_bbox and not self.black_box_active:
            # Check if it's time for a black box event
            if self.update_counter % self.black_box_frequency == 0:
                self.black_box_active = True
                self.black_box_start_idx = self.update_counter
                # Highlight black box period on graph
                if self.selected_stock in self.price_history:
                    current_idx = len(self.price_history[self.selected_stock])
                    self.price_graph.add_black_box_period(current_idx, current_idx + self.black_box_duration)
        
        # Check if black box period should end
        if self.black_box_active:
            if self.update_counter - self.black_box_start_idx >= self.black_box_duration:
                self.black_box_active = False
                # End the black box period visualization
                self.price_graph.end_black_box_period()
                # Resample black box parameters if rng_bbox is set
                if self.rng_bbox:
                    min_dur, max_dur, min_freq, max_freq = self.rng_bbox
                    self.black_box_duration = random.randint(min_dur, max_dur)
                    self.black_box_frequency = random.randint(min_freq, max_freq)
                    print(f"Resampled black box parameters: duration={self.black_box_duration}, frequency={self.black_box_frequency}")
        
        # Update stock ticker with simulated price changes
        if self.update_counter % 5 == 0:  # Update prices every 5 ticks
            for symbol in self.stock_data:
                change = random.uniform(-2, 2)
                self.stock_data[symbol] = max(1.0, self.stock_data[symbol] + change)
            self.stock_ticker.update_prices(self.stock_data)
        
        # Update price history
        if self.update_counter % 10 == 0:  # Update history every 10 ticks
            for symbol in self.stock_data:
                if symbol not in self.price_history:
                    self.price_history[symbol] = []
                self.price_history[symbol].append(self.stock_data[symbol])
                if len(self.price_history[symbol]) > 100:
                    self.price_history[symbol] = self.price_history[symbol][-100:]
            self.price_graph.update_price_history(self.price_history)
        
        # Update news log (only add new events occasionally)
        if self.update_counter % 20 == 0:  # Add news every 20 ticks
            news_types = ["News", "Villain Move", "Kaiser Election"]
            news_messages = [
                "Market volatility increases",
                "Price manipulation detected",
                "New Kaiser election announced",
                "Sector rotation observed",
                "Liquidity concerns in tech stocks"
            ]
            self.news_log.add_entry(
                random.choice(news_types),
                random.choice(news_messages)
            )
        
        # Update agent predictions (simulate predictions)
        if self.update_counter % 30 == 0 and self.selected_stock in self.price_history:
            actual_prices = self.price_history[self.selected_stock][-20:]  # Last 20 prices
            agent_predictions = {}
            for agent_id in self.agents.keys():
                # Simulate predictions with some noise
                predictions = [p + random.uniform(-5, 5) for p in actual_prices]
                agent_predictions[agent_id] = predictions
            self.agent_predictions_panel.update_predictions(actual_prices, agent_predictions)
        
        # Update villain shop
        if 'villain_points' in self.simulation_state:
            self.villain_shop.set_villain_points(self.simulation_state['villain_points'])


def launch_gui(simulation_state: Dict[str, Any]):
    """Launch the GUI application"""
    app = QApplication(sys.argv)
    window = TradingGUI(simulation_state)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    # Test GUI with mock data
    mock_state = {
        'stock_data': {
            'AAPL': 150.0,
            'GOOGL': 2800.0,
            'MSFT': 300.0,
            'TSLA': 800.0,
        },
        'price_history': {
            'AAPL': [145.0, 146.0, 147.0, 148.0, 149.0, 150.0],
            'GOOGL': [2750.0, 2760.0, 2770.0, 2780.0, 2790.0, 2800.0],
            'MSFT': [295.0, 296.0, 297.0, 298.0, 299.0, 300.0],
            'TSLA': [795.0, 796.0, 797.0, 798.0, 799.0, 800.0],
        },
        'agents': {
            'agent_0': {'agent_type': 'rlaif', 'cash': 10000, 'holdings': 10, 'total_value': 11500},
            'agent_1': {'agent_type': 'adversarial', 'cash': 10000, 'holdings': 5, 'total_value': 10400},
        },
        'news_events': [
            {'type': 'News', 'message': 'Market opens higher'},
            {'type': 'Villain Move', 'message': 'Price manipulation detected'},
        ],
        'villain_points': 150,
    }
    
    launch_gui(mock_state)
