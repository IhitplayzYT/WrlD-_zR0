import sys
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import random

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QTextEdit,
    QSplitter, QFrame, QTabWidget, QComboBox, QSpinBox, QDoubleSpinBox,
    QGroupBox, QGridLayout, QScrollArea, QSlider, QHeaderView, QLineEdit,
    QMessageBox, QSizePolicy
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QColor, QFont

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np


QSS_THEME = """
QMainWindow, QWidget {
    background-color: #12131C;
    color: #E0E6ED;
    font-family: 'Segoe UI', Inter, Helvetica, Arial, sans-serif;
    font-size: 13px;
}
QGroupBox {
    border: 1px solid #2A2C3D;
    border-radius: 8px;
    margin-top: 14px;
    padding-top: 12px;
    font-weight: bold;
    color: #7AA2F7;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}
QTableWidget {
    background-color: #1A1B26;
    gridline-color: #2A2C3D;
    border: 1px solid #2A2C3D;
    border-radius: 6px;
    color: #C0CAF5;
    selection-background-color: #3B4261;
    selection-color: #7AA2F7;
}
QHeaderView::section {
    background-color: #242536;
    color: #7AA2F7;
    padding: 6px;
    border: 1px solid #2A2C3D;
    font-weight: bold;
}
QPushButton {
    background-color: #242536;
    color: #7AA2F7;
    border: 1px solid #3B4261;
    border-radius: 6px;
    padding: 6px 14px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #3B4261;
    color: #BB9AF7;
    border-color: #7AA2F7;
}
QPushButton:pressed {
    background-color: #7AA2F7;
    color: #15161E;
}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
    background-color: #1A1B26;
    border: 1px solid #3B4261;
    border-radius: 4px;
    color: #C0CAF5;
    padding: 4px 8px;
}
QTabWidget::pane {
    border: 1px solid #2A2C3D;
    border-radius: 6px;
    background: #1A1B26;
}
QTabBar::tab {
    background: #12131C;
    color: #A9B1D6;
    padding: 8px 18px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    border: 1px solid #2A2C3D;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background: #242536;
    color: #7AA2F7;
    font-weight: bold;
    border-bottom: 2px solid #7AA2F7;
}
QSlider::groove:horizontal {
    border: 1px solid #2A2C3D;
    height: 6px;
    background: #1A1B26;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #7AA2F7;
    width: 14px;
    margin-top: -4px;
    margin-bottom: -4px;
    border-radius: 7px;
}
QScrollBar:vertical, QScrollBar:horizontal {
    background: #12131C;
    border: none;
    width: 8px;
    height: 8px;
}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background: #2A2C3D;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {
    background: #7AA2F7;
}
"""


class StockTicker(QWidget):
    """Top ticker bar displaying live prices across circulating stocks"""
    
    def __init__(self, stock_data: Dict[str, float]):
        super().__init__()
        self.stock_data = stock_data
        self.previous_prices = stock_data.copy()
        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(6, 4, 6, 4)
        self.layout.setSpacing(12)
        self.setLayout(self.layout)
        self.update_ticker()
    
    def update_prices(self, new_prices: Dict[str, float]):
        self.previous_prices = self.stock_data.copy()
        self.stock_data = new_prices.copy()
        self.update_ticker()
    
    def update_ticker(self):
        for i in reversed(range(self.layout.count())):
            item = self.layout.itemAt(i)
            if item and item.widget():
                item.widget().setParent(None)
                
        for symbol, price in self.stock_data.items():
            prev_price = self.previous_prices.get(symbol, price)
            change = price - prev_price
            change_pct = (change / prev_price * 100.0) if prev_price > 0 else 0.0
            
            card = QFrame()
            card.setStyleSheet("""
                QFrame {
                    background-color: #1A1B26;
                    border: 1px solid #2A2C3D;
                    border-radius: 6px;
                    padding: 2px 8px;
                }
            """)
            card_layout = QHBoxLayout(card)
            card_layout.setContentsMargins(4, 2, 4, 2)
            
            sym_lbl = QLabel(f"{symbol}")
            sym_lbl.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            sym_lbl.setStyleSheet("color: #7AA2F7;")
            
            price_lbl = QLabel(f"${price:.2f}")
            price_lbl.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            
            badge_text = f"{change_pct:+.2f}%"
            badge_lbl = QLabel(badge_text)
            badge_lbl.setFont(QFont("Arial", 9, QFont.Weight.Bold))
            
            if change > 0:
                badge_lbl.setStyleSheet("color: #9ECE6A; background: #1F342B; border-radius: 3px; padding: 2px 4px;")
            elif change < 0:
                badge_lbl.setStyleSheet("color: #F7768E; background: #37222E; border-radius: 3px; padding: 2px 4px;")
            else:
                badge_lbl.setStyleSheet("color: #A9B1D6; background: #242536; border-radius: 3px; padding: 2px 4px;")
                
            card_layout.addWidget(sym_lbl)
            card_layout.addWidget(price_lbl)
            card_layout.addWidget(badge_lbl)
            
            self.layout.addWidget(card)
        self.layout.addStretch()


class PriceGraph(QWidget):
    """Full-width horizontal top slab displaying stock price chart with Matplotlib"""
    
    def __init__(self, price_history: Dict[str, List[float]]):
        super().__init__()
        self.price_history = price_history
        self.selected_stock = list(price_history.keys())[0] if price_history else "AAPL"
        self.active_black_box = None
        self.historical_black_boxes = []
        
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(4, 2, 4, 2)
        self.setLayout(self.layout)
        
        # Figure with dark theme styling
        self.figure = Figure(figsize=(14, 4), dpi=100, facecolor='#12131C')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.layout.addWidget(self.canvas)
        
        self.update_graph()

    def add_black_box_period(self, start_idx: int, end_idx: int):
        self.active_black_box = (start_idx, end_idx)
        self.update_graph()
    
    def end_black_box_period(self):
        if self.active_black_box:
            self.historical_black_boxes.append(self.active_black_box)
            self.active_black_box = None
            self.update_graph()
            
    def clear_black_box_periods(self):
        self.active_black_box = None
        self.historical_black_boxes = []
        self.update_graph()

    def set_selected_stock(self, stock: str):
        self.selected_stock = stock
        self.update_graph()

    def update_price_history(self, price_history: Dict[str, List[float]]):
        self.price_history = price_history
        self.update_graph()

    def update_graph(self):
        self.figure.clear()
        
        prices = self.price_history.get(self.selected_stock, [])
        if not prices:
            ax = self.figure.add_subplot(111, facecolor='#1A1B26')
            ax.text(0.5, 0.5, "No Price Data Available", color='#7AA2F7', ha='center', va='center', fontsize=12)
            self.canvas.draw()
            return
            
        ax = self.figure.add_subplot(111, facecolor='#1A1B26')
        ax.tick_params(colors='#A9B1D6', labelsize=9)
        for spine in ax.spines.values():
            spine.set_color('#2A2C3D')
            
        x = list(range(len(prices)))
        ax.plot(x, prices, color='#7AA2F7', linewidth=2.0, label=f'{self.selected_stock} Live Price')
        
        # Color gradient segments
        for i in range(1, len(prices)):
            if prices[i] >= prices[i-1]:
                ax.plot([i-1, i], [prices[i-1], prices[i]], color='#9ECE6A', linewidth=2.0)
            else:
                ax.plot([i-1, i], [prices[i-1], prices[i]], color='#F7768E', linewidth=2.0)
                
        # 5-step Moving Average line
        if len(prices) >= 5:
            ma = np.convolve(prices, np.ones(5)/5, mode='valid')
            ma_x = list(range(4, len(prices)))
            ax.plot(ma_x, ma, color='#E0AF68', linestyle='--', linewidth=1.2, label='5-Step MA')

        # Black box spans
        if self.active_black_box:
            st, ed = self.active_black_box
            if st < len(prices):
                ax.axvspan(st, min(ed, len(prices)), color='#565F89', alpha=0.45, label='Active Black Box')
                
        for st, ed in self.historical_black_boxes:
            if st < len(prices):
                ax.axvspan(st, min(ed, len(prices)), color='#E0AF68', alpha=0.15)
                ax.axvline(x=st, color='#BB9AF7', linestyle=':', alpha=0.6)

        ax.set_title(f"Market Horizontal Slab: {self.selected_stock} Real-Time Price Trend", color='#7AA2F7', fontsize=11, fontweight='bold')
        ax.set_xlabel("Timestep", color='#A9B1D6', fontsize=9)
        ax.set_ylabel("Price ($)", color='#A9B1D6', fontsize=9)
        ax.grid(True, color='#292A3E', linestyle='-', linewidth=0.7, alpha=0.8)
        ax.legend(facecolor='#1A1B26', edgecolor='#2A2C3D', labelcolor='#C0CAF5', loc='upper left', fontsize=8)
        
        self.figure.tight_layout()
        self.canvas.draw()


class AgentInspector(QWidget):
    """Live updating agent inspector showing dynamic cash, portfolio value, holdings, and PnL"""
    
    def __init__(self, agents: Dict[str, Any], stock_data: Dict[str, float]):
        super().__init__()
        self.agents = agents
        self.stock_data = stock_data
        self.selected_agent = None
        
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # Header & Summary Bar
        header_layout = QHBoxLayout()
        title_lbl = QLabel("Agent Live Resource Inspector")
        title_lbl.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        title_lbl.setStyleSheet("color: #7AA2F7;")
        header_layout.addWidget(title_lbl)
        
        self.summary_lbl = QLabel("Total Agents: 0 | Total Wealth: $0.00")
        self.summary_lbl.setStyleSheet("color: #A9B1D6; font-weight: bold;")
        header_layout.addStretch()
        header_layout.addWidget(self.summary_lbl)
        self.layout.addLayout(header_layout)
        
        # Splitter: Left Table, Right Agent Detail & Live Resource Tuning
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Agent Table
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "Agent ID", "Type", "Cash ($)", "Holdings", "Total Value ($)", "PnL ($)", "PnL (%)"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.itemSelectionChanged.connect(self._on_table_selection)
        splitter.addWidget(self.table)
        
        # Detail & Fine-tune Panel
        detail_widget = QWidget()
        detail_layout = QVBoxLayout()
        detail_widget.setLayout(detail_layout)
        
        detail_group = QGroupBox("Selected Agent Live Resources & Alteration")
        group_layout = QVBoxLayout()
        detail_group.setLayout(group_layout)
        
        self.detail_area = QTextEdit()
        self.detail_area.setReadOnly(True)
        self.detail_area.setMaximumHeight(160)
        group_layout.addWidget(self.detail_area)
        
        # Control Buttons
        ctrl_grid = QGridLayout()
        
        self.cash_spin = QDoubleSpinBox()
        self.cash_spin.setRange(-1e6, 1e7)
        self.cash_spin.setValue(1000.0)
        self.cash_spin.setPrefix("$")
        ctrl_grid.addWidget(QLabel("Cash Adjust:"), 0, 0)
        ctrl_grid.addWidget(self.cash_spin, 0, 1)
        
        btn_add_cash = QPushButton("+ Inject Cash")
        btn_add_cash.clicked.connect(self._inject_cash)
        ctrl_grid.addWidget(btn_add_cash, 0, 2)
        
        self.stock_spin = QSpinBox()
        self.stock_spin.setRange(-1000, 1000)
        self.stock_spin.setValue(10)
        ctrl_grid.addWidget(QLabel("Holdings Adjust:"), 1, 0)
        ctrl_grid.addWidget(self.stock_spin, 1, 1)
        
        btn_grant_stock = QPushButton("+ Grant Shares")
        btn_grant_stock.clicked.connect(self._grant_stock)
        ctrl_grid.addWidget(btn_grant_stock, 1, 2)
        
        btn_toggle_type = QPushButton("Toggle Type (RL / Adversarial)")
        btn_toggle_type.clicked.connect(self._toggle_agent_type)
        ctrl_grid.addWidget(btn_toggle_type, 2, 0, 1, 3)
        
        group_layout.addLayout(ctrl_grid)
        detail_layout.addWidget(detail_group)
        detail_layout.addStretch()
        
        splitter.addWidget(detail_widget)
        splitter.setSizes([650, 350])
        
        self.layout.addWidget(splitter)
        self.update_agents(self.agents, self.stock_data)

    def update_agents(self, agents: Dict[str, Any], stock_data: Dict[str, float]):
        self.agents = agents
        self.stock_data = stock_data
        
        self.table.setRowCount(len(agents))
        total_wealth = 0.0
        
        for row, (agent_id, data) in enumerate(agents.items()):
            cash = data.get('cash', 0.0)
            holdings = data.get('holdings', {})
            initial_cash = data.get('initial_cash', 10000.0)
            
            # Compute live portfolio value based on latest stock prices
            holdings_val = 0.0
            if isinstance(holdings, dict):
                for stk, qty in holdings.items():
                    p = stock_data.get(stk, 100.0)
                    holdings_val += qty * p
                holdings_str = ", ".join([f"{stk}:{qty}" for stk, qty in holdings.items() if qty != 0])
                if not holdings_str:
                    holdings_str = "None"
            else:
                p = list(stock_data.values())[0] if stock_data else 100.0
                holdings_val = float(holdings) * p
                holdings_str = f"Qty: {holdings}"
                
            tot_val = cash + holdings_val
            data['total_value'] = tot_val
            pnl = tot_val - initial_cash
            pnl_pct = (pnl / initial_cash * 100.0) if initial_cash > 0 else 0.0
            total_wealth += tot_val
            
            # Set items
            self.table.setItem(row, 0, QTableWidgetItem(agent_id))
            
            agent_type = data.get('agent_type', 'rlaif')
            type_item = QTableWidgetItem(agent_type.upper())
            if agent_type == 'adversarial':
                type_item.setForeground(QColor('#F7768E'))
            else:
                type_item.setForeground(QColor('#7AA2F7'))
            self.table.setItem(row, 1, type_item)
            
            self.table.setItem(row, 2, QTableWidgetItem(f"${cash:,.2f}"))
            self.table.setItem(row, 3, QTableWidgetItem(holdings_str))
            self.table.setItem(row, 4, QTableWidgetItem(f"${tot_val:,.2f}"))
            
            pnl_item = QTableWidgetItem(f"${pnl:+,.2f}")
            pnl_pct_item = QTableWidgetItem(f"{pnl_pct:+.2f}%")
            if pnl > 0:
                pnl_item.setForeground(QColor('#9ECE6A'))
                pnl_pct_item.setForeground(QColor('#9ECE6A'))
            elif pnl < 0:
                pnl_item.setForeground(QColor('#F7768E'))
                pnl_pct_item.setForeground(QColor('#F7768E'))
            self.table.setItem(row, 5, pnl_item)
            self.table.setItem(row, 6, pnl_pct_item)
            
        self.summary_lbl.setText(f"Total Agents: {len(agents)} | Total Wealth: ${total_wealth:,.2f}")
        
        # Keep selected detail display updated live
        if self.selected_agent and self.selected_agent in agents:
            self._display_agent_detail(self.selected_agent)
        elif len(agents) > 0 and self.selected_agent is None:
            first_agent = list(agents.keys())[0]
            self.selected_agent = first_agent
            self._display_agent_detail(first_agent)

    def _on_table_selection(self):
        selected_rows = self.table.selectedItems()
        if selected_rows:
            agent_id = self.table.item(selected_rows[0].row(), 0).text()
            self.selected_agent = agent_id
            self._display_agent_detail(agent_id)

    def _display_agent_detail(self, agent_id: str):
        if agent_id not in self.agents:
            return
        data = self.agents[agent_id]
        c = data.get('cash', 0.0)
        tv = data.get('total_value', c)
        init_c = data.get('initial_cash', 10000.0)
        pnl = tv - init_c
        
        html = f"""
        <b>Agent ID:</b> <span style="color:#7AA2F7;">{agent_id}</span><br>
        <b>Type:</b> {data.get('agent_type', 'rlaif').upper()}<br>
        <b>Live Cash:</b> ${c:,.2f}<br>
        <b>Portfolio Value:</b> ${tv:,.2f}<br>
        <b>Net Profit/Loss:</b> <span style="color:{'#9ECE6A' if pnl >= 0 else '#F7768E'};">${pnl:+,.2f}</span><br>
        <b>Holdings:</b> {data.get('holdings', {})}<br>
        """
        if 'villain_points' in data:
            html += f"<b>Villain Points:</b> {data['villain_points']}<br>"
        self.detail_area.setHtml(html)

    def _inject_cash(self):
        if not self.selected_agent or self.selected_agent not in self.agents:
            return
        amt = self.cash_spin.value()
        self.agents[self.selected_agent]['cash'] = self.agents[self.selected_agent].get('cash', 0.0) + amt
        self.update_agents(self.agents, self.stock_data)

    def _grant_stock(self):
        if not self.selected_agent or self.selected_agent not in self.agents:
            return
        qty = self.stock_spin.value()
        agent = self.agents[self.selected_agent]
        holdings = agent.get('holdings', {})
        stk = list(self.stock_data.keys())[0] if self.stock_data else "AAPL"
        if isinstance(holdings, dict):
            holdings[stk] = holdings.get(stk, 0) + qty
        else:
            agent['holdings'] = int(holdings) + qty
        self.update_agents(self.agents, self.stock_data)

    def _toggle_agent_type(self):
        if not self.selected_agent or self.selected_agent not in self.agents:
            return
        curr = self.agents[self.selected_agent].get('agent_type', 'rlaif')
        self.agents[self.selected_agent]['agent_type'] = 'adversarial' if curr == 'rlaif' else 'rlaif'
        self.update_agents(self.agents, self.stock_data)


class RuntimeTuningPanel(QWidget):
    """Runtime control panel allowing live fine-tuning of market, agent, and simulation variables"""
    
    def __init__(self, parent_gui):
        super().__init__()
        self.gui = parent_gui
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        grid = QGridLayout(content)
        
        # Group 1: Simulation Speed & Step Controls
        g_flow = QGroupBox("Simulation & Speed Control")
        f_lay = QGridLayout()
        
        f_lay.addWidget(QLabel("Timer Interval (ms):"), 0, 0)
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(10, 500)
        self.speed_slider.setValue(100)
        self.speed_slider.valueChanged.connect(self._on_speed_change)
        self.speed_lbl = QLabel("100 ms")
        f_lay.addWidget(self.speed_slider, 0, 1)
        f_lay.addWidget(self.speed_lbl, 0, 2)
        
        self.btn_pause = QPushButton("Pause / Resume")
        self.btn_pause.clicked.connect(self.gui.toggle_pause)
        f_lay.addWidget(self.btn_pause, 1, 0)
        
        self.btn_step = QPushButton("Manual Step (1 Tick)")
        self.btn_step.clicked.connect(self.gui.step_manual_tick)
        f_lay.addWidget(self.btn_step, 1, 1, 1, 2)
        
        g_flow.setLayout(f_lay)
        grid.addWidget(g_flow, 0, 0)
        
        # Group 2: Market Volatility & Fee Rates
        g_market = QGroupBox("Market & Price Dynamics Tuning")
        m_lay = QGridLayout()
        
        m_lay.addWidget(QLabel("Price Noise Multiplier:"), 0, 0)
        self.vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.vol_slider.setRange(1, 50)
        self.vol_slider.setValue(10)
        self.vol_slider.valueChanged.connect(self._on_vol_change)
        self.vol_lbl = QLabel("1.0x")
        m_lay.addWidget(self.vol_slider, 0, 1)
        m_lay.addWidget(self.vol_lbl, 0, 2)
        
        m_lay.addWidget(QLabel("Transaction Fee Rate (%):"), 1, 0)
        self.fee_spin = QDoubleSpinBox()
        self.fee_spin.setRange(0.0, 5.0)
        self.fee_spin.setSingleStep(0.1)
        self.fee_spin.setValue(0.1)
        m_lay.addWidget(self.fee_spin, 1, 1, 1, 2)
        
        g_market.setLayout(m_lay)
        grid.addWidget(g_market, 0, 1)
        
        # Group 3: Black Box Anomaly Controls
        g_bbox = QGroupBox("Black Box Anomaly Parameters")
        b_lay = QGridLayout()
        
        b_lay.addWidget(QLabel("Duration (Ticks):"), 0, 0)
        self.dur_spin = QSpinBox()
        self.dur_spin.setRange(1, 100)
        self.dur_spin.setValue(self.gui.black_box_duration)
        self.dur_spin.valueChanged.connect(self._on_dur_change)
        b_lay.addWidget(self.dur_spin, 0, 1)
        
        b_lay.addWidget(QLabel("Frequency (Ticks):"), 0, 2)
        self.freq_spin = QSpinBox()
        self.freq_spin.setRange(5, 500)
        self.freq_spin.setValue(self.gui.black_box_frequency)
        self.freq_spin.valueChanged.connect(self._on_freq_change)
        b_lay.addWidget(self.freq_spin, 0, 3)
        
        self.btn_trigger_bb = QPushButton("Trigger Black Box Now")
        self.btn_trigger_bb.clicked.connect(self.gui.trigger_black_box_now)
        b_lay.addWidget(self.btn_trigger_bb, 1, 0, 1, 2)
        
        self.btn_end_bb = QPushButton("End Black Box Period")
        self.btn_end_bb.clicked.connect(self.gui.end_black_box_now)
        b_lay.addWidget(self.btn_end_bb, 1, 2, 1, 2)
        
        g_bbox.setLayout(b_lay)
        grid.addWidget(g_bbox, 1, 0, 1, 2)
        
        # Group 4: Global Agent & Resource Injections
        g_agents = QGroupBox("Global Agent Resource Injection")
        a_lay = QGridLayout()
        
        btn_all_cash = QPushButton("Inject $5,000 Cash to ALL Agents")
        btn_all_cash.clicked.connect(self._inject_all_cash)
        a_lay.addWidget(btn_all_cash, 0, 0)
        
        btn_add_rl = QPushButton("+ Spawn New RLAIF Agent")
        btn_add_rl.clicked.connect(self._spawn_rl_agent)
        a_lay.addWidget(btn_add_rl, 0, 1)
        
        btn_add_adv = QPushButton("+ Spawn New Adversarial Agent")
        btn_add_adv.clicked.connect(self._spawn_adv_agent)
        a_lay.addWidget(btn_add_adv, 0, 2)
        
        g_agents.setLayout(a_lay)
        grid.addWidget(g_agents, 2, 0, 1, 2)
        
        # Group 5: Shock Event & News Broadcast
        g_events = QGroupBox("Runtime Market Shocks & Event Broadcast")
        e_lay = QGridLayout()
        
        btn_crash = QPushButton("Trigger Market Crash (-15%)")
        btn_crash.setStyleSheet("background-color: #37222E; color: #F7768E;")
        btn_crash.clicked.connect(self._trigger_crash)
        e_lay.addWidget(btn_crash, 0, 0)
        
        btn_spike = QPushButton("Trigger Price Surge (+15%)")
        btn_spike.setStyleSheet("background-color: #1F342B; color: #9ECE6A;")
        btn_spike.clicked.connect(self._trigger_spike)
        e_lay.addWidget(btn_spike, 0, 1)
        
        btn_kaiser = QPushButton("Announce Kaiser Election")
        btn_kaiser.setStyleSheet("background-color: #3B3822; color: #E0AF68;")
        btn_kaiser.clicked.connect(self._trigger_kaiser)
        e_lay.addWidget(btn_kaiser, 0, 2)
        
        self.news_input = QLineEdit()
        self.news_input.setPlaceholderText("Enter custom market news broadcast...")
        e_lay.addWidget(self.news_input, 1, 0, 1, 2)
        
        btn_send_news = QPushButton("Broadcast News")
        btn_send_news.clicked.connect(self._broadcast_news)
        e_lay.addWidget(btn_send_news, 1, 2)
        
        g_events.setLayout(e_lay)
        grid.addWidget(g_events, 3, 0, 1, 2)
        
        scroll.setWidget(content)
        self.layout.addWidget(scroll)

    def _on_speed_change(self, val: int):
        self.speed_lbl.setText(f"{val} ms")
        self.gui.timer.setInterval(val)

    def _on_vol_change(self, val: int):
        mult = val / 10.0
        self.vol_lbl.setText(f"{mult:.1f}x")
        self.gui.volatility_multiplier = mult

    def _on_dur_change(self, val: int):
        self.gui.black_box_duration = val

    def _on_freq_change(self, val: int):
        self.gui.black_box_frequency = val

    def _inject_all_cash(self):
        for ag in self.gui.agents.values():
            ag['cash'] = ag.get('cash', 0.0) + 5000.0
        self.gui.agent_inspector.update_agents(self.gui.agents, self.gui.stock_data)

    def _spawn_rl_agent(self):
        idx = len(self.gui.agents)
        aid = f"agent_{idx}"
        self.gui.agents[aid] = {
            'agent_type': 'rlaif',
            'cash': 10000.0,
            'initial_cash': 10000.0,
            'holdings': {},
            'total_value': 10000.0
        }
        self.gui.agent_inspector.update_agents(self.gui.agents, self.gui.stock_data)
        self.gui.news_log.add_entry("News", f"New RLAIF Agent {aid} joined the market")

    def _spawn_adv_agent(self):
        idx = len(self.gui.agents)
        aid = f"agent_{idx}"
        self.gui.agents[aid] = {
            'agent_type': 'adversarial',
            'cash': 10000.0,
            'initial_cash': 10000.0,
            'holdings': {},
            'total_value': 10000.0,
            'villain_points': 50
        }
        self.gui.agent_inspector.update_agents(self.gui.agents, self.gui.stock_data)
        self.gui.news_log.add_entry("Villain Move", f"Adversarial Agent {aid} entered the market")

    def _trigger_crash(self):
        for s in self.gui.stock_data:
            self.gui.stock_data[s] = max(1.0, self.gui.stock_data[s] * 0.85)
        self.gui.stock_ticker.update_prices(self.gui.stock_data)
        self.gui.news_log.add_entry("Villain Move", "FLASH CRASH: Prices dropped by -15% across all assets!")

    def _trigger_spike(self):
        for s in self.gui.stock_data:
            self.gui.stock_data[s] = self.gui.stock_data[s] * 1.15
        self.gui.stock_ticker.update_prices(self.gui.stock_data)
        self.gui.news_log.add_entry("News", "BULL RALLY: Market surged by +15%!")

    def _trigger_kaiser(self):
        self.gui.simulation_state['villain_points'] = self.gui.simulation_state.get('villain_points', 0) + 100
        self.gui.news_log.add_entry("Kaiser Election", "Kaiser Election announced! +100 Villain Points awarded.")

    def _broadcast_news(self):
        msg = self.news_input.text().strip()
        if msg:
            self.gui.news_log.add_entry("News", msg)
            self.news_input.clear()


class NewsLog(QWidget):
    """Right sidebar for news and events"""
    
    def __init__(self):
        super().__init__()
        self.entries: List[Dict[str, Any]] = []
        
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        self.label = QLabel("Real-Time News & Events Stream")
        self.label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.label.setStyleSheet("color: #7AA2F7;")
        self.layout.addWidget(self.label)
        
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.layout.addWidget(self.text_area)
    
    def add_entry(self, entry_type: str, message: str, timestamp: datetime = None):
        if timestamp is None:
            timestamp = datetime.now()
        
        self.entries.append({
            'type': entry_type,
            'message': message,
            'timestamp': timestamp
        })
        
        if len(self.entries) > 100:
            self.entries = self.entries[-100:]
        
        self.update_display()
    
    def update_display(self):
        text = ""
        for entry in reversed(self.entries[-25:]):
            timestamp = entry['timestamp'].strftime("%H:%M:%S")
            entry_type = entry['type']
            message = entry['message']
            
            if entry_type == "Kaiser Election":
                color = "#E0AF68"
            elif entry_type == "Villain Move":
                color = "#F7768E"
            elif entry_type == "News":
                color = "#7AA2F7"
            else:
                color = "#C0CAF5"
            
            text += f"<span style='color:#565F89;'>[{timestamp}]</span> <b style='color:{color};'>{entry_type}:</b> {message}<br>"
        
        self.text_area.setHtml(text)


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
        
        self.label = QLabel(f"{agent_id} (Acc: {self.accuracy:.2%})")
        self.label.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        self.label.setStyleSheet("color: #7AA2F7;")
        self.layout.addWidget(self.label)
        
        self.figure = Figure(figsize=(3, 2), dpi=80, facecolor='#1A1B26')
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        
        self.update_graph()
    
    def calculate_accuracy(self) -> float:
        if len(self.actual_prices) != len(self.predicted_prices) or len(self.actual_prices) == 0:
            return 0.0
        errors = [abs((a - p) / a) for a, p in zip(self.actual_prices, self.predicted_prices) if a != 0]
        if not errors:
            return 0.0
        return max(0.0, 1.0 - (sum(errors) / len(errors)))
    
    def update_graph(self):
        self.figure.clear()
        if not self.actual_prices or not self.predicted_prices:
            return
        
        ax = self.figure.add_subplot(111, facecolor='#1A1B26')
        ax.tick_params(colors='#A9B1D6', labelsize=6)
        for spine in ax.spines.values():
            spine.set_color('#2A2C3D')
            
        x = range(len(self.actual_prices))
        ax.plot(x, self.actual_prices, color='#7AA2F7', linewidth=1.2, label='Actual')
        ax.plot(x, self.predicted_prices, color='#F7768E', linestyle='--', linewidth=1.2, label='Predicted')
        ax.grid(True, color='#292A3E', linewidth=0.5, alpha=0.6)
        self.canvas.draw()


class AgentPredictionsPanel(QWidget):
    """Horizontal scrollable panel for agent predictions"""
    
    def __init__(self):
        super().__init__()
        self.agent_predictions: Dict[str, Dict[str, Any]] = {}
        
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        self.label = QLabel("Agent Price Predictions (Sorted by Forecast Accuracy)")
        self.label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.label.setStyleSheet("color: #7AA2F7;")
        self.layout.addWidget(self.label)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self.graphs_container = QWidget()
        self.graphs_layout = QHBoxLayout()
        self.graphs_container.setLayout(self.graphs_layout)
        self.graphs_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        
        self.scroll_area.setWidget(self.graphs_container)
        self.layout.addWidget(self.scroll_area)
    
    def update_predictions(self, actual_prices: List[float], agent_predictions: Dict[str, List[float]]):
        self.agent_predictions = {}
        for agent_id, predictions in agent_predictions.items():
            acc = self._calculate_accuracy(actual_prices, predictions)
            self.agent_predictions[agent_id] = {
                'predictions': predictions,
                'accuracy': acc
            }
        
        sorted_agents = sorted(self.agent_predictions.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        for i in reversed(range(self.graphs_layout.count())):
            item = self.graphs_layout.itemAt(i)
            if item and item.widget():
                item.widget().setParent(None)
        
        for agent_id, data in sorted_agents:
            graph = AgentPredictionGraph(agent_id, actual_prices, data['predictions'])
            self.graphs_layout.addWidget(graph)
    
    def _calculate_accuracy(self, actual: List[float], predicted: List[float]) -> float:
        if len(actual) != len(predicted) or len(actual) == 0:
            return 0.0
        errors = [abs((a - p) / a) for a, p in zip(actual, predicted) if a != 0]
        if not errors:
            return 0.0
        return max(0.0, 1.0 - (sum(errors) / len(errors)))


class VillainShop(QWidget):
    """Villain points shop for purchasing market disruptions"""
    
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
        
        self.label = QLabel("Villain Marketplace")
        self.label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.label.setStyleSheet("color: #F7768E;")
        self.layout.addWidget(self.label)
        
        self.points_label = QLabel(f"Villain Points: {self.villain_points}")
        self.points_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.points_label.setStyleSheet("color: #E0AF68;")
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
        
        self.purchase_button = QPushButton("Execute Disruption Purchase")
        self.purchase_button.setStyleSheet("background-color: #37222E; color: #F7768E;")
        self.purchase_button.clicked.connect(self.purchase_item)
        self.layout.addWidget(self.purchase_button)
        
        if self.shop_items:
            self.select_item(list(self.shop_items.keys())[0])
    
    def set_villain_points(self, points: int):
        self.villain_points = points
        self.points_label.setText(f"Villain Points: {self.villain_points}")
        self.update_cost_display()
    
    def calculate_dynamic_cost(self, item_name: str, duration: int = 5, timing: str = 'immediate') -> int:
        if item_name not in self.shop_items:
            return 0
        item = self.shop_items[item_name]
        base_cost = item['base_cost']
        probability = item['probability']
        probability_multiplier = 1.0 / (probability + 0.01) if probability > 0 else 1.0
        duration_multiplier = 1.0 + (duration / 10.0)
        timing_multiplier = 1.5 if timing == 'delayed' else 1.0
        return int(base_cost * probability_multiplier * duration_multiplier * timing_multiplier)
    
    def select_item(self, item_name: str):
        self.selected_item = item_name
        if item_name in self.shop_items:
            item = self.shop_items[item_name]
            self.duration_combo.clear()
            self.duration_combo.addItems([str(d) for d in item['duration_options']])
            self.duration_combo.currentTextChanged.connect(self.set_duration)
            
            self.timing_combo.clear()
            self.timing_combo.addItems(item['timing_options'])
            self.timing_combo.currentTextChanged.connect(self.set_timing)
            
            self.update_cost_display()
    
    def set_duration(self, duration_str: str):
        if duration_str:
            self.selected_duration = int(duration_str)
            self.update_cost_display()
    
    def set_timing(self, timing: str):
        if timing:
            self.selected_timing = timing
            self.update_cost_display()
    
    def update_cost_display(self):
        if self.selected_item:
            cost = self.calculate_dynamic_cost(self.selected_item, self.selected_duration, self.selected_timing)
            self.cost_label.setText(f"Cost: {cost} pts")
            self.purchase_button.setEnabled(self.villain_points >= cost)
    
    def purchase_item(self):
        if self.selected_item:
            cost = self.calculate_dynamic_cost(self.selected_item, self.selected_duration, self.selected_timing)
            if self.villain_points >= cost:
                self.villain_points -= cost
                self.points_label.setText(f"Villain Points: {self.villain_points}")
                self.update_cost_display()
                return True
        return False


class TradingGUI(QMainWindow):
    """Main Trading GUI Window featuring top horizontal slab chart, runtime controls, and live resource agent inspector"""
    
    def __init__(self, simulation_state: Dict[str, Any]):
        super().__init__()
        self.simulation_state = simulation_state
        self.stock_data = simulation_state.get('current_stock_data', {})
        self.price_history = simulation_state.get('price_history', {})
        self.selected_stock = list(self.stock_data.keys())[0] if self.stock_data else "AAPL"
        self.agents = simulation_state.get('agents', {})
        self.paused = False
        self.update_counter = 0
        self.volatility_multiplier = 1.0
        
        self.black_box_active = False
        self.black_box_start_idx = 0
        self.black_box_duration = simulation_state.get('black_box_duration', 5)
        self.black_box_frequency = simulation_state.get('black_box_frequency', 20)
        self.rng_bbox = simulation_state.get('rng_bbox', None)
        
        self.setWindowTitle("Synthetic RL Market Platform - Advanced Simulation Console")
        self.setGeometry(50, 50, 1500, 950)
        
        # Apply dark theme
        self.setStyleSheet(QSS_THEME)
        
        self.setup_ui()
        self.setup_timer()
    
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)
        
        # 1. Top Stock Ticker Bar
        self.stock_ticker = StockTicker(self.stock_data)
        main_layout.addWidget(self.stock_ticker)
        
        # Vertical Splitter: Top Horizontal Price Chart Slab, Bottom Tabbed Control Center
        vertical_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # TOP SLAB: Price Graph & Stock Toolbar (Full Horizontal Width)
        top_slab_widget = QWidget()
        top_slab_layout = QVBoxLayout(top_slab_widget)
        top_slab_layout.setContentsMargins(4, 4, 4, 4)
        top_slab_layout.setSpacing(4)
        
        toolbar_layout = QHBoxLayout()
        stock_sel_lbl = QLabel("Active Stock View:")
        stock_sel_lbl.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        toolbar_layout.addWidget(stock_sel_lbl)
        
        self.stock_combo = QComboBox()
        self.stock_combo.addItems(list(self.stock_data.keys()))
        self.stock_combo.currentTextChanged.connect(self.change_stock)
        toolbar_layout.addWidget(self.stock_combo)
        
        toolbar_layout.addStretch()
        
        self.btn_pause_top = QPushButton("Pause / Resume")
        self.btn_pause_top.clicked.connect(self.toggle_pause)
        toolbar_layout.addWidget(self.btn_pause_top)
        
        top_slab_layout.addLayout(toolbar_layout)
        
        # Horizontal Slab Price Graph
        self.price_graph = PriceGraph(self.price_history)
        top_slab_layout.addWidget(self.price_graph)
        
        vertical_splitter.addWidget(top_slab_widget)
        
        # BOTTOM SECTION: Main Control & Live Inspection Center Tabs
        bottom_tabs = QTabWidget()
        
        # Tab 1: Agent Inspector (Live Resource Table & Fine Tuning)
        self.agent_inspector = AgentInspector(self.agents, self.stock_data)
        bottom_tabs.addTab(self.agent_inspector, "Agent Live Inspector")
        
        # Tab 2: Runtime Parameter Tuning
        self.runtime_tuning = RuntimeTuningPanel(self)
        bottom_tabs.addTab(self.runtime_tuning, "Runtime Parameter Tuning")
        
        # Tab 3: News & Events Stream
        self.news_log = NewsLog()
        bottom_tabs.addTab(self.news_log, "News & Events Log")
        
        # Tab 4: Agent Predictions Panel
        self.agent_predictions_panel = AgentPredictionsPanel()
        bottom_tabs.addTab(self.agent_predictions_panel, "Agent Predictions")
        
        # Tab 5: Villain Marketplace
        self.villain_shop = VillainShop()
        bottom_tabs.addTab(self.villain_shop, "Villain Shop")
        
        vertical_splitter.addWidget(bottom_tabs)
        
        # Default split sizes: Top Slab (380px), Bottom Tabs (550px)
        vertical_splitter.setSizes([380, 550])
        main_layout.addWidget(vertical_splitter)
    
    def setup_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation_data)
        self.timer.start(100)
    
    def change_stock(self, stock_name: str):
        if stock_name:
            self.selected_stock = stock_name
            self.price_graph.set_selected_stock(stock_name)
    
    def toggle_pause(self):
        self.paused = not self.paused
        txt = "Resume Simulation" if self.paused else "Pause / Resume"
        self.btn_pause_top.setText(txt)
        self.runtime_tuning.btn_pause.setText(txt)
    
    def step_manual_tick(self):
        was_paused = self.paused
        self.paused = False
        self.update_simulation_data()
        self.paused = was_paused

    def trigger_black_box_now(self):
        self.black_box_active = True
        self.black_box_start_idx = self.update_counter
        if self.selected_stock in self.price_history:
            curr_idx = len(self.price_history[self.selected_stock])
            self.price_graph.add_black_box_period(curr_idx, curr_idx + self.black_box_duration)
        self.news_log.add_entry("Villain Move", "BLACK BOX ANOMALY ACTIVATED NOW!")

    def end_black_box_now(self):
        self.black_box_active = False
        self.price_graph.end_black_box_period()
        self.news_log.add_entry("News", "Black box anomaly period ended.")

    def update_simulation_data(self):
        if self.paused:
            return
            
        self.update_counter += 1
        
        # Black Box Logic
        if self.rng_bbox and not self.black_box_active:
            if self.update_counter % self.black_box_frequency == 0:
                self.trigger_black_box_now()
        
        if self.black_box_active:
            if self.update_counter - self.black_box_start_idx >= self.black_box_duration:
                self.end_black_box_now()
                if self.rng_bbox:
                    mn_d, mx_d, mn_f, mx_f = self.rng_bbox
                    self.black_box_duration = random.randint(mn_d, mx_d)
                    self.black_box_frequency = random.randint(mn_f, mx_f)
                    self.runtime_tuning.dur_spin.setValue(self.black_box_duration)
                    self.runtime_tuning.freq_spin.setValue(self.black_box_frequency)
        
        # Dynamic Price Simulation
        if self.update_counter % 2 == 0:
            for symbol in self.stock_data:
                change = random.uniform(-1.5, 1.5) * self.volatility_multiplier
                self.stock_data[symbol] = max(1.0, self.stock_data[symbol] + change)
            self.stock_ticker.update_prices(self.stock_data)
        
        # Record Price History
        if self.update_counter % 5 == 0:
            for symbol in self.stock_data:
                if symbol not in self.price_history:
                    self.price_history[symbol] = []
                self.price_history[symbol].append(self.stock_data[symbol])
                if len(self.price_history[symbol]) > 150:
                    self.price_history[symbol] = self.price_history[symbol][-150:]
            self.price_graph.update_price_history(self.price_history)
        
        # DYNAMIC LIVE AGENT RESOURCE UPDATE
        self.agent_inspector.update_agents(self.agents, self.stock_data)
        
        # Update Random News
        if self.update_counter % 25 == 0:
            types = ["News", "Villain Move", "Kaiser Election"]
            msgs = [
                "High-frequency algorithmic trading detected",
                "Order book depth imbalanced in tech sector",
                "Kaiser delegates negotiating tax policy",
                "Adversarial agent exploiting liquidity gap",
                "Market volatility index adjusting upward"
            ]
            self.news_log.add_entry(random.choice(types), random.choice(msgs))
        
        # Update Predictions
        if self.update_counter % 20 == 0 and self.selected_stock in self.price_history:
            actual = self.price_history[self.selected_stock][-20:]
            preds = {}
            for aid in self.agents.keys():
                noise = random.uniform(-4.0, 4.0)
                preds[aid] = [max(1.0, p + noise) for p in actual]
            self.agent_predictions_panel.update_predictions(actual, preds)
            
        # Sync villain points
        v_pts = self.simulation_state.get('villain_points', 0)
        self.villain_shop.set_villain_points(v_pts)


def launch_gui(simulation_state: Dict[str, Any]):
    """Launch the Trading GUI Application"""
    app = QApplication(sys.argv)
    window = TradingGUI(simulation_state)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    mock_state = {
        'current_stock_data': {
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
            'agent_0': {'agent_type': 'rlaif', 'cash': 10000.0, 'initial_cash': 10000.0, 'holdings': {'AAPL': 10, 'MSFT': 5}, 'total_value': 13000.0},
            'agent_1': {'agent_type': 'adversarial', 'cash': 8500.0, 'initial_cash': 10000.0, 'holdings': {'TSLA': 2}, 'total_value': 10100.0, 'villain_points': 80},
        },
        'news_events': [
            {'type': 'News', 'message': 'Market opens with high liquidity'},
        ],
        'villain_points': 150,
        'black_box_duration': 5,
        'black_box_frequency': 20,
    }
    
    launch_gui(mock_state)
