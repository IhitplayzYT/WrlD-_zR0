from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Static, DataTable, Tabs, Tab, Log, Button, Input, Label
from textual.scroll_view import ScrollView
from textual.reactive import reactive
from textual import events
from textual.screen import Screen
from textual.binding import Binding
from typing import Dict, List, Optional, Any
import asyncio
from datetime import datetime


class StockTicker(Static):
    """Top ticker showing circulating stock prices"""
    
    def __init__(self, stock_data: Dict[str, float]):
        super().__init__()
        self.stock_data = stock_data
        self.previous_prices = stock_data.copy()
    
    def update_prices(self, new_prices: Dict[str, float]):
        """Update stock prices and calculate changes"""
        self.previous_prices = self.stock_data.copy()
        self.stock_data = new_prices
        self.update()
    
    def render(self):
        """Render the ticker with color-coded prices"""
        ticker_text = []
        for symbol, price in self.stock_data.items():
            prev_price = self.previous_prices.get(symbol, price)
            change = price - prev_price
            change_pct = (change / prev_price * 100) if prev_price > 0 else 0
            
            if change > 0:
                color = "green"
                arrow = "▲"
            elif change < 0:
                color = "red"
                arrow = "▼"
            else:
                color = "white"
                arrow = "→"
            
            ticker_text.append(f"[{color}]{symbol}: ${price:.2f} {arrow} {abs(change_pct):.2f}%[/{color}]")
        
        return " | ".join(ticker_text)


class PriceGraph(ScrollView):
    """Price graph with zoom functionality"""
    
    zoom_level = reactive(1.0)
    selected_stock = reactive("AAPL")
    
    def __init__(self, price_history: Dict[str, List[float]]):
        super().__init__()
        self.price_history = price_history
        self.zoom_level = 1.0
    
    def on_mouse_scroll(self, event: events.MouseScroll) -> None:
        """Handle mouse scroll for zoom"""
        if event.delta_y > 0:
            self.zoom_level = max(0.1, self.zoom_level - 0.1)
        else:
            self.zoom_level = min(10.0, self.zoom_level + 0.1)
        self.update()
    
    def render(self) -> str:
        """Render the price graph based on zoom level"""
        if self.selected_stock not in self.price_history:
            return "No data available"
        
        prices = self.price_history[self.selected_stock]
        if not prices:
            return "No data available"
        
        # Determine resolution based on zoom level
        if self.zoom_level > 5.0:
            # High zoom: show every price change with characters
            return self._render_detailed_graph(prices)
        else:
            # Low zoom: show daily closing prices
            return self._render_summary_graph(prices)
    
    def _render_detailed_graph(self, prices: List[float]) -> str:
        """Render detailed graph with character-based representation"""
        if len(prices) < 2:
            return "Insufficient data"
        
        min_price = min(prices)
        max_price = max(prices)
        price_range = max_price - min_price if max_price > min_price else 1.0
        
        lines = []
        for i in range(len(prices) - 1):
            current = prices[i]
            next_price = prices[i + 1]
            
            # Calculate position and magnitude
            current_pos = int((current - min_price) / price_range * 10)
            next_pos = int((next_price - min_price) / price_range * 10)
            
            # Calculate change magnitude for character selection
            change = next_price - current
            change_pct = abs(change / current * 100) if current > 0 else 0
            
            # Determine connector based on magnitude
            if change_pct > 0.5:
                if change > 0:
                    connector = "▲"
                else:
                    connector = "▼"
            elif change_pct > 0.1:
                if change > 0:
                    connector = "/\\"
                else:
                    connector = "\\/"
            elif change_pct > 0.01:
                if change > 0:
                    connector = "/-"
                else:
                    connector = "\\-"
            else:
                if change > 0:
                    connector = "."
                elif change < 0:
                    connector = ","
                else:
                    connector = "_"
            
            # Color based on direction and magnitude
            if change > 0:
                if change_pct > 0.5:
                    color = "bold green"
                elif change_pct > 0.1:
                    color = "green"
                else:
                    color = "dim green"
            elif change < 0:
                if change_pct > 0.5:
                    color = "bold red"
                elif change_pct > 0.1:
                    color = "red"
                else:
                    color = "dim red"
            else:
                color = "white"
            
            line = f"[{color}]{connector} ${current:.2f} -> ${next_price:.2f} ({change_pct:+.2f}%)[/{color}]"
            lines.append(line)
        
        return "\n".join(lines[-30:])  # Show last 30 changes
    
    def _render_summary_graph(self, prices: List[float]) -> str:
        """Render summary graph with daily closing prices"""
        if len(prices) < 2:
            return "Insufficient data"
        
        min_price = min(prices)
        max_price = max(prices)
        price_range = max_price - min_price if max_price > min_price else 1.0
        
        # Create ASCII graph
        height = 10
        width = min(50, len(prices))
        
        graph = []
        for row in range(height):
            line = ""
            for col in range(width):
                price_idx = int(col / width * len(prices))
                price = prices[price_idx]
                
                price_pos = int((price - min_price) / price_range * (height - 1))
                price_pos = height - 1 - price_pos  # Invert for top-down rendering
                
                if price_pos == row:
                    line += "*"
                else:
                    line += " "
            graph.append(line)
        
        return "\n".join(graph)


class NewsLog(ScrollView):
    """Right sidebar for news and events"""
    
    def __init__(self):
        super().__init__()
        self.entries: List[Dict[str, Any]] = []
        self.content = ""
    
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
        
        # Update content
        self.content = self._render_content()
    
    def _render_content(self) -> str:
        """Render the news log content"""
        if not self.entries:
            return "No news or events"
        
        lines = []
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
            
            lines.append(f"[{color}][{timestamp}] {entry_type}: {message}[/{color}]")
        
        return "\n".join(lines)
    
    def render(self) -> str:
        """Render the news log"""
        return self.content if self.content else "No news or events"


class AgentInspector(Static):
    """Agent inspection and modification panel"""
    
    def __init__(self, agents: Dict[str, Any]):
        super().__init__()
        self.agents = agents
        self.selected_agent = None
    
    def select_agent(self, agent_id: str):
        """Select an agent for inspection"""
        self.selected_agent = agent_id
        self.update()
    
    def modify_agent(self, agent_id: str, modifications: Dict[str, Any]):
        """Modify agent properties"""
        if agent_id in self.agents:
            for key, value in modifications.items():
                setattr(self.agents[agent_id], key, value)
            self.update()
    
    def render(self) -> str:
        """Render agent inspection panel"""
        if self.selected_agent is None:
            return "Select an agent to inspect"
        
        if self.selected_agent not in self.agents:
            return f"Agent {self.selected_agent} not found"
        
        agent = self.agents[self.selected_agent]
        
        # Display agent properties
        lines = [
            f"Agent ID: {self.selected_agent}",
            f"Type: {getattr(agent, 'agent_type', 'Unknown')}",
            f"Cash: ${getattr(agent, 'cash', 0):.2f}",
            f"Holdings: {getattr(agent, 'holdings', 0)}",
            f"Total Value: ${getattr(agent, 'total_value', 0):.2f}",
        ]
        
        # Add villain points if applicable
        if hasattr(agent, 'villain_points'):
            lines.append(f"Villain Points: {agent.villain_points}")
        
        # Add traits if applicable
        if hasattr(agent, 'traits'):
            lines.append("\nTraits:")
            for trait_name, trait_value in agent.traits.items():
                lines.append(f"  {trait_name}: {trait_value:.2f}")
        
        return "\n".join(lines)


class VillainShop(Static):
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
    
    def set_villain_points(self, points: int):
        """Set available villain points"""
        self.villain_points = points
        self.update()
    
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
    
    def purchase_item(self, item_name: str, duration: int = 5, timing: str = 'immediate') -> bool:
        """Purchase an item from the shop"""
        cost = self.calculate_dynamic_cost(item_name, duration, timing)
        
        if self.villain_points < cost:
            return False
        
        self.villain_points -= cost
        self.update()
        return True
    
    def select_item(self, item_name: str):
        """Select an item for purchase"""
        self.selected_item = item_name
        self.update()
    
    def set_duration(self, duration: int):
        """Set duration for selected item"""
        self.selected_duration = duration
        self.update()
    
    def set_timing(self, timing: str):
        """Set timing for selected item"""
        self.selected_timing = timing
        self.update()
    
    def render(self) -> str:
        """Render the villain shop with dynamic pricing"""
        lines = [
            f"Villain Points: {self.villain_points}",
            "\nShop Items (Dynamic Pricing):",
        ]
        
        for item_name, item_data in self.shop_items.items():
            base_cost = item_data['base_cost']
            impact = item_data['impact']
            probability = item_data['probability']
            
            # Calculate dynamic cost for default options
            dynamic_cost = self.calculate_dynamic_cost(item_name, self.selected_duration, self.selected_timing)
            
            can_afford = self.villain_points >= dynamic_cost
            status = "green" if can_afford else "red"
            selected = " [bold]<-[/bold]" if self.selected_item == item_name else ""
            
            lines.append(f"[{status}]{item_name}: {dynamic_cost} pts (Base: {base_cost}, Impact: {impact}, Prob: {probability})[/{status}]{selected}")
        
        if self.selected_item:
            item = self.shop_items[self.selected_item]
            lines.append(f"\nSelected: {self.selected_item}")
            lines.append(f"Duration: {self.selected_duration} (Options: {item['duration_options']})")
            lines.append(f"Timing: {self.selected_timing} (Options: {item['timing_options']})")
            lines.append(f"Total Cost: {self.calculate_dynamic_cost(self.selected_item, self.selected_duration, self.selected_timing)} pts")
        
        return "\n".join(lines)


class ModelManagement(Static):
    """Model management tab for injecting data and currency"""
    
    def __init__(self):
        super().__init__()
        self.models: Dict[str, Any] = {}
        self.selected_model = None
    
    def add_model(self, model_id: str, model: Any):
        """Add a model to management"""
        self.models[model_id] = model
        self.update()
    
    def inject_data(self, model_id: str, data: Dict[str, Any]):
        """Inject data into a model"""
        if model_id in self.models:
            self.models[model_id].update(data)
            self.update()
    
    def render(self) -> str:
        """Render model management panel"""
        if not self.models:
            return "No models available"
        
        lines = ["Available Models:"]
        for model_id in self.models.keys():
            lines.append(f"  - {model_id}")
        
        if self.selected_model:
            lines.append(f"\nSelected: {self.selected_model}")
            if self.selected_model in self.models:
                model_data = self.models[self.selected_model]
                lines.append(f"Data: {model_data}")
        
        return "\n".join(lines)


class TradingTUI(App):
    """Main TUI application for RLAIF trading simulation"""
    
    CSS = """
    Screen {
        background: $primary;
    }
    
    #main_container {
        width: 70%;
        height: 100%;
        background: transparent;
    }
    
    #sidebar {
        width: 30%;
        height: 100%;
        dock: right;
        background: transparent;
    }
    
    StockTicker {
        height: 3;
        dock: top;
        background: $surface;
        text-align: center;
        padding: 0 1;
    }
    
    PriceGraph {
        height: 40%;
        background: transparent;
        border: solid green;
        padding: 1;
    }
    
    NewsLog {
        height: 60%;
        background: transparent;
        border: solid yellow;
        padding: 1;
        overflow-y: auto;
    }
    
    AgentInspector {
        height: 30%;
        background: transparent;
        border: solid cyan;
        padding: 1;
    }
    
    VillainShop {
        height: 30%;
        background: transparent;
        border: solid red;
        padding: 1;
    }
    
    Static {
        text-style: bold;
    }
    
    .green {
        color: green;
    }
    
    .red {
        color: red;
    }
    
    .yellow {
        color: yellow;
    }
    
    .cyan {
        color: cyan;
    }
    
    .white {
        color: white;
    }
    """
    
    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("space", "toggle_pause", "Pause/Resume", show=True),
        Binding("1", "switch_tab_1", "Market View", show=True),
        Binding("2", "switch_tab_2", "Model Management", show=True),
        Binding("up", "scroll_up", "Scroll Up", show=False),
        Binding("down", "scroll_down", "Scroll Down", show=False),
        Binding("tab", "next_stock", "Next Stock", show=False),
        Binding("left", "prev_item", "Previous Shop Item", show=False),
        Binding("right", "next_item", "Next Shop Item", show=False),
        Binding("enter", "purchase_item", "Purchase Item", show=False),
        Binding("+", "increase_duration", "Increase Duration", show=False),
        Binding("-", "decrease_duration", "Decrease Duration", show=False),
        Binding("t", "toggle_timing", "Toggle Timing", show=False),
    ]
    
    paused = reactive(False)
    current_tab = reactive(1)
    
    def __init__(self, simulation_state: Dict[str, Any]):
        super().__init__()
        self.simulation_state = simulation_state
        self.stock_data = simulation_state.get('stock_data', {})
        self.price_history = simulation_state.get('price_history', {})
        self.agents = simulation_state.get('agents', {})
        self.news_events = simulation_state.get('news_events', [])
        self.simulation_task = None
        self.update_counter = 0
    
    def compose(self) -> ComposeResult:
        """Compose the TUI layout"""
        yield Header()
        
        with Container(id="main_container"):
            yield StockTicker(self.stock_data)
            yield PriceGraph(self.price_history)
        
        with Container(id="sidebar"):
            yield NewsLog()
            yield AgentInspector(self.agents)
            yield VillainShop()
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Initialize the TUI"""
        self.update_simulation_data()
        # Start simulation loop as a background task
        self.simulation_task = self.set_interval(0.1, self.update_simulation_data)
    
    def action_toggle_pause(self) -> None:
        """Toggle pause state"""
        self.paused = not self.paused
        if self.paused:
            if self.simulation_task:
                self.simulation_task.stop()
        else:
            if self.simulation_task:
                self.simulation_task.start()
    
    def action_switch_tab_1(self) -> None:
        """Switch to market view tab"""
        self.current_tab = 1
    
    def action_switch_tab_2(self) -> None:
        """Switch to model management tab"""
        self.current_tab = 2
    
    def action_scroll_up(self) -> None:
        """Scroll up in focused widget"""
        focused = self.focused
        if focused and hasattr(focused, 'scroll_up'):
            focused.scroll_up()
    
    def action_scroll_down(self) -> None:
        """Scroll down in focused widget"""
        focused = self.focused
        if focused and hasattr(focused, 'scroll_down'):
            focused.scroll_down()
    
    def action_next_stock(self) -> None:
        """Switch to next stock in price graph"""
        stocks = list(self.stock_data.keys())
        if stocks:
            current_idx = stocks.index(self.query_one(PriceGraph).selected_stock) if self.query_one(PriceGraph).selected_stock in stocks else 0
            next_idx = (current_idx + 1) % len(stocks)
            self.query_one(PriceGraph).selected_stock = stocks[next_idx]
    
    def action_prev_item(self) -> None:
        """Select previous item in villain shop"""
        shop = self.query_one(VillainShop)
        items = list(shop.shop_items.keys())
        if items and shop.selected_item:
            current_idx = items.index(shop.selected_item)
            prev_idx = (current_idx - 1) % len(items)
            shop.select_item(items[prev_idx])
    
    def action_next_item(self) -> None:
        """Select next item in villain shop"""
        shop = self.query_one(VillainShop)
        items = list(shop.shop_items.keys())
        if items:
            current_idx = items.index(shop.selected_item) if shop.selected_item in items else 0
            next_idx = (current_idx + 1) % len(items)
            shop.select_item(items[next_idx])
    
    def action_purchase_item(self) -> None:
        """Purchase selected item from villain shop"""
        shop = self.query_one(VillainShop)
        if shop.selected_item:
            success = shop.purchase_item(shop.selected_item, shop.selected_duration, shop.selected_timing)
            if success:
                # Update simulation state
                self.simulation_state['villain_points'] = shop.villain_points
    
    def action_increase_duration(self) -> None:
        """Increase duration for selected shop item"""
        shop = self.query_one(VillainShop)
        if shop.selected_item:
            item = shop.shop_items[shop.selected_item]
            current_idx = item['duration_options'].index(shop.selected_duration) if shop.selected_duration in item['duration_options'] else 0
            next_idx = min(current_idx + 1, len(item['duration_options']) - 1)
            shop.set_duration(item['duration_options'][next_idx])
    
    def action_decrease_duration(self) -> None:
        """Decrease duration for selected shop item"""
        shop = self.query_one(VillainShop)
        if shop.selected_item:
            item = shop.shop_items[shop.selected_item]
            current_idx = item['duration_options'].index(shop.selected_duration) if shop.selected_duration in item['duration_options'] else 0
            prev_idx = max(current_idx - 1, 0)
            shop.set_duration(item['duration_options'][prev_idx])
    
    def action_toggle_timing(self) -> None:
        """Toggle timing for selected shop item"""
        shop = self.query_one(VillainShop)
        if shop.selected_item:
            item = shop.shop_items[shop.selected_item]
            current_idx = item['timing_options'].index(shop.selected_timing) if shop.selected_timing in item['timing_options'] else 0
            next_idx = (current_idx + 1) % len(item['timing_options'])
            shop.set_timing(item['timing_options'][next_idx])
    
    def update_simulation_data(self):
        """Update simulation data from state"""
        if self.paused:
            return
            
        self.update_counter += 1
        
        # Update stock ticker with simulated price changes
        if self.update_counter % 5 == 0:  # Update prices every 5 thecks
            for symbol in self.stock_data:
                import random
                change = random.uniform(-2, 2)
                self.stock_data[symbol] = max(1.0, self.stock_data[symbol] + change)
            self.query_one(StockTicker).update_prices(self.stock_data)
        
        # Update price history
        if self.update_counter % 10 == 0:  # Update history every 10 ticks
            for symbol in self.stock_data:
                if symbol not in self.price_history:
                    self.price_history[symbol] = []
                self.price_history[symbol].append(self.stock_data[symbol])
                if len(self.price_history[symbol]) > 100:
                    self.price_history[symbol] = self.price_history[symbol][-100:]
            self.query_one(PriceGraph).price_history = self.price_history
        
        # Update news log (only add new events occasionally)
        if self.update_counter % 20 == 0:  # Add news every 20 ticks
            import random
            news_types = ["News", "Villain Move", "Kaiser Election"]
            news_messages = [
                "Market volatility increases",
                "Price manipulation detected",
                "New Kaiser election announced",
                "Sector rotation observed",
                "Liquidity concerns in tech stocks"
            ]
            news_log = self.query_one(NewsLog)
            news_log.add_entry(
                random.choice(news_types),
                random.choice(news_messages)
            )
        
        # Update villain shop
        if 'villain_points' in self.simulation_state:
            self.query_one(VillainShop).set_villain_points(self.simulation_state['villain_points'])


def launch_tui(simulation_state: Dict[str, Any]):
    """Launch the TUI application"""
    app = TradingTUI(simulation_state)
    app.run()


if __name__ == "__main__":
    # Test TUI with mock data
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
    
    launch_tui(mock_state)
