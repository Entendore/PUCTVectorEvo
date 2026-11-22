import sys
import math
import random
import copy
from dataclasses import dataclass, field
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QSlider, QProgressBar, QSplitter)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QPointF, QRectF
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPolygonF

# ==========================================
# 1. The Environment (Evolution Sim)
# ==========================================
GRID_SIZE = 12
NUM_FOODS = 5
NUM_OBSTACLES = 20
NUM_PREDATORS = 2
ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)] # Right, Left, Down, Up

@dataclass
class SimState:
    creature_pos: tuple
    food_positions: list
    obstacles: list
    predators: list
    energy: int
    steps_survived: int
    score: int

class EvolutionEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        obstacles = []
        while len(obstacles) < NUM_OBSTACLES:
            obs = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
            if obs not in obstacles and obs != (0, 0):
                obstacles.append(obs)
        
        foods = []
        while len(foods) < NUM_FOODS:
            f = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
            if f not in obstacles and f not in foods and f != (0, 0):
                foods.append(f)

        predators = []
        while len(predators) < NUM_PREDATORS:
            p = (random.randint(GRID_SIZE//2, GRID_SIZE-1), random.randint(GRID_SIZE//2, GRID_SIZE-1))
            if p not in obstacles and p not in foods and p != (0, 0):
                predators.append(p)
                
        self.state = SimState(
            creature_pos=(0, 0),
            food_positions=foods,
            obstacles=obstacles,
            predators=predators,
            energy=40,
            steps_survived=0,
            score=0
        )
        return self.get_state()

    def get_state(self):
        return copy.deepcopy(self.state)

    def move_predators(self):
        cx, cy = self.state.creature_pos
        new_preds = []
        for px, py in self.state.predators:
            possible_moves = [(px+dx, py+dy) for dx, dy in ACTIONS]
            valid_moves = [
                m for m in possible_moves 
                if 0 <= m[0] < GRID_SIZE and 0 <= m[1] < GRID_SIZE and m not in self.state.obstacles
            ]
            if valid_moves:
                if random.random() < 0.6:
                    best_move = min(valid_moves, key=lambda m: abs(m[0]-cx) + abs(m[1]-cy))
                    new_preds.append(best_move)
                else:
                    new_preds.append(random.choice(valid_moves))
            else:
                new_preds.append((px, py))
        self.state.predators = new_preds

    def step(self, action_idx):
        dx, dy = ACTIONS[action_idx]
        cx, cy = self.state.creature_pos
        nx, ny = cx + dx, cy + dy
        
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and (nx, ny) not in self.state.obstacles:
            self.state.creature_pos = (nx, ny)
            
        self.state.energy -= 1
        self.state.steps_survived += 1

        reward = -0.05
        done = False

        if self.state.creature_pos in self.state.food_positions:
            self.state.food_positions.remove(self.state.creature_pos)
            reward = 10.0
            self.state.energy += 25
            self.state.score += 1
            while True:
                nf = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
                if nf not in self.state.obstacles and nf not in self.state.food_positions and nf != self.state.creature_pos:
                    self.state.food_positions.append(nf)
                    break

        if self.state.creature_pos in self.state.predators:
            reward = -10.0
            done = True

        if self.state.energy <= 0:
            reward = -5.0
            done = True

        return self.get_state(), reward, done

# ==========================================
# 2. DeepMind-style PUCT AI
# ==========================================
class MockNeuralNetwork:
    def predict(self, state: SimState):
        cx, cy = state.creature_pos
        min_food_dist = min((abs(fx-cx) + abs(fy-cy) for fx, fy in state.food_positions), default=GRID_SIZE*2)
        min_pred_dist = min((abs(px-cx) + abs(py-cy) for px, py in state.predators), default=GRID_SIZE*2)
        
        value = (state.energy / 50.0) - (min_food_dist / (GRID_SIZE)) 
        if min_pred_dist <= 2: value -= 0.8
        value = max(-1.0, min(1.0, value))

        policy = [0.1, 0.1, 0.1, 0.1]
        if state.food_positions:
            closest_food = min(state.food_positions, key=lambda f: abs(f[0]-cx) + abs(f[1]-cy))
            fx, fy = closest_food
            if fx > cx: policy[0] += 0.5
            if fx < cx: policy[1] += 0.5
            if fy > cy: policy[2] += 0.5
            if fy < cy: policy[3] += 0.5

        if state.predators:
            closest_pred = min(state.predators, key=lambda p: abs(p[0]-cx) + abs(p[1]-cy))
            px, py = closest_pred
            if px > cx: policy[1] += 1.0
            if px < cx: policy[0] += 1.0
            if py > cy: policy[3] += 1.0
            if py < cy: policy[2] += 1.0
            
        for i, (dx, dy) in enumerate(ACTIONS):
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE) or (nx, ny) in state.obstacles:
                policy[i] = 0.01
                
        total = sum(policy)
        policy = [p/total for p in policy]
        return policy, value

class PUCTNode:
    def __init__(self, state, parent=None, action=None, prior=0.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.prior = prior
        self.children = []
        self.visit_count = 0
        self.total_value = 0.0

    def q_value(self):
        return self.total_value / self.visit_count if self.visit_count > 0 else 0

    def ucb_score(self, c_puct=1.5):
        if self.visit_count == 0: return float('inf')
        u = c_puct * self.prior * (math.sqrt(self.parent.visit_count) / (1 + self.visit_count))
        return self.q_value() + u

    def best_child(self, c_puct=1.5):
        return max(self.children, key=lambda c: c.ucb_score(c_puct))

class PUCTAgent:
    def __init__(self, env, network):
        self.env = env
        self.network = network

    def add_dirichlet_noise(self, node):
        noise = [random.gammavariate(0.3, 1) for _ in range(4)]
        total = sum(noise)
        noise = [n/total for n in noise]
        for child, n in zip(node.children, noise):
            child.prior = (1 - 0.25) * child.prior + 0.25 * n

    def search(self, state, num_simulations=50):
        root = PUCTNode(state)
        policy, _ = self.network.predict(root.state)
        for i in range(4):
            sim_env = EvolutionEnv()
            sim_env.state = copy.deepcopy(root.state)
            sim_env.step(i)
            root.children.append(PUCTNode(sim_env.get_state(), parent=root, action=i, prior=policy[i]))
        
        self.add_dirichlet_noise(root)
        root.visit_count = 1

        for _ in range(num_simulations):
            node = root
            sim_env = EvolutionEnv()
            sim_env.state = copy.deepcopy(root.state)
            
            while node.children:
                node = node.best_child()
                sim_env.state = copy.deepcopy(node.state)
            
            if sim_env.state.energy > 0 and sim_env.state.creature_pos not in sim_env.state.predators:
                p, value = self.network.predict(sim_env.state)
                for i in range(4):
                    temp_env = EvolutionEnv()
                    temp_env.state = copy.deepcopy(sim_env.state)
                    next_state, _, _ = temp_env.step(i)
                    node.children.append(PUCTNode(next_state, parent=node, action=i, prior=p[i]))
            
            while node is not None:
                node.visit_count += 1
                node.total_value += value
                node = node.parent

        action_visits = {child.action: child.visit_count for child in root.children}
        best_action = max(action_visits, key=action_visits.get)
        total_visits = sum(action_visits.values())
        action_probs = {a: v/total_visits for a, v in action_visits.items()}

        return best_action, action_probs, root

# ==========================================
# 3. PySide6 GUI and Threading
# ==========================================
class AIThread(QThread):
    action_ready = Signal(int, dict, object)

    def __init__(self, env, agent):
        super().__init__()
        self.env = env
        self.agent = agent
        self.running = True
        self.paused = False
        self.simulations = 50
        self.delay_ms = 100

    def run(self):
        while self.running:
            if self.paused:
                self.msleep(50)
                continue
            current_state = self.env.get_state()
            if current_state.energy <= 0 or current_state.creature_pos in current_state.predators:
                break
            best_action, probs, root_node = self.agent.search(current_state, self.simulations)
            self.action_ready.emit(best_action, probs, root_node)
            self.msleep(self.delay_ms)

class SimCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.state = None
        self.trail = []
        self.root_node = None
        self.chosen_action = None
        self.particles = []
        self.setFixedSize(500, 500)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_particles)
        self.timer.start(50)

    def spawn_particles(self, pos, color, count=15):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 6)
            self.particles.append({
                'x': pos[0], 'y': pos[1],
                'dx': math.cos(angle) * speed, 'dy': math.sin(angle) * speed,
                'life': 1.0, 'color': color
            })

    def update_particles(self):
        for p in self.particles:
            p['x'] += p['dx']
            p['y'] += p['dy']
            p['life'] -= 0.05
        self.particles = [p for p in self.particles if p['life'] > 0]
        self.update()

    def update_state(self, state, root_node=None, action=None):
        if self.state and self.state.creature_pos != state.creature_pos:
            self.trail.append(self.state.creature_pos)
            if len(self.trail) > 10: self.trail.pop(0)
        self.state = state
        self.root_node = root_node
        self.chosen_action = action
        
        # Trigger VFX
        if state.creature_pos in state.predators:
            self.spawn_particles(state.creature_pos, QColor(255, 50, 50), count=30)
        elif state.food_positions and self.state and state.creature_pos not in self.state.food_positions:
             # check if it just ate
             pass
        self.update()

    def paintEvent(self, event):
        if not self.state: return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        cell_size = self.width() / GRID_SIZE
        
        painter.fillRect(self.rect(), QColor(15, 15, 15))
        painter.setPen(QPen(QColor(35, 35, 35), 1))
        for i in range(GRID_SIZE + 1):
            painter.drawLine(i * cell_size, 0, i * cell_size, self.height())
            painter.drawLine(0, i * cell_size, self.width(), i * cell_size)
            
        # Value Heatmap (Thermal Overlay)
        if self.root_node and self.root_node.children:
            for child in self.root_node.children:
                val = child.q_value() # -1 to 1
                color_val = int((val + 1) / 2 * 255)
                color = QColor(color_val, 50, 255 - color_val, 100)
                painter.setBrush(color)
                painter.setPen(Qt.NoPen)
                cx, cy = child.state.creature_pos
                painter.drawRect(cx * cell_size, cy * cell_size, cell_size, cell_size)
            
        # Draw Trail
        painter.setBrush(QColor(0, 150, 255, 30))
        for tx, ty in self.trail:
            painter.drawRoundedRect(tx * cell_size + 8, ty * cell_size + 8, cell_size - 16, cell_size - 16, 3, 3)
            
        # Draw Obstacles
        painter.setBrush(QColor(70, 70, 70))
        for ox, oy in self.state.obstacles:
            painter.drawRect(ox * cell_size, oy * cell_size, cell_size, cell_size)
            
        # Draw Food
        for fx, fy in self.state.food_positions:
            painter.setBrush(QColor(50, 255, 50, 150))
            painter.drawEllipse(fx * cell_size + 5, fy * cell_size + 5, cell_size - 10, cell_size - 10)
            painter.setBrush(QColor(50, 255, 50))
            painter.drawEllipse(fx * cell_size + 12, fy * cell_size + 12, cell_size - 24, cell_size - 24)
        
        # Draw Predators
        for px, py in self.state.predators:
            painter.setBrush(QColor(255, 50, 50, 150))
            painter.drawEllipse(px * cell_size + 2, py * cell_size + 2, cell_size - 4, cell_size - 4)
            painter.setBrush(QColor(255, 50, 50))
            painter.drawEllipse(px * cell_size + 10, py * cell_size + 10, cell_size - 20, cell_size - 20)
        
        # Draw Creature
        cx, cy = self.state.creature_pos
        painter.setBrush(QColor(0, 200, 255))
        painter.drawRoundedRect(cx * cell_size + 5, cy * cell_size + 5, cell_size - 10, cell_size - 10, 5, 5)
        
        # Intent Arrow
        if self.chosen_action is not None:
            dx, dy = ACTIONS[self.chosen_action]
            c_x, c_y = cx * cell_size + cell_size / 2, cy * cell_size + cell_size / 2
            e_x, e_y = c_x + (dx * cell_size * 0.4), c_y + (dy * cell_size * 0.4)
            painter.setPen(QPen(QColor(255, 255, 255), 3))
            painter.drawLine(c_x, c_y, e_x, e_y)

        # Draw Particles
        for p in self.particles:
            color = p['color']
            color.setAlpha(int(p['life'] * 255))
            painter.setBrush(color)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(p['x'] * cell_size + cell_size/2, p['y'] * cell_size + cell_size/2, 4, 4)

class TreeCanvas(QWidget):
    """Visualizes the MCTS Search Tree"""
    def __init__(self):
        super().__init__()
        self.root_node = None
        self.setMinimumWidth(300)

    def update_tree(self, root_node):
        self.root_node = root_node
        self.update()

    def paintEvent(self, event):
        if not self.root_node: return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(20, 20, 20))
        
        self.draw_node(painter, self.root_node, self.width() / 2, 30, self.width() / 4)

    def draw_node(self, painter, node, x, y, x_spread):
        if not node: return
        
        # Calculate node properties
        visits = node.visit_count
        val = node.q_value()
        
        # Color: Red (-1) to Green (1)
        r = int(max(0, -val) * 255)
        g = int(max(0, val) * 255)
        color = QColor(r, g, 0)
        
        # Size: based on visits
        radius = max(4, min(20, math.log(visits + 1) * 5)) if visits > 0 else 4
        
        # Draw Edges & Children
        num_children = len(node.children)
        if num_children > 0:
            child_x_spread = x_spread / 1.5
            start_x = x - (num_children - 1) * child_x_spread / 2
            
            for i, child in enumerate(node.children):
                cx = start_x + i * child_x_spread
                cy = y + 60
                
                painter.setPen(QPen(QColor(80, 80, 80), 1))
                painter.drawLine(x, y, cx, cy)
                self.draw_node(painter, child, cx, cy, child_x_spread)

        # Draw Node Circle
        painter.setBrush(color)
        painter.setPen(QPen(QColor(255, 255, 255, 100), 1))
        painter.drawEllipse(QPointF(x, y), radius, radius)
        
        # Text
        if radius > 8:
            painter.setPen(QColor(255, 255, 255))
            font = QFont("Arial", 7)
            painter.setFont(font)
            painter.drawText(QRectF(x - 20, y - 5, 40, 10), Qt.AlignCenter, f"V:{visits}")

class HistoryCanvas(QWidget):
    """Draws the generational score line graph"""
    def __init__(self):
        super().__init__()
        self.scores = []
        self.setMinimumHeight(100)

    def add_score(self, score):
        self.scores.append(score)
        if len(self.scores) > 20:
            self.scores.pop(0)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(25, 25, 25))
        painter.setPen(QPen(QColor(50, 50, 50), 1))
        
        # Draw Grid lines
        for i in range(1, 5):
            painter.drawLine(0, self.height() * i / 5, self.width(), self.height() * i / 5)
            
        if len(self.scores) < 2: return
        
        max_score = max(self.scores) if self.scores else 1
        if max_score == 0: max_score = 1
        
        # Draw Line
        points = []
        for i, s in enumerate(self.scores):
            x = (i / (len(self.scores) - 1)) * self.width()
            y = self.height() - (s / max_score) * (self.height() - 10) - 5
            points.append(QPointF(x, y))
            
        painter.setPen(QPen(QColor(0, 200, 255), 2))
        for i in range(len(points) - 1):
            painter.drawLine(points[i], points[i+1])
            
        painter.setBrush(QColor(0, 200, 255))
        for p in points:
            painter.drawEllipse(p, 3, 3)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeepMind PUCT Observatory v4.0")
        self.generation = 1
        self.best_score = 0
        
        self.env = EvolutionEnv()
        self.nn = MockNeuralNetwork()
        self.agent = PUCTAgent(self.env, self.nn)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left Panel (Grid + History Graph)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        self.stats_label = QLabel("Status: Ready")
        self.stats_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #00FF00; padding: 5px;")
        left_layout.addWidget(self.stats_label)
        
        self.gen_label = QLabel("Generation: 1 | Best Score: 0")
        self.gen_label.setStyleSheet("font-size: 12px; color: #00CCFF; padding: 5px;")
        left_layout.addWidget(self.gen_label)
        
        self.canvas = SimCanvas()
        left_layout.addWidget(self.canvas, alignment=Qt.AlignCenter)
        
        left_layout.addWidget(QLabel("Evolution History (Scores):"))
        self.history_canvas = HistoryCanvas()
        left_layout.addWidget(self.history_canvas)
        
        splitter.addWidget(left_panel)
        
        # Right Panel (Tree Viz + Controls)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        right_layout.addWidget(QLabel("<b style='color:#00CCFF'>Live PUCT Search Tree</b>"))
        right_layout.addWidget(QLabel("<i style='color:#666'>Size=Visits, Color=Value(Q)</i>"))
        self.tree_canvas = TreeCanvas()
        right_layout.addWidget(self.tree_canvas, 1)
        
        # Action Probabilities
        right_layout.addWidget(QLabel("<b>Action Probabilities (MCTS):</b>"))
        self.action_bars = {}
        for name, act in [("Right", 0), ("Left", 1), ("Down", 2), ("Up", 3)]:
            row = QHBoxLayout()
            lbl = QLabel(name)
            lbl.setFixedWidth(50)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(True)
            row.addWidget(lbl)
            row.addWidget(bar)
            right_layout.addLayout(row)
            self.action_bars[act] = bar
            
        right_layout.addStretch()
        
        # Controls
        self.start_btn = QPushButton("Start / Restart")
        self.start_btn.clicked.connect(self.start_sim)
        right_layout.addWidget(self.start_btn)
        
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.pause_btn.setEnabled(False)
        right_layout.addWidget(self.pause_btn)
        
        right_layout.addWidget(QLabel("MCTS Simulations:"))
        self.sim_slider = QSlider(Qt.Horizontal)
        self.sim_slider.setRange(10, 300)
        self.sim_slider.setValue(80)
        right_layout.addWidget(self.sim_slider)
        
        right_layout.addWidget(QLabel("Speed (Delay ms):"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(0, 500)
        self.speed_slider.setValue(100)
        self.speed_slider.valueChanged.connect(self.update_speed)
        right_layout.addWidget(self.speed_slider)

        splitter.addWidget(right_panel)
        splitter.setSizes([600, 400])
        
        self.canvas.update_state(self.env.get_state())
        self.update_stats()

    def start_sim(self):
        self.env.reset()
        self.canvas.update_state(self.env.get_state())
        self.update_stats()
        
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.pause_btn.setText("Pause")
        
        self.ai_thread = AIThread(self.env, self.agent)
        self.ai_thread.action_ready.connect(self.execute_action)
        self.ai_thread.simulations = self.sim_slider.value()
        self.ai_thread.delay_ms = self.speed_slider.value()
        self.ai_thread.start()

    def toggle_pause(self):
        if self.ai_thread.isRunning():
            self.ai_thread.paused = not self.ai_thread.paused
            self.pause_btn.setText("Resume" if self.ai_thread.paused else "Pause")

    def update_speed(self, val):
        if hasattr(self, 'ai_thread') and self.ai_thread.isRunning():
            self.ai_thread.delay_ms = val

    def execute_action(self, action, probs, root_node):
        # Check if creature moves into food to trigger VFX
        old_pos = self.env.state.creature_pos
        self.env.move_predators()
        state, reward, done = self.env.step(action)
        
        if state.creature_pos in state.food_positions and old_pos != state.creature_pos:
            self.canvas.spawn_particles(state.creature_pos, QColor(50, 255, 50), count=15)
            
        self.canvas.update_state(state, root_node, action)
        self.tree_canvas.update_tree(root_node)
        self.update_stats()
        
        for act, prob in probs.items():
            self.action_bars[act].setValue(int(prob * 100))
        
        if done:
            self.ai_thread.running = False
            self.ai_thread.quit()
            self.start_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
            
            cause = "Eaten!" if state.creature_pos in state.predators else "Starved!"
            self.stats_label.setText(f"Status: Died ({cause}) Score: {state.score}")
            
            if state.score > self.best_score:
                self.best_score = state.score
                
            self.generation += 1
            self.gen_label.setText(f"Generation: {self.generation} | Best Score: {self.best_score}")
            self.history_canvas.add_score(state.score)
            
            QTimer.singleShot(2000, self.start_sim)

    def update_stats(self):
        state = self.env.get_state()
        self.stats_label.setText(f"Energy: {state.energy} | Score: {state.score} | Steps: {state.steps_survived}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet("""
        QWidget { background-color: #121212; color: #ddd; font-family: Consolas, Arial; }
        QPushButton { background-color: #0088CC; border: none; padding: 8px; border-radius: 4px; color: white; font-weight: bold; }
        QPushButton:hover { background-color: #00AAFF; }
        QPushButton:disabled { background-color: #333; color: #666; }
        QProgressBar { background-color: #222; border: 1px solid #333; text-align: center; border-radius: 4px; color: white; }
        QProgressBar::chunk { background-color: #00CCFF; border-radius: 3px; }
        QSlider::groove:horizontal { background: #222; height: 6px; border-radius: 3px; }
        QSlider::handle:horizontal { background: #00CCFF; width: 14px; margin: -5px 0; border-radius: 7px; }
        QSplitter::handle { background-color: #333; }
        QLabel { color: #ccc; }
    """)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())