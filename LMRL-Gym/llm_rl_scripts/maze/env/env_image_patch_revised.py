# llm_rl_scripts/maze/env/env.py

from typing import Callable, Optional, Dict, List, Tuple
from LLM_RL.environment import Text, TextEnv, TextHistory
import numpy as np
import random
from llm_rl_scripts.maze.env.randomness import RandomState
from IPython import embed
from PIL import Image

try:
    from IPython.display import display
except Exception:
    display = None

# =========================
# NEW: VISUAL OBS HELPERS
# =========================

def extract_centered_patch(
    maze: np.ndarray,
    position: Tuple[int, int],
    patch_size: int = 3,
    pad_value: int = 1,
    mark_agent: bool = True,
) -> np.ndarray:
    """
    Returns a patch_size x patch_size crop centered at the agent.
    pad_value=1 means outside-maze is treated as wall.
    Output values:
      0 = free
      1 = wall
      2 = agent center (optional)
    """
    assert patch_size % 2 == 1, "patch_size must be odd"
    radius = patch_size // 2

    patch = np.full((patch_size, patch_size), pad_value, dtype=np.int32)

    y0, x0 = position
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            yy = y0 + dy
            xx = x0 + dx
            py = dy + radius
            px = dx + radius
            if 0 <= yy < maze.shape[0] and 0 <= xx < maze.shape[1]:
                patch[py, px] = int(maze[yy, xx])

    if mark_agent:
        patch[radius, radius] = 2

    return patch


def render_patch_ascii(patch: np.ndarray) -> str:
    """
    Human-readable printout for debugging.
      1 -> '#'
      0 -> '.'
      2 -> 'A'
    """
    symbol_map = {
        0: ".",
        1: "#",
        2: "A",
    }
    rows = []
    for row in patch:
        rows.append(" ".join(symbol_map.get(int(v), "?") for v in row))
    return "\n".join(rows)


def render_patch_image(
    patch: np.ndarray,
    cell_size: int = 40,
    upscale: int = 4,
    wall_color: Tuple[int, int, int] = (140, 140, 140),
    path_color: Tuple[int, int, int] = (245, 245, 245),
    agent_color: Tuple[int, int, int] = (255, 0, 0),
    draw_grid: bool = True,
) -> Image.Image:
    """
    Render a numeric local patch into an RGB image that visually matches the
    global maze style:
      1 -> wall (gray)
      0 -> free path (light)
      2 -> agent (red)
    """
    h, w = patch.shape
    img = np.zeros((h * cell_size, w * cell_size, 3), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            if int(patch[i, j]) == 1:
                color = wall_color
            elif int(patch[i, j]) == 0:
                color = path_color
            elif int(patch[i, j]) == 2:
                color = agent_color
            else:
                color = (0, 0, 0)

            img[
                i * cell_size:(i + 1) * cell_size,
                j * cell_size:(j + 1) * cell_size,
            ] = color

    if draw_grid:
        # thin black grid lines
        for i in range(h + 1):
            y = min(i * cell_size, img.shape[0] - 1)
            img[max(y - 1, 0):min(y + 1, img.shape[0]), :, :] = 0
        for j in range(w + 1):
            x = min(j * cell_size, img.shape[1] - 1)
            img[:, max(x - 1, 0):min(x + 1, img.shape[1]), :] = 0

    pil_img = Image.fromarray(img)
    if upscale > 1:
        pil_img = pil_img.resize(
            (pil_img.width * upscale, pil_img.height * upscale),
            Image.NEAREST,
        )
    return pil_img


def show_patch_image(patch: np.ndarray) -> None:
    """
    Display the rendered local patch in notebook / IPython environments.
    Falls back to printing the ASCII version if inline display is unavailable.
    """
    img = render_patch_image(patch)
    if display is not None:
        display(img)
    else:
        print(render_patch_ascii(patch))


def build_visual_extras(
    maze: np.ndarray,
    position: Tuple[int, int],
    patch_size: int = 3,
) -> Dict[str, np.ndarray]:
    """
    extras attached to Text observation.
    """
    patch = extract_centered_patch(
        maze=maze,
        position=position,
        patch_size=patch_size,
        pad_value=1,
        mark_agent=True,
    )
    patch_img = render_patch_image(patch)
    return {
        "local_patch": patch.astype(np.int32),
        "local_patch_ascii": render_patch_ascii(patch),
        "local_patch_image": patch_img,
        "local_patch_image_array": np.array(patch_img, dtype=np.uint8),
        "agent_position": tuple(position),
    }


# existing code-describe objects and observations
def describe_objects(object: str, relations: List[str]):
    if len(relations) == 0:
        return f"There are no {object}s near you."
    if len(relations) == 1:
        return f"There is a {object} {relations[0]}."
    return f"There are {object}s {', '.join(relations)}."

def describe_observation(maze: np.ndarray, 
                         position: Tuple[int, int], 
                         goal_position: Tuple[int, int], 
                         initial_position: Tuple[int, int]=None,
                         move_history: List[str]=None,
                         ) -> int:
    assert len(maze.shape) == 2

    goal_description = f"The goal is at position {' '.join(str(goal_position[0]))}, {' '.join(str(goal_position[1]))}."
    # if initial_position is not None:
    #     initial_description = f"Your starting position is at position {' '.join(str(initial_position[0]))}, {' '.join(str(initial_position[1]))}."
    delta_descriptions = {"to your right": (0, 1), "to your left": (0, -1), "above you": (-1, 0), "below you": (1, 0)} 
                        #   "to your top left": (-1, -1), "to your top right": (-1, 1), 
                        #   "to your bottom left": (1, -1), "to your bottom right": (1, 1)}
    walls = []
    # goals = []
    for k, (dy, dx) in delta_descriptions.items():
        if maze[position[0]+dy, position[1]+dx] == 1:
            walls.append(k)
        # elif maze[position[0]+dy, position[1]+dx] == 3:
        #     goals.append(k)
    
    wall_description = describe_objects("wall", walls)
    
    # # history description
    # history_description = ""
    # if move_history is not None:
    #     history_description = f"Your move history is {' '.join(move_history)}."

    # goal_location_description = describe_objects("goal", goals)

    # return f"{goal_description} {wall_description} {goal_location_description}\n"
    # if initial_position is not None:
    #     return f"{goal_description} {initial_description} {history_description} {wall_description}\n"
    return f"{goal_description} {wall_description}\n"

def describe_observation_give_position(maze:np.ndarray,
                                       position: Tuple[int, int],
                                       goal_position: Tuple[int, int],
                                       initial_position: Tuple[int, int]=None,
                                       move_history: List[str]=None,
                                       ) -> str:
    goal_description = f"The goal is at position {' '.join(str(goal_position[0]))}, {' '.join(str(goal_position[1]))}."
    curr_position_description = f"Your current position is at position {' '.join(str(position[0]))}, {' '.join(str(position[1]))}."
    delta_descriptions = {"to your right": (0, 1), "to your left": (0, -1), "above you": (-1, 0), "below you": (1, 0)} 

    walls = []
    for k, (dy, dx) in delta_descriptions.items():
        if maze[position[0]+dy, position[1]+dx] == 1:
            walls.append(k)
    
    wall_description = describe_objects("wall", walls)
    
    return f"{goal_description} {curr_position_description} {wall_description}\n"

def describe_observation_only_walls(maze:np.ndarray, 
                                    position: Tuple[int, int],
                                    goal_position: Tuple[int, int]=None,
                                    initial_position: Tuple[int, int]=None,
                                    move_history: List[str]=None,) -> str:
    delta_descriptions = {"to your right": (0, 1), "to your left": (0, -1), "above you": (-1, 0), "below you": (1, 0)} 
    walls = []
    for k, (dy, dx) in delta_descriptions.items():
        if maze[position[0]+dy, position[1]+dx] == 1:
            walls.append(k)
    wall_description = describe_objects("wall", walls)
    return f"{wall_description}\n"
    

diagonal_actions = {
    'move left\n': (0, -1), 
    'move right\n': (0, 1), 
    'move up\n': (-1, 0), 
    'move down\n': (1, 0), 
    'move top left\n': (-1, -1), 
    'move top right\n': (-1, 1), 
    'move bottom left\n': (1, -1), 
    'move bottom right\n': (1, 1), 
}

manhatten_actions = {
    'move left\n': (0, -1), 
    'move right\n': (0, 1), 
    'move up\n': (-1, 0), 
    'move down\n': (1, 0), 
}

def maze_proposal_function(text_history: TextHistory) -> List[TextHistory]:
    return [text_history+(Text(action, True),) for action in manhatten_actions.keys()]

def update_position(maze: np.ndarray, position: Tuple[int, int], action: str, actions: Dict[str, Tuple[int, int]]) -> Tuple[int, int]:
    if action in actions and maze[position[0] + actions[action][0], position[1] + actions[action][1]] == 0:
        return (position[0] + actions[action][0], position[1] + actions[action][1])
    return position

def standard_reward(action, goal, position, possible_actions):
    if position[0] == goal[0] and position[1] == goal[1]:
        return 0.0
    elif action not in possible_actions:
        return -4.0
    else:
        return -1.0

def illegal_penalty_reward(action, goal, position, possible_actions):
    if position[0] == goal[0] and position[1] == goal[1]:
        return 1.0
    elif action not in possible_actions:
        return -1.0
    else:
        return 0.0

def illegal_penalty_diff_scale(action, goal, position, possible_actions):
    if position[0] == goal[0] and position[1] == goal[1]:
        return 1.0
    elif action not in possible_actions:
        return -100.0
    else:
        return -1.0

class MazeEnv(TextEnv):
    def __init__(self, maze: np.ndarray, 
                 valid_goals: np.ndarray, 
                 actions: Dict[str, Tuple[int, int]], 
                 max_steps: Optional[int]=None, 
                 display_initial_position: bool=False,
                 describe_function: Callable[[np.ndarray, Tuple[int, int], Tuple[int, int], Optional[Tuple[int, int]], Optional[List[str]]], str]=describe_observation_give_position,
                 reward_function: Callable[[str, Tuple[int, int], Tuple[int, int], Dict[str, Tuple[int, int]]], float]=standard_reward,
                 last_k:int=40,
                 # NEW: VISUAL
                 patch_size: int = 3,
                 print_local_patch: bool = False,
                ):
        
        assert len(maze.shape) == 2
        assert all([maze[goal[0], goal[1]] == 0 for goal in valid_goals])

        self.maze = maze
        self.valid_goals = valid_goals
        self.actions = actions
        self.max_steps = max_steps
        self.display_initial_position = display_initial_position
        self.num_steps = 0
        self.describe_function = describe_function
        self.move_history = []
        self.last_k = last_k
        
        self.reward_function = reward_function
        
        self.random_state = RandomState(None)
        # NEW: VISUAL
        self.patch_size = patch_size
        self.print_local_patch = print_local_patch

        self.reset()
    
    def step(self, text_history: TextHistory) -> Tuple[TextHistory, float, bool]:
        assert text_history[-1].is_action
        # embed()
        if self.max_steps is not None and self.num_steps >= self.max_steps:
            return (Text("Failure\n", False),), -1.0, True
        
        action = text_history[-1].text    
        self.position = update_position(self.maze, self.position, action, self.actions)
        
        self.move_history.append(action.replace('\n', ''))
        
        reward = self.reward_function(action, self.goal, self.position, self.actions)
        if self.position[0] == self.goal[0] and self.position[1] == self.goal[1]:
            success_extras = build_visual_extras(
                maze=self.maze,
                position=self.position,
                patch_size=self.patch_size,
            )
            if self.print_local_patch:
                print("\n[STEP: SUCCESS] local patch:")
                show_patch_image(success_extras["local_patch"])
            return (Text("Success\n", False, extras=success_extras),), reward, True
        
        # move_history = [text_history[i].text for i in range(0, len(text_history), 2) if text_history[i].is_action]
        self.num_steps += 1
        obs_description = self.describe_function(self.maze, self.position, self.goal, self.initial_position, self.move_history)

        # NEW: VISUAL
        obs_extras = build_visual_extras(
            maze=self.maze,
            position=self.position,
            patch_size=self.patch_size,
        )

        if self.print_local_patch:
            print(f"\n[STEP {self.num_steps}] action={action.strip()}")
            show_patch_image(obs_extras["local_patch"])

        if action not in self.actions:
            return (Text(obs_description, False, extras=obs_extras),), reward, False

        new_history = list(text_history) + [Text(obs_description, False, extras= obs_extras)]
        new_history = new_history[max(0, len(new_history)-self.last_k):]
        
        return tuple(new_history), reward, False
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> TextHistory:
        self.random_state.reset(seed)
        self.num_steps = 0

        if options is not None and 'goal' in options:
            self.goal = options['goal']
        else:
            self.goal = random.choice(self.valid_goals).tolist()
        
        positions = np.argwhere(self.maze == 0).tolist()
        positions.remove(self.goal)
        
        if options is not None and 'init_position' in options:
            assert list(options['init_position']) in positions
            self.position = list(options['init_position'])
        else:
            self.position = random.choice(positions)
        
        if self.display_initial_position:
            self.initial_position = self.position.copy()
        else:
            self.initial_position = None

        # NEW: reset move history explicitly
        self.move_history = []
        
        # print('initial position:', self.position)
        obs_description = self.describe_function(self.maze, self.position, self.goal, self.initial_position)

        # NEW: VISUAL
        obs_extras = build_visual_extras(
            maze=self.maze,
            position=self.position,
            patch_size=self.patch_size,
        )

        if self.print_local_patch:
            print("\n[RESET] local patch:")
            show_patch_image(obs_extras["local_patch"])

        self.random_state.freeze()

        return (Text(obs_description, False, extras=obs_extras),)