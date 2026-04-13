import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# 1. maze
# =========================================================
def double_t_maze():
    x = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1], 
        [1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1], 
        [1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1], 
        [1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1], 
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ], dtype=np.uint8)
    return x


# =========================================================
# 2. Visualization helpers
# =========================================================
def show_raw_maze(maze):
    plt.figure(figsize=(6, 4))
    plt.imshow(maze, cmap="gray_r")
    plt.title("Raw Maze")
    plt.axis("off")
    plt.show()


def show_processed_image(image_tensor, title="Processed Maze Image"):
    """
    image_tensor: [1, 1, H, W] or [B, 1, H, W]
    """
    img = image_tensor[0, 0].detach().cpu().numpy()
    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
    plt.title(title)
    plt.axis("off")
    plt.show()


def show_four_input_quadrants(image_tensor):
    """
    Show the 4 image regions that roughly correspond to the 4 visual tokens.
    """
    img = image_tensor[0, 0].detach().cpu().numpy()
    H, W = img.shape
    h2, w2 = H // 2, W // 2

    patches = [
        img[:h2, :w2],
        img[:h2, w2:],
        img[h2:, :w2],
        img[h2:, w2:]
    ]

    titles = ["Token 0 Region (Top-Left)",
              "Token 1 Region (Top-Right)",
              "Token 2 Region (Bottom-Left)",
              "Token 3 Region (Bottom-Right)"]

    plt.figure(figsize=(8, 8))
    for i, p in enumerate(patches):
        plt.subplot(2, 2, i + 1)
        plt.imshow(p, cmap="gray", vmin=0.0, vmax=1.0)
        plt.title(titles[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# =========================================================
# 3. Preprocess maze -> image tensor
# =========================================================
class MazePreprocessor:
    """
    Convert maze grid to CNN input image.

    Input maze:
        1 = wall
        0 = path

    Output:
        tensor [1, 1, out_size, out_size]
        wall -> 0.0 (black)
        path -> 1.0 (white)
    """
    def __init__(self, out_size=32):
        self.out_size = out_size

    def __call__(self, maze: np.ndarray) -> torch.Tensor:
        if not isinstance(maze, np.ndarray):
            raise TypeError(f"maze must be np.ndarray, got {type(maze)}")

        if maze.ndim != 2:
            raise ValueError(f"maze must be 2D, got shape {maze.shape}")

        # wall=1 -> black=0, path=0 -> white=1
        img = np.where(maze == 1, 0.0, 1.0).astype(np.float32)

        # [H, W] -> [1, 1, H, W]
        x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)

        # Resize with nearest to preserve grid structure
        x = F.interpolate(
            x,
            size=(self.out_size, self.out_size),
            mode="nearest"
        )

        return x


# =========================================================
# 4. CNN backbone
# =========================================================
class CNNBackbone(nn.Module):
    """
    Input:  [B, 1, 32, 32]
    Output: [B, 128, 4, 4]
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 32 -> 16

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 16 -> 8

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)    # 8 -> 4
        )

    def forward(self, x):
        return self.net(x)


# =========================================================
# 5. Convert CNN feature map -> 4 TOKENS
# =========================================================
class SpatialTokenExtractor(nn.Module):
    """
    Convert feature map [B, C, 4, 4] into exactly 4 spatial tokens.
    Each token comes from one 2x2 spatial patch:
        top-left, top-right, bottom-left, bottom-right

    Output:
        [B, 4, gpt_dim]
    """
    def __init__(self, in_channels=128, gpt_dim=768):
        super().__init__()
        self.proj = nn.Linear(in_channels * 2 * 2, gpt_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        if H != 4 or W != 4:
            raise ValueError(f"Expected feature map [B, C, 4, 4], got {x.shape}")

        tokens = []

        for i in range(2):
            for j in range(2):
                patch = x[:, :, i * 2:(i + 1) * 2, j * 2:(j + 1) * 2]  # [B, C, 2, 2]
                patch = patch.reshape(B, -1)                            # [B, C*2*2]
                token = self.proj(patch)                                # [B, gpt_dim]
                tokens.append(token)

        tokens = torch.stack(tokens, dim=1)  # [B, 4, gpt_dim]
        return tokens


# =========================================================
# 6. Full maze -> 4 visual token model
# =========================================================
class MazeTo4Tokens(nn.Module):
    def __init__(self, gpt_dim=768):
        super().__init__()
        self.cnn = CNNBackbone()
        self.tokenizer = SpatialTokenExtractor(in_channels=128, gpt_dim=gpt_dim)

        # Learnable positional embeddings for the 4 visual tokens
        self.pos_embed = nn.Parameter(torch.randn(1, 4, gpt_dim) * 0.02)

    def forward(self, x):
        """
        x: [B, 1, 32, 32]
        returns:
            visual_tokens: [B, 4, 768]
            feat_map: [B, 128, 4, 4]
        """
        feat_map = self.cnn(x)
        visual_tokens = self.tokenizer(feat_map)
        visual_tokens = visual_tokens + self.pos_embed
        return visual_tokens, feat_map


# =========================================================
# 7. Combine image tokens with GPT-2 text token embeddings
# =========================================================
def combine_with_gpt2(visual_tokens, input_ids, gpt2_model):
    """
    visual_tokens: [B, 4, D]
    input_ids:     [B, T]
    gpt2_model: HuggingFace GPT2LMHeadModel or GPT2Model

    returns:
        combined_embeds: [B, 4+T, D]
        attention_mask:  [B, 4+T]
    """
    text_embeds = gpt2_model.transformer.wte(input_ids)  # [B, T, D]

    if visual_tokens.size(0) != text_embeds.size(0):
        raise ValueError("Batch size mismatch")

    if visual_tokens.size(2) != text_embeds.size(2):
        raise ValueError("Embedding dim mismatch")

    combined_embeds = torch.cat([visual_tokens, text_embeds], dim=1)

    B = combined_embeds.size(0)
    total_len = combined_embeds.size(1)
    attention_mask = torch.ones(B, total_len, dtype=torch.long, device=combined_embeds.device)

    return combined_embeds, attention_mask


# =========================================================
# 8. Optional: CNN feature maps
# =========================================================
def show_feature_maps(feature_map, num_channels=8):
    """
    feature_map: [B, C, 4, 4]
    """
    fmap = feature_map[0].detach().cpu()
    num_channels = min(num_channels, fmap.size(0))

    plt.figure(figsize=(12, 4))
    for i in range(num_channels):
        plt.subplot(2, (num_channels + 1) // 2, i + 1)
        plt.imshow(fmap[i], cmap="viridis")
        plt.title(f"Ch {i}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# =========================================================
# 9. Main demo
# =========================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -----------------------------
    # Step A: create maze
    # -----------------------------
    maze = double_t_maze()
    print("Raw maze shape:", maze.shape)

    show_raw_maze(maze)

    # -----------------------------
    # Step B: preprocess maze image
    # -----------------------------
    preprocessor = MazePreprocessor(out_size=32)
    image_tensor = preprocessor(maze).to(device)

    print("Processed image tensor shape:", image_tensor.shape)  # [1,1,32,32]
    show_processed_image(image_tensor, title="Processed Maze (32x32)")
    show_four_input_quadrants(image_tensor)

    # -----------------------------
    # Step C: CNN -> 4 visual tokens
    # -----------------------------
    model = MazeTo4Tokens(gpt_dim=768).to(device)
    visual_tokens, feat_map = model(image_tensor)

    print("Feature map shape:", feat_map.shape)        # [1,128,4,4]
    print("Visual tokens shape:", visual_tokens.shape) # [1,4,768]

    show_feature_maps(feat_map, num_channels=8)

    # -----------------------------
    # Step D: Optional GPT-2 example
    # -----------------------------
    use_gpt2_demo = True

    if use_gpt2_demo:
        try:
            from transformers import GPT2Tokenizer, GPT2LMHeadModel

            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            gpt2 = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

            # GPT-2 has no pad token by default
            tokenizer.pad_token = tokenizer.eos_token
            gpt2.config.pad_token_id = tokenizer.eos_token_id

            text = "You are at the start of the maze."
            encoded = tokenizer(text, return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)

            print("Text input_ids shape:", input_ids.shape)

            combined_embeds, attention_mask = combine_with_gpt2(
                visual_tokens=visual_tokens,
                input_ids=input_ids,
                gpt2_model=gpt2
            )

            print("Combined embeds shape:", combined_embeds.shape)
            print("Attention mask shape:", attention_mask.shape)

            outputs = gpt2(
                inputs_embeds=combined_embeds,
                attention_mask=attention_mask
            )

            print("GPT-2 logits shape:", outputs.logits.shape)

        except ImportError:
            print("transformers is not installed. Run:")
            print("pip install transformers")