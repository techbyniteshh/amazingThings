import argparse
import math
import os
import sys
import cv2
import numpy as np
import pygame

# ==============================================================================
# CONFIGURATION (VINTAGE ENGRAVING & LIVE REEL DRAWING)
# ==============================================================================
WIN_W, WIN_H = 620, 880          # Portrait canvas matching reference aspect ratio
OUTER_PAD = 22                   # Outer white space around frame
INNER_PAD = 42                   # Vintage cream margin inside black border
STROKES_PER_FRAME = 110          # Drawing speed (zyada = fast, kam = slow cinematic)
FPS = 60

PAPER_COLOR = (244, 239, 228)    # Warm vintage ivory/parchment tone
INK_DARK = (24, 21, 18)          # Classic charcoal/sepia engraving ink
INK_MID = (52, 47, 42)
INK_LIGHT = (95, 88, 78)


def create_parchment_canvas(w, h):
    """Generates warm vintage paper texture with subtle organic grain and double border."""
    rng = np.random.default_rng(42)
    paper = np.zeros((h, w, 3), dtype=np.float32)
    paper[:, :, 0] = PAPER_COLOR[0]
    paper[:, :, 1] = PAPER_COLOR[1]
    paper[:, :, 2] = PAPER_COLOR[2]

    # Subtle paper fiber grain
    noise = rng.normal(0, 4.5, (h, w, 1)).astype(np.float32)
    paper = np.clip(paper + noise, 0, 255).astype(np.uint8)

    # Outer white edge outside the main thin black border
    paper[:OUTER_PAD, :] = (255, 255, 255)
    paper[h - OUTER_PAD:, :] = (255, 255, 255)
    paper[:, :OUTER_PAD] = (255, 255, 255)
    paper[:, w - OUTER_PAD:] = (255, 255, 255)

    # Outer crisp black frame line
    cv2.rectangle(paper, (OUTER_PAD, OUTER_PAD), (w - OUTER_PAD, h - OUTER_PAD), (20, 18, 16), 2)
    # Inner artwork border box
    cv2.rectangle(paper, (INNER_PAD - 2, INNER_PAD - 2), (w - INNER_PAD + 2, h - INNER_PAD + 2), (30, 26, 22), 2)
    return paper


def extract_hatching_strokes(gray, angle_deg, step, threshold, color, thickness, offset_x, offset_y):
    """
    Rotates the grayscale image, scans parallel ink runs darker than `threshold`,
    and transforms coordinates back to build authentic copperplate cross-hatching lines.
    """
    ih, iw = gray.shape
    diag = int(math.hypot(iw, ih)) + 16
    pad_x = (diag - iw) // 2
    pad_y = (diag - ih) // 2

    padded = np.full((diag, diag), 255, dtype=np.uint8)
    padded[pad_y:pad_y + ih, pad_x:pad_x + iw] = gray

    center = (diag / 2.0, diag / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    M_inv = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)
    rotated = cv2.warpAffine(padded, M, (diag, diag), borderValue=255)

    strokes = []
    for y in range(0, diag, step):
        row = rotated[y] < threshold
        if not np.any(row):
            continue

        # Find contiguous dark runs along the scanline
        diff = np.diff(row.astype(np.int8))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        if row[0]:
            starts = np.insert(starts, 0, 0)
        if row[-1]:
            ends = np.append(ends, diag)

        for sx, ex in zip(starts, ends):
            if ex - sx < 2:
                continue
            # Transform start and end points back to artwork coordinates
            p1 = M_inv @ np.array([sx, y, 1.0])
            p2 = M_inv @ np.array([ex, y, 1.0])

            x1 = int(np.clip(p1[0] - pad_x, 0, iw - 1)) + offset_x
            y1 = int(np.clip(p1[1] - pad_y, 0, ih - 1)) + offset_y
            x2 = int(np.clip(p2[0] - pad_x, 0, iw - 1)) + offset_x
            y2 = int(np.clip(p2[1] - pad_y, 0, ih - 1)) + offset_y
            strokes.append((x1, y1, x2, y2, color, thickness))

    return strokes


def extract_contour_strokes(gray, offset_x, offset_y):
    """Extracts sharp ink outlines for eyes, smile, hands, and drapery details."""
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 55, 145)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    strokes = []
    for cnt in contours:
        if len(cnt) < 4:
            continue
        pts = cnt.reshape(-1, 2)
        for i in range(0, len(pts) - 1, 2):
            x1, y1 = int(pts[i][0]) + offset_x, int(pts[i][1]) + offset_y
            x2, y2 = int(pts[i + 1][0]) + offset_x, int(pts[i + 1][1]) + offset_y
            strokes.append((x1, y1, x2, y2, INK_DARK, 1))
    return strokes


def build_all_engraving_strokes(image_path):
    """Loads input photo and converts it into layered cross-hatching ink strokes."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] '{image_path}' nahi mila!")
        print("Koi bhi Mona Lisa ki photo download karke is folder mein 'monalisa.jpg' naam se save karo.")
        sys.exit(1)

    art_w = WIN_W - 2 * INNER_PAD
    art_h = WIN_H - 2 * INNER_PAD

    # Center-crop input image to match artwork aspect ratio
    h, w = img.shape[:2]
    target_ratio = art_w / art_h
    cur_ratio = w / h
    if cur_ratio > target_ratio:
        new_w = int(h * target_ratio)
        x0 = (w - new_w) // 2
        img = img[:, x0:x0 + new_w]
    else:
        new_h = int(w / target_ratio)
        y0 = (h - new_h) // 2
        img = img[y0:y0 + new_h, :]

    resized = cv2.resize(img, (art_w, art_h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Enhance local contrast so face, eyes & background landscape look crisp
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    print("[ENGINE] Generating multi-layer copperplate cross-hatching strokes...")
    all_strokes = []

    # Layer 1: Fine vertical etching lines (sky, soft skin shading & background)
    all_strokes.extend(extract_hatching_strokes(gray, 88, step=4, threshold=205, color=INK_LIGHT, thickness=1, offset_x=INNER_PAD, offset_y=INNER_PAD))

    # Layer 2: Diagonal 45-degree midtone hatching
    all_strokes.extend(extract_hatching_strokes(gray, 42, step=4, threshold=155, color=INK_MID, thickness=1, offset_x=INNER_PAD, offset_y=INNER_PAD))

    # Layer 3: Cross-diagonal -45-degree shadow hatching
    all_strokes.extend(extract_hatching_strokes(gray, -42, step=4, threshold=110, color=INK_DARK, thickness=1, offset_x=INNER_PAD, offset_y=INNER_PAD))

    # Layer 4: Dense deep shadows (hair, dark dress folds)
    all_strokes.extend(extract_hatching_strokes(gray, 6, step=3, threshold=65, color=INK_DARK, thickness=1, offset_x=INNER_PAD, offset_y=INNER_PAD))

    # Layer 5: Fine ink contours for sharp facial details
    all_strokes.extend(extract_contour_strokes(gray, INNER_PAD, INNER_PAD))

    print(f"[READY] Total ink strokes generated: {len(all_strokes):,}")
    return all_strokes


def main():
    parser = argparse.ArgumentParser(description="Live Vintage Copperplate Engraving Sketcher in Python")
    parser.add_argument("--image", type=str, default="monalisa.jpg", help="Input image path (default: monalisa.jpg)")
    parser.add_argument("--output", type=str, default="monalisa_engraving_output.png", help="Final output PNG filename")
    parser.add_argument("--speed", type=int, default=STROKES_PER_FRAME, help="Strokes drawn per frame")
    args = parser.parse_args()

    strokes = build_all_engraving_strokes(args.image)
    base_paper = create_parchment_canvas(WIN_W, WIN_H)
    canvas = base_paper.copy()

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Python Engraving Sketcher [SPACE: Pause | R: Restart | S: Save | ESC: Quit]")
    clock = pygame.time.Clock()

    idx = 0
    total = len(strokes)
    paused = False
    saved = False
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    canvas = base_paper.copy()
                    idx = 0
                    saved = False
                elif event.key == pygame.K_s:
                    cv2.imwrite(args.output, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
                    print(f"[SAVED] Engraving image saved to '{args.output}'")

        if not paused and idx < total:
            end_idx = min(total, idx + args.speed)
            for s in strokes[idx:end_idx]:
                x1, y1, x2, y2, col, thick = s
                cv2.line(canvas, (x1, y1), (x2, y2), col, thick, cv2.LINE_AA)
            idx = end_idx

            # Auto-save final PNG once sketching finishes
            if idx >= total and not saved:
                cv2.imwrite(args.output, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
                print(f"[DONE] Drawing complete! Saved to '{args.output}'")
                saved = True

        # Render onto Pygame window
        surf = pygame.surfarray.make_surface(canvas.transpose([1, 0, 2]))
        screen.blit(surf, (0, 0))

        # Subtle ink nib cursor while actively drawing
        if idx < total and not paused:
            cur_x, cur_y = strokes[idx - 1][2], strokes[idx - 1][3]
            pygame.draw.circle(screen, (180, 40, 40), (cur_x, cur_y), 3)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()