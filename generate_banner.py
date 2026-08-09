#!/usr/bin/env python3
"""
DDW-X GitHub Profile Animated Banner Generator Pipeline
Generates dark.svg and light.svg animated SVG banners based on:
- 1-bit Serpentine Floyd-Steinberg dithering with strict facial geometry preservation
- Segmented background with hard-cleared error diffusion bleed
- 60-group interleaved shimmer intro animation
- Dual-layer morphing system: 94 noisy drift bands + ~900 optimal-transport traveller dots
- Pixel-locked SYSTEM.INFO terminal readout with dotted leaders
"""

import os
import glob
import math
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from scipy.ndimage import binary_closing, binary_fill_holes, label
from scipy.optimize import linear_sum_assignment

# Canvas Constants
CANVAS_WIDTH = 1180
CANVAS_HEIGHT = 610

# Portrait Grid Constants (Target ~17k-22k dots)
PORTRAIT_W = 300
PORTRAIT_H = 340
PORTRAIT_X = 75
PORTRAIT_Y = 135

# Color Palettes
DARK_PALETTE = {
    "bg": "#0A101F",
    "card_bg": "#0E172A",
    "card_border": "#1E293B",
    "header_bar": "#0F172A",
    "title_text": "#94A3B8",
    "ui_chrome": "#22D3EE",
    "ui_dim": "#0891B2",
    "accent": "#10B981",
    "portrait_dots": "#A78BFA",
    "traveller_dots": "#22D3EE",
    "text_primary": "#F8FAFC",
    "text_muted": "#64748B",
    "text_accent": "#38BDF8",
    "badge_bg": "#1E293B",
    "pill_bg": "rgba(34, 211, 238, 0.12)",
    "pill_border": "#22D3EE",
    "pill_text": "#22D3EE",
    "live_pulse": "#EF4444",
    "dot_leader": "#334155"
}

LIGHT_PALETTE = {
    "bg": "#F8FAFC",
    "card_bg": "#FFFFFF",
    "card_border": "#E2E8F0",
    "header_bar": "#F1F5F9",
    "title_text": "#64748B",
    "ui_chrome": "#0891B2",
    "ui_dim": "#0E7490",
    "accent": "#059669",
    "portrait_dots": "#7C3AED",
    "traveller_dots": "#0891B2",
    "text_primary": "#0F172A",
    "text_muted": "#64748B",
    "text_accent": "#0284C7",
    "badge_bg": "#E2E8F0",
    "pill_bg": "rgba(8, 145, 178, 0.12)",
    "pill_border": "#0891B2",
    "pill_text": "#0891B2",
    "live_pulse": "#DC2626",
    "dot_leader": "#CBD5E1"
}


def find_image_file():
    """Locate the profile photo (pp, pp.jpg, pp.png, etc.)."""
    candidates = ["pp", "pp.jpg", "pp.png", "pp.jpeg", "pp.webp"]
    for c in candidates:
        if os.path.exists(c) and os.path.isfile(c):
            return c
    matches = glob.glob("pp*")
    for m in matches:
        if os.path.isfile(m) and not m.endswith(".py") and not m.endswith(".svg"):
            return m
    raise FileNotFoundError("Could not find 'pp' image file in project root.")


def preprocess_image(img_path):
    """
    Preprocesses the portrait while strictly preserving facial geometry:
    - Head & shoulders framing
    - High-precision resize to 300x340
    - Autocontrast(cutoff=1) + UnsharpMask(radius=3, percent=140)
    - 1.3x contrast boost
    - NO generative smoothing, blurring or structural alteration
    """
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    # Head and shoulders framing crop
    target_aspect = PORTRAIT_W / PORTRAIT_H
    src_aspect = w / h

    if src_aspect > target_aspect:
        new_w = int(h * target_aspect)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    else:
        new_h = int(w / target_aspect)
        top = int((h - new_h) * 0.2)
        img = img.crop((0, top, w, top + new_h))

    img = img.resize((PORTRAIT_W, PORTRAIT_H), Image.Resampling.LANCZOS)

    # Master prompt contrast & sharpening pipeline
    img = ImageOps.autocontrast(img, cutoff=1)
    img = img.filter(ImageFilter.UnsharpMask(radius=3, percent=140))
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.3)

    return img


def segment_background(img_rgb):
    """
    Background segmentation:
    - Measures color distance to corner background samples
    - Binary closing & hole filling
    - Keeps largest connected component (the subject)
    """
    arr = np.array(img_rgb, dtype=np.float32)
    h, w, _ = arr.shape

    # Sample top corners (background)
    corner_samples = np.vstack([
        arr[0:15, 0:15].reshape(-1, 3),
        arr[0:15, -15:].reshape(-1, 3)
    ])
    bg_mean = np.median(corner_samples, axis=0)

    # Color distance
    dist = np.linalg.norm(arr - bg_mean, axis=2)
    thresh = max(25.0, np.percentile(dist, 25))
    fg_mask = dist > thresh

    # Morphological cleanup
    fg_mask = binary_closing(fg_mask, structure=np.ones((5, 5)))
    fg_mask = binary_fill_holes(fg_mask)

    # Keep largest connected component
    labeled, num_features = label(fg_mask)
    if num_features > 0:
        sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]
        largest_label = np.argmax(sizes) + 1
        fg_mask = (labeled == largest_label)

    return fg_mask


def floyd_steinberg_dither(img_rgb, is_dark=True, fg_mask=None):
    """
    1-bit Floyd-Steinberg dither in serpentine order:
    - Hard-clears error diffusion bleed at mask boundary
    - Returns boolean array where True indicates a placed dot
    """
    gray = ImageOps.grayscale(img_rgb)
    arr = np.array(gray, dtype=np.float32)
    h, w = arr.shape

    dither_grid = np.zeros((h, w), dtype=bool)

    if is_dark:
        if fg_mask is not None:
            arr[~fg_mask] = 0.0
    else:
        # Light mode: invert so darker parts draw dots, but keep background white
        if fg_mask is not None:
            arr = 255.0 - arr
            arr[~fg_mask] = 0.0
        else:
            arr = 255.0 - arr

    # Serpentine Floyd-Steinberg
    for y in range(h):
        if y % 2 == 0:
            for x in range(w):
                if fg_mask is not None and not fg_mask[y, x]:
                    arr[y, x] = 0.0
                    continue

                old_val = arr[y, x]
                new_val = 255.0 if old_val >= 128.0 else 0.0
                err = old_val - new_val
                dither_grid[y, x] = (new_val == 255.0)

                if x + 1 < w:
                    arr[y, x + 1] += err * (7.0 / 16.0)
                if y + 1 < h:
                    if x - 1 >= 0:
                        arr[y + 1, x - 1] += err * (3.0 / 16.0)
                    arr[y + 1, x] += err * (5.0 / 16.0)
                    if x + 1 < w:
                        arr[y + 1, x + 1] += err * (1.0 / 16.0)
        else:
            for x in range(w - 1, -1, -1):
                if fg_mask is not None and not fg_mask[y, x]:
                    arr[y, x] = 0.0
                    continue

                old_val = arr[y, x]
                new_val = 255.0 if old_val >= 128.0 else 0.0
                err = old_val - new_val
                dither_grid[y, x] = (new_val == 255.0)

                if x - 1 >= 0:
                    arr[y, x - 1] += err * (7.0 / 16.0)
                if y + 1 < h:
                    if x + 1 < w:
                        arr[y + 1, x + 1] += err * (3.0 / 16.0)
                    arr[y + 1, x] += err * (5.0 / 16.0)
                    if x - 1 >= 0:
                        arr[y + 1, x - 1] += err * (1.0 / 16.0)

    return dither_grid


def generate_morph_logos(num_dots=900):
    """
    Generates 3 cyber/systems logos (~900 dots each) centered at the portrait box:
    1. Code / Kernel Glyph: < / >
    2. CPU Silicon Die / Microchip Core Matrix
    3. Cyber Defense / Offensive Security Shield Matrix
    """
    cx = PORTRAIT_X + PORTRAIT_W / 2.0
    cy = PORTRAIT_Y + PORTRAIT_H / 2.0

    # 1. Logo 1: Code / Kernel Glyph < / >
    l1_pts = []
    for t in np.linspace(-1, 1, num_dots // 3):
        x = cx - 55 - (1 - abs(t)) * 38
        y = cy + t * 75
        l1_pts.append((x, y))
    for t in np.linspace(-1, 1, num_dots // 3):
        x = cx + t * 28
        y = cy - t * 85
        l1_pts.append((x, y))
    for t in np.linspace(-1, 1, num_dots - len(l1_pts)):
        x = cx + 55 + (1 - abs(t)) * 38
        y = cy + t * 75
        l1_pts.append((x, y))

    # 2. Logo 2: CPU Microchip Matrix
    l2_pts = []
    box_s = 65
    for t in np.linspace(-box_s, box_s, 100):
        l2_pts.append((cx + t, cy - box_s))
        l2_pts.append((cx + t, cy + box_s))
        l2_pts.append((cx - box_s, cy + t))
        l2_pts.append((cx + box_s, cy + t))
    inner_s = 32
    for t in np.linspace(-inner_s, inner_s, 40):
        l2_pts.append((cx + t, cy - inner_s))
        l2_pts.append((cx + t, cy + inner_s))
        l2_pts.append((cx - inner_s, cy + t))
        l2_pts.append((cx + inner_s, cy + t))
    rem = num_dots - len(l2_pts)
    pin_step = rem // 4
    for i in range(pin_step):
        offset = -48 + (96 * i / max(1, pin_step - 1))
        l2_pts.append((cx + offset, cy - box_s - 16))
        l2_pts.append((cx + offset, cy + box_s + 16))
        l2_pts.append((cx - box_s - 16, cy + offset))
        l2_pts.append((cx + box_s + 16, cy + offset))
    while len(l2_pts) < num_dots:
        l2_pts.append((cx, cy))
    l2_pts = l2_pts[:num_dots]

    # 3. Logo 3: Cyber Shield Matrix
    l3_pts = []
    shield_pts = num_dots - 180
    for t in np.linspace(0, 1, shield_pts // 2):
        x = cx - 72 * (1 - 0.7 * (t ** 2))
        y = cy - 75 + t * 155
        l3_pts.append((x, y))
        x = cx + 72 * (1 - 0.7 * (t ** 2))
        y = cy - 75 + t * 155
        l3_pts.append((x, y))
    for t in np.linspace(-72, 72, 70):
        l3_pts.append((cx + t, cy - 75))
    for t in np.linspace(-1, 1, 55):
        l3_pts.append((cx - 22 + (1 - abs(t)) * 16, cy - 10 + t * 24))
    for t in np.linspace(-12, 12, num_dots - len(l3_pts)):
        l3_pts.append((cx + 10 + t, cy + 18))

    pts1 = np.array(l1_pts[:num_dots], dtype=np.float32)
    pts2 = np.array(l2_pts[:num_dots], dtype=np.float32)
    pts3 = np.array(l3_pts[:num_dots], dtype=np.float32)

    # Optimal Transport Matching via Linear Assignment
    cost12 = np.linalg.norm(pts1[:, None, :] - pts2[None, :, :], axis=2)
    _, col_ind12 = linear_sum_assignment(cost12)
    pts2_sorted = pts2[col_ind12]

    cost23 = np.linalg.norm(pts2_sorted[:, None, :] - pts3[None, :, :], axis=2)
    _, col_ind23 = linear_sum_assignment(cost23)
    pts3_sorted = pts3[col_ind23]

    return pts1, pts2_sorted, pts3_sorted


def encode_dot_runs(coords):
    """
    RLE compresses coordinate list into SVG path data:
    Draws 1.1x1.1px crisp square dots/runs with integer/1-decimal precision.
    """
    if len(coords) == 0:
        return ""
    sorted_coords = sorted(coords, key=lambda p: (round(p[1]), round(p[0])))
    path_segments = []
    
    i = 0
    n = len(sorted_coords)
    while i < n:
        x, y = round(sorted_coords[i][0]), round(sorted_coords[i][1])
        run_len = 1
        while (i + 1 < n and 
               round(sorted_coords[i + 1][1]) == y and 
               round(sorted_coords[i + 1][0]) == x + run_len):
            run_len += 1
            i += 1
        if run_len == 1:
            path_segments.append(f"M{x},{y}h1v1h-1z")
        else:
            path_segments.append(f"M{x},{y}h{run_len}v1h-{run_len}z")
        i += 1

    return "".join(path_segments)


def build_system_info_svg(palette):
    """
    Builds the pixel-locked SYSTEM.INFO terminal readout with dotted leaders,
    pulsing LIVE badge, and handle pill.
    """
    info_x = 485
    info_y_start = 148
    row_spacing = 29
    total_row_width = 620

    rows = [
        ("Subject", "DDW-X"),
        ("Role", "Core, CNO & Low-Level Systems Architect"),
        ("Origin", "Shiraz, Fars, Iran"),
        ("Education", "Autodidact (CS & Systems Eng)"),
        ("Status", "Architecting Fundamental Tools & CNO"),
        ("ToolChain", "C/C++, ASM, Rust, Python, Ghidra, WinDbg"),
        ("Core.Lang", "C, C++, x86/ARM ASM, Rust, Python"),
        ("Core.LowLevel", "Hypervisor, KMDF/WDK, Linux Kernel"),
        ("Core.OffSec", "Ghidra, IDA Pro, WinDbg, Sysinternals"),
        ("Grid.Telegram", "t.me/CONTROLSERVER"),
        ("Grid.Mail", "ddw.x.dev@gmail.com"),
        ("Grid.GitHub", "github.com/DDW-X")
    ]

    elements = []
    for idx, (label, val) in enumerate(rows):
        y = info_y_start + idx * row_spacing
        
        elements.append(f'<text x="{info_x}" y="{y}" fill="{palette["ui_chrome"]}" font-family="ui-monospace, SFMono-Regular, Consolas, monospace" font-size="13" font-weight="700" letter-spacing="0.5">{label}</text>')
        
        val_x = info_x + total_row_width
        elements.append(f'<text x="{val_x}" y="{y}" text-anchor="end" fill="{palette["text_primary"]}" font-family="ui-monospace, SFMono-Regular, Consolas, monospace" font-size="13" font-weight="500">{val}</text>')
        
        dot_start = info_x + len(label) * 8.2 + 10
        dot_end = val_x - len(val) * 8.2 - 10
        if dot_end > dot_start + 15:
            elements.append(f'<line x1="{dot_start:.1f}" y1="{y-3:.1f}" x2="{dot_end:.1f}" y2="{y-3:.1f}" stroke="{palette["dot_leader"]}" stroke-width="1.5" stroke-dasharray="2 5" stroke-linecap="round" />')

    return "\n    ".join(elements)


def generate_banner_svg(is_dark=True, output_filename=None):
    """
    Compiles the complete animated SVG banner according to Master Prompt specs.
    """
    palette = DARK_PALETTE if is_dark else LIGHT_PALETTE
    if output_filename is None:
        output_filename = "dark.svg" if is_dark else "light.svg"

    img_path = find_image_file()
    img_rgb = preprocess_image(img_path)

    fg_mask = segment_background(img_rgb)
    dither_grid = floyd_steinberg_dither(img_rgb, is_dark=is_dark, fg_mask=fg_mask)

    dot_coords = []
    h, w = dither_grid.shape
    for y in range(h):
        for x in range(w):
            if dither_grid[y, x]:
                dot_coords.append((PORTRAIT_X + x, PORTRAIT_Y + y))

    num_dots = len(dot_coords)
    print(f"[{'DARK' if is_dark else 'LIGHT'}] Dithered portrait dots: {num_dots}")

    # 1. Intro Animation Assignment (60 interleaved random groups)
    np.random.seed(42)
    intro_groups = np.random.randint(0, 60, size=num_dots)
    
    # 2. Loop Animation Assignment (94 drift bands with noise)
    centroid_logo1_x = PORTRAIT_X + PORTRAIT_W / 2.0
    centroid_logo1_y = PORTRAIT_Y + PORTRAIT_H / 2.0
    centroid_portrait_x = np.mean([p[0] for p in dot_coords])
    centroid_portrait_y = np.mean([p[1] for p in dot_coords])

    drift_dx = 0.42 * (centroid_logo1_x - centroid_portrait_x)
    drift_dy = 0.42 * (centroid_logo1_y - centroid_portrait_y)

    noise = np.random.normal(0, 4.0, size=num_dots)
    band_vals = np.array([p[1] for p in dot_coords]) + noise
    min_b, max_b = np.min(band_vals), np.max(band_vals)
    drift_bands = np.clip(((band_vals - min_b) / (max_b - min_b + 1e-5) * 94).astype(int), 0, 93)

    # Group portrait dots into ~60 intro chunks
    intro_paths = []
    for g in range(60):
        grp_coords = [dot_coords[i] for i in range(num_dots) if intro_groups[i] == g]
        path_d = encode_dot_runs(grp_coords)
        if not path_d:
            continue
        delay = (g / 60.0) * 1.8
        intro_paths.append(
            f'<path d="{path_d}" fill="{palette["portrait_dots"]}" shape-rendering="crispEdges" opacity="0">\n'
            f'  <animate attributeName="opacity" from="0" to="1" dur="0.4s" begin="{delay:.2f}s" fill="freeze" />\n'
            f'</path>'
        )

    # Group drift animation into ~94 bands
    drift_layer_groups = []
    for b in range(94):
        b_coords = [dot_coords[i] for i in range(num_dots) if drift_bands[i] == b]
        path_d = encode_dot_runs(b_coords)
        if not path_d:
            continue
        frac = (b / 94.0)
        bdx = drift_dx * (0.6 + 0.4 * frac)
        bdy = drift_dy * (0.6 + 0.4 * frac)

        drift_layer_groups.append(
            f'<g>\n'
            f'  <animateTransform attributeName="transform" type="translate"\n'
            f'    values="0 0; 0 0; {bdx:.1f} {bdy:.1f}; {bdx:.1f} {bdy:.1f}; {bdx:.1f} {bdy:.1f}; {bdx:.1f} {bdy:.1f}; {bdx:.1f} {bdy:.1f}; {bdx:.1f} {bdy:.1f}; 0 0"\n'
            f'    keyTimes="0; 0.211; 0.303; 0.444; 0.535; 0.676; 0.768; 0.908; 1"\n'
            f'    dur="14.2s" begin="3.2s" repeatCount="indefinite" />\n'
            f'  <animate attributeName="opacity"\n'
            f'    values="1; 1; 0; 0; 0; 0; 0; 0; 1"\n'
            f'    keyTimes="0; 0.211; 0.303; 0.444; 0.535; 0.676; 0.768; 0.908; 1"\n'
            f'    dur="14.2s" begin="3.2s" repeatCount="indefinite" />\n'
            f'  <path d="{path_d}" fill="{palette["portrait_dots"]}" shape-rendering="crispEdges" />\n'
            f'</g>'
        )

    # 3. Travellers Layer (~900 dots morphing between 3 logos)
    l1_pts, l2_pts, l3_pts = generate_morph_logos(num_dots=900)
    traveller_elements = []
    
    batch_size = 30
    num_batches = len(l1_pts) // batch_size
    for bi in range(num_batches):
        idx_start = bi * batch_size
        idx_end = idx_start + batch_size
        
        p1_d = encode_dot_runs(l1_pts[idx_start:idx_end])
        p2_d = encode_dot_runs(l2_pts[idx_start:idx_end])
        p3_d = encode_dot_runs(l3_pts[idx_start:idx_end])

        traveller_elements.append(
            f'<g>\n'
            f'  <animate attributeName="opacity"\n'
            f'    values="0; 0; 1; 1; 1; 1; 1; 1; 0"\n'
            f'    keyTimes="0; 0.211; 0.303; 0.444; 0.535; 0.676; 0.768; 0.908; 1"\n'
            f'    dur="14.2s" begin="3.2s" repeatCount="indefinite" />\n'
            f'  <path fill="{palette["traveller_dots"]}" shape-rendering="crispEdges">\n'
            f'    <animate attributeName="d"\n'
            f'      values="{p1_d}; {p1_d}; {p1_d}; {p1_d}; {p2_d}; {p2_d}; {p3_d}; {p3_d}; {p1_d}"\n'
            f'      keyTimes="0; 0.211; 0.303; 0.444; 0.535; 0.676; 0.768; 0.908; 1"\n'
            f'      dur="14.2s" begin="3.2s" repeatCount="indefinite" />\n'
            f'  </path>\n'
            f'</g>'
        )

    system_info_svg = build_system_info_svg(palette)

    svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {CANVAS_WIDTH} {CANVAS_HEIGHT}" width="{CANVAS_WIDTH}" height="{CANVAS_HEIGHT}">
  <defs>
    <style>
      @keyframes pulse {{
        0%, 100% {{ opacity: 1; transform: scale(1); }}
        50% {{ opacity: 0.3; transform: scale(0.85); }}
      }}
      .pulse-dot {{
        transform-origin: 1004px 89px;
        animation: pulse 1.8s ease-in-out infinite;
      }}
    </style>
    <linearGradient id="headerGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="{palette["header_bar"]}" stop-opacity="0.95" />
      <stop offset="100%" stop-color="{palette["card_bg"]}" stop-opacity="0.95" />
    </linearGradient>
    <clipPath id="portraitClip">
      <rect x="{PORTRAIT_X - 10}" y="{PORTRAIT_Y - 10}" width="{PORTRAIT_W + 20}" height="{PORTRAIT_H + 20}" rx="6" />
    </clipPath>
  </defs>

  <!-- Main Background Canvas -->
  <rect width="{CANVAS_WIDTH}" height="{CANVAS_HEIGHT}" rx="12" fill="{palette["bg"]}" stroke="{palette["card_border"]}" stroke-width="1.5" />

  <!-- Terminal Window Title Bar -->
  <path d="M 0 12 Q 0 0 12 0 L {CANVAS_WIDTH - 12} 0 Q {CANVAS_WIDTH} 0 {CANVAS_WIDTH} 12 L {CANVAS_WIDTH} 42 L 0 42 Z" fill="url(#headerGrad)" stroke="{palette["card_border"]}" stroke-width="1" />
  
  <!-- Terminal Control Buttons -->
  <circle cx="26" cy="21" r="6" fill="#EF4444" opacity="0.9" />
  <circle cx="46" cy="21" r="6" fill="#F59E0B" opacity="0.9" />
  <circle cx="66" cy="21" r="6" fill="#10B981" opacity="0.9" />

  <!-- Terminal Window Title -->
  <text x="{CANVAS_WIDTH / 2}" y="26" text-anchor="middle" fill="{palette["title_text"]}" font-family="ui-monospace, SFMono-Regular, Consolas, monospace" font-size="13" font-weight="600" letter-spacing="1">profile.sh --live</text>
  <text x="{CANVAS_WIDTH - 30}" y="26" text-anchor="end" fill="{palette["ui_dim"]}" font-family="ui-monospace, SFMono-Regular, Consolas, monospace" font-size="11" font-weight="600">v4.8-RELEASE</text>

  <!-- Left Frame: VISUAL.MAP -->
  <rect x="45" y="65" width="390" height="515" rx="8" fill="{palette["card_bg"]}" stroke="{palette["card_border"]}" stroke-width="1.2" />
  <rect x="45" y="65" width="390" height="38" rx="8" fill="{palette["header_bar"]}" stroke="{palette["card_border"]}" stroke-width="1" />
  <text x="65" y="89" fill="{palette["ui_chrome"]}" font-family="ui-monospace, SFMono-Regular, Consolas, monospace" font-size="12" font-weight="700" letter-spacing="1.5">VISUAL.MAP</text>
  <text x="415" y="89" text-anchor="end" fill="{palette["accent"]}" font-family="ui-monospace, SFMono-Regular, Consolas, monospace" font-size="11" font-weight="600">ONLINE // 1-BIT DITHER</text>
  
  <!-- Portrait Inner Border Box -->
  <rect x="{PORTRAIT_X - 6}" y="{PORTRAIT_Y - 6}" width="{PORTRAIT_W + 12}" height="{PORTRAIT_H + 12}" rx="6" fill="none" stroke="{palette["card_border"]}" stroke-width="1" stroke-dasharray="4 4" />

  <!-- Intro Layer (Plays once over 3.2s) -->
  <g id="intro-layer">
    <animate attributeName="display" values="inline; none" keyTimes="0; 1" dur="3.2s" fill="freeze" />
    {"".join(intro_paths)}
  </g>

  <!-- Portrait Drift Layer (Loops with Master Cycle) -->
  <g id="drift-layer" clip-path="url(#portraitClip)">
    {"".join(drift_layer_groups)}
  </g>

  <!-- Travellers Layer (Morphing Logos) -->
  <g id="travellers-layer">
    {"".join(traveller_elements)}
  </g>

  <!-- Left Frame Bottom Meta Info -->
  <rect x="65" y="495" width="350" height="65" rx="6" fill="{palette["badge_bg"]}" opacity="0.6" stroke="{palette["card_border"]}" stroke-width="1" />
  <text x="80" y="518" fill="{palette["text_muted"]}" font-family="ui-monospace, SFMono-Regular, Consolas, monospace" font-size="11" font-weight="600">RENDER_ENGINE</text>
  <text x="395" y="518" text-anchor="end" fill="{palette["text_primary"]}" font-family="ui-monospace, SFMono-Regular, Consolas, monospace" font-size="11" font-weight="700">FLOYD-STEINBERG [1-BIT]</text>
  <text x="80" y="542" fill="{palette["text_muted"]}" font-family="ui-monospace, SFMono-Regular, Consolas, monospace" font-size="11" font-weight="600">MORPH_TOPOLOGY</text>
  <text x="395" y="542" text-anchor="end" fill="{palette["text_accent"]}" font-family="ui-monospace, SFMono-Regular, Consolas, monospace" font-size="11" font-weight="700">OPTIMAL TRANSPORT (3-STAGE)</text>

  <!-- Right Frame: SYSTEM.INFO Readout -->
  <rect x="455" y="65" width="680" height="515" rx="8" fill="{palette["card_bg"]}" stroke="{palette["card_border"]}" stroke-width="1.2" />
  <rect x="455" y="65" width="680" height="46" rx="8" fill="{palette["header_bar"]}" stroke="{palette["card_border"]}" stroke-width="1" />
  
  <text x="480" y="94" fill="{palette["ui_chrome"]}" font-family="ui-monospace, SFMono-Regular, Consolas, monospace" font-size="14" font-weight="800" letter-spacing="2">SYSTEM.INFO</text>
  
  <!-- Pulsing LIVE Badge -->
  <rect x="990" y="78" width="68" height="22" rx="11" fill="{palette["badge_bg"]}" stroke="{palette["card_border"]}" stroke-width="1" />
  <circle cx="1004" cy="89" r="4" fill="{palette["live_pulse"]}" class="pulse-dot" />
  <text x="1016" y="93" fill="{palette["live_pulse"]}" font-family="ui-monospace, SFMono-Regular, Consolas, monospace" font-size="10" font-weight="800" letter-spacing="1">LIVE</text>

  <!-- Coloured Pill with Handle -->
  <rect x="1068" y="78" width="52" height="22" rx="11" fill="{palette["pill_bg"]}" stroke="{palette["pill_border"]}" stroke-width="1" />
  <text x="1094" y="93" text-anchor="middle" fill="{palette["pill_text"]}" font-family="ui-monospace, SFMono-Regular, Consolas, monospace" font-size="11" font-weight="700">DDW-X</text>

  <!-- System Info Dotted Leader Rows -->
  <g id="system-info-rows">
    {system_info_svg}
  </g>

  <!-- Bottom Console Status Ribbon -->
  <rect x="480" y="522" width="630" height="38" rx="6" fill="{palette["badge_bg"]}" opacity="0.6" stroke="{palette["card_border"]}" stroke-width="1" />
  <circle cx="500" cy="541" r="3.5" fill="{palette["accent"]}" />
  <text x="515" y="545" fill="{palette["text_muted"]}" font-family="ui-monospace, SFMono-Regular, Consolas, monospace" font-size="11" font-weight="600">KERNEL_STATE:</text>
  <text x="610" y="545" fill="{palette["accent"]}" font-family="ui-monospace, SFMono-Regular, Consolas, monospace" font-size="11" font-weight="700">RING-0 HYPERVISOR ACTIVE</text>
  <text x="1095" y="545" text-anchor="end" fill="{palette["ui_chrome"]}" font-family="ui-monospace, SFMono-Regular, Consolas, monospace" font-size="11" font-weight="700">ENCRYPTED // SHIRAZ_NODE</text>

</svg>'''

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(svg_content)

    print(f"Successfully generated {output_filename} ({len(svg_content) / 1024:.1f} KB)")


if __name__ == "__main__":
    generate_banner_svg(is_dark=True, output_filename="dark.svg")
    generate_banner_svg(is_dark=False, output_filename="light.svg")
