import argparse
import math
import os
import sys
import time
import cv2
import numpy as np
import pygame

# Enable OpenCV OpenCL GPU acceleration for fast bloom & compositing
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
cv2.ocl.setUseOpenCL(True)
cv2.setUseOptimized(True)

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

# ==============================================================================
# QUALITY TIERS (LOW-END TO HIGH-END HARDWARE OPTIMIZATION)
# ==============================================================================
QUALITY_PROFILES = {
    "low": {
        "particles": 28000,
        "stars": 260,
        "constellation_nodes": 85,
        "bloom_passes": 1,
        "fps": 30,
        "preview_scale": 0.42,
        "crf_quality": 7
    },
    "medium": {
        "particles": 65000,
        "stars": 420,
        "constellation_nodes": 130,
        "bloom_passes": 2,
        "fps": 60,
        "preview_scale": 0.46,
        "crf_quality": 8
    },
    "high": {
        "particles": 120000,
        "stars": 650,
        "constellation_nodes": 175,
        "bloom_passes": 3,
        "fps": 60,
        "preview_scale": 0.46,
        "crf_quality": 9
    }
}

PRESET_LIST = ["galaxy", "vortex", "wave", "explosion", "constellation"]

# ==============================================================================
# PROCEDURAL 3D TARGET GENERATORS FOR ALL 5 PRESETS
# ==============================================================================
def generate_preset_targets(preset, n, rng):
    """
    Returns vectorized target coordinates (tx, ty, tz) and per-particle
    RGB energy weights (cr, cg, cb) matching the cosmic blue/violet/magenta palette.
    """
    u = rng.uniform(0.0, 1.0, n).astype(np.float32)
    v = rng.uniform(0.0, 1.0, n).astype(np.float32)
    angle = rng.uniform(0.0, 2.0 * np.pi, n).astype(np.float32)

    if preset == "galaxy":
        # 4-arm Logarithmic Spiral Galaxy (matches deep-blue outer arms & magenta-white core)
        num_arms = 4
        arm_idx = rng.integers(0, num_arms, n).astype(np.float32)
        r = (u ** 1.35) * 620.0 + 14.0
        r_norm = np.clip(r / 634.0, 0.0, 1.0)

        # Logarithmic spiral winding + organic dust lane scatter
        spiral_theta = (arm_idx / num_arms) * 2.0 * np.pi + np.log(r * 0.045 + 1.0) * 3.8
        scatter = rng.normal(0.0, 1.0, n).astype(np.float32) * (0.22 + 0.55 * r_norm)
        theta = spiral_theta + scatter

        tx = r * np.cos(theta)
        ty = r * np.sin(theta)
        # Bulging galactic core + thin outer spiral disk
        disk_thickness = 12.0 + 95.0 * np.exp(-r_norm * 4.5)
        tz = rng.normal(0.0, 1.0, n).astype(np.float32) * disk_thickness

        # Palette: Warm Pink-White Core -> Vibrant Violet/Magenta Mid -> Electric Cyan/Blue Outer
        core_w = np.exp(-r_norm * 3.6)
        mid_w = np.exp(-((r_norm - 0.36) / 0.24) ** 2)
        outer_w = np.clip((r_norm - 0.32) / 0.68, 0.0, 1.0)

        cr = (core_w * 1.95 + mid_w * 1.35 + outer_w * 0.16).astype(np.float32)
        cg = (core_w * 0.95 + mid_w * 0.26 + outer_w * 0.68).astype(np.float32)
        cb = (core_w * 1.35 + mid_w * 1.55 + outer_w * 1.75).astype(np.float32)

    elif preset == "vortex":
        # Deep 3D Wormhole / Cosmic Funnel Vortex
        r = (u ** 0.85) * 520.0 + 45.0
        r_norm = np.clip((r - 45.0) / 520.0, 0.0, 1.0)
        theta = angle + (1.0 - r_norm) * 14.0
        tx = r * np.cos(theta)
        ty = r * np.sin(theta)
        tz = -420.0 * np.exp(-r_norm * 2.8) + rng.normal(0.0, 14.0, n).astype(np.float32)

        cr = (0.15 + 1.45 * (1.0 - r_norm) ** 2).astype(np.float32)
        cg = (0.85 * r_norm + 0.55 * (1.0 - r_norm)).astype(np.float32)
        cb = (1.65 - 0.3 * r_norm).astype(np.float32)

    elif preset == "wave":
        # 3D Quantum Interference Ocean Wave
        grid_x = (u - 0.5) * 1150.0
        grid_z = (v - 0.5) * 1150.0
        dist = np.sqrt(grid_x ** 2 + grid_z ** 2)
        wave_y = np.sin(dist * 0.022) * 125.0 + np.cos(grid_x * 0.014) * np.sin(grid_z * 0.014) * 75.0
        tx = grid_x
        ty = wave_y + rng.normal(0.0, 6.0, n).astype(np.float32)
        tz = grid_z

        crest = np.clip((wave_y + 180.0) / 360.0, 0.0, 1.0)
        cr = (0.12 + 1.55 * (crest ** 2.0)).astype(np.float32)
        cg = (0.45 + 0.95 * crest).astype(np.float32)
        cb = (1.65 - 0.35 * crest).astype(np.float32)

    elif preset == "explosion":
        # Multi-Shell Supernova Nebula Remnant
        phi = np.arccos(2.0 * v - 1.0)
        shell_r = (160.0 + 380.0 * (u ** 0.55)) * (1.0 + 0.22 * np.sin(phi * 6.0) * np.cos(angle * 5.0))
        r_norm = np.clip((shell_r - 160.0) / 420.0, 0.0, 1.0)
        tx = shell_r * np.sin(phi) * np.cos(angle)
        ty = shell_r * np.sin(phi) * np.sin(angle)
        tz = shell_r * np.cos(phi)

        cr = (1.85 * (1.0 - r_norm) + 0.95 * r_norm).astype(np.float32)
        cg = (1.15 * (1.0 - r_norm) ** 1.5 + 0.22 * r_norm).astype(np.float32)
        cb = (0.55 * (1.0 - r_norm) + 1.65 * r_norm).astype(np.float32)

    else:  # "constellation"
        # Sacred Geodesic Star-Cluster Web (clustered around 3D attractors)
        clusters = 16
        c_phi = np.arccos(1.0 - 2.0 * (np.arange(clusters) + 0.5) / clusters)
        c_theta = np.pi * (1.0 + 5.0 ** 0.5) * np.arange(clusters)
        c_r = 340.0
        cx = c_r * np.sin(c_phi) * np.cos(c_theta)
        cy = c_r * np.sin(c_phi) * np.sin(c_theta)
        cz = c_r * np.cos(c_phi)

        cid = rng.integers(0, clusters, n)
        spread = rng.exponential(52.0, n).astype(np.float32)
        s_phi = np.arccos(2.0 * v - 1.0)
        tx = (cx[cid] + spread * np.sin(s_phi) * np.cos(angle)).astype(np.float32)
        ty = (cy[cid] + spread * np.sin(s_phi) * np.sin(angle)).astype(np.float32)
        tz = (cz[cid] + spread * np.cos(s_phi)).astype(np.float32)

        core_c = np.exp(-spread / 45.0)
        cr = (0.35 + 1.45 * core_c).astype(np.float32)
        cg = (0.75 + 0.85 * core_c).astype(np.float32)
        cb = (1.75 + 0.25 * core_c).astype(np.float32)

    # Random sparkling bright stars scattered across the universe
    sparkle = (rng.uniform(0.0, 1.0, n) < 0.045).astype(np.float32)
    cr += sparkle * 1.4
    cg += sparkle * 1.3
    cb += sparkle * 1.5

    return tx, ty, tz, cr, cg, cb

# ==============================================================================
# 6-ACT TIMELINE & LOOP CHOREOGRAPHY
# ==============================================================================
def get_act_parameters(p, loop_mode=True):
    """
    Maps normalized time p in [0, 1] to the 6-act universe choreography:
      Act 1 (0.00 - 0.14): Appear from darkness & drift
      Act 2 (0.14 - 0.34): Swirling flow & gravitational attraction
      Act 3 (0.34 - 0.56): Peak formation (Galaxy / Vortex / Wave) + cinematic zoom
      Act 4 (0.56 - 0.72): Supernova repulsion explosion
      Act 5 (0.72 - 0.90): Magnetic reorganization into luminous constellation/core
      Act 6 (0.90 - 1.00): Smooth ending / seamless loop transition
    """
    # Brightness fade-in / fade-out for seamless looping
    fade_in = min(1.0, p / 0.12)
    fade_out = min(1.0, (1.0 - p) / 0.10) if loop_mode else 1.0
    master_brightness = fade_in * fade_out

    # Gravitational attraction progress (0 = scattered void, 1 = formed preset)
    if p < 0.12:
        attract = 0.05 * (p / 0.12)
    elif p < 0.38:
        k = (p - 0.12) / 0.26
        attract = 0.05 + 0.95 * (k * k * (3.0 - 2.0 * k))
    else:
        attract = 1.0

    # Supernova explosion repulsion wave (Act 4)
    explode = 0.0
    if 0.56 <= p < 0.65:
        k = (p - 0.56) / 0.09
        explode = 1.0 - (1.0 - k) ** 3
    elif 0.65 <= p < 0.84:
        k = (p - 0.65) / 0.19
        explode = 1.0 - (k * k * (3.0 - 2.0 * k))

    # Constellation network glow visibility (peeks during Act 5 reorganization)
    constellation_alpha = 0.0
    if 0.68 <= p <= 0.96:
        constellation_alpha = math.sin(((p - 0.68) / 0.28) * math.pi)

    # Cinematic camera zoom & tilt
    cam_dist = 1480.0 - 460.0 * math.sin(min(1.0, p / 0.58) * math.pi * 0.5)
    cam_dist += explode * 220.0
    if loop_mode and p > 0.86:
        k = (p - 0.86) / 0.14
        ease = k * k * (3.0 - 2.0 * k)
        cam_dist = cam_dist * (1.0 - ease) + 1480.0 * ease

    # 3D tilt angle (tilts galaxy diagonally like the reference image)
    pitch = math.radians(48.0 - 18.0 * math.sin(p * math.pi * 2.0))
    roll = math.radians(-18.0 + 12.0 * math.cos(p * math.pi * 2.0))
    yaw = p * math.pi * 2.0  # Full 360-degree seamless rotation over duration

    return master_brightness, attract, explode, constellation_alpha, cam_dist, pitch, yaw, roll

# ==============================================================================
# MAIN RENDER & EXPORT PIPELINE
# ==============================================================================
def run_generator(args):
    W, H = 1080, 1920
    profile = QUALITY_PROFILES[args.quality]
    N = args.particles if args.particles else profile["particles"]
    FPS = args.fps if args.fps else profile["fps"]
    DURATION = args.duration
    TOTAL_FRAMES = int(FPS * DURATION)
    PW, PH = int(W * profile["preview_scale"]), int(H * profile["preview_scale"])

    # Initialize Pygame live preview window
    pygame.init()
    preview_win = pygame.display.set_mode((PW, PH), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption(f"Particle Universe [{args.preset.upper()} | {args.quality.upper()} | 1080x1920]")

    rng = np.random.default_rng(2026)

    # Generate initial dark void positions & explosion velocity vectors
    void_x = rng.uniform(-900.0, 900.0, N).astype(np.float32)
    void_y = rng.uniform(-1200.0, 1200.0, N).astype(np.float32)
    void_z = rng.uniform(-900.0, 900.0, N).astype(np.float32)

    exp_dir = rng.normal(0.0, 1.0, (N, 3)).astype(np.float32)
    exp_dir /= (np.linalg.norm(exp_dir, axis=1, keepdims=True) + 1e-6)
    exp_speed = rng.uniform(260.0, 820.0, N).astype(np.float32)
    p_phase = rng.uniform(0.0, 2.0 * np.pi, N).astype(np.float32)
    orbital_speed = rng.uniform(0.65, 1.55, N).astype(np.float32)

    # Load primary preset targets
    current_preset = args.preset
    tx, ty, tz, cr, cg, cb = generate_preset_targets(current_preset, N, rng)

    # Deep space background stars & ambient nebula vignette (RGB)
    bg_base = np.zeros((H, W, 3), dtype=np.uint8)
    y_g, x_g = np.mgrid[0:H, 0:W].astype(np.float32)
    dist_bg = np.sqrt(((x_g - W * 0.5) / (W * 0.65)) ** 2 + ((y_g - H * 0.48) / (H * 0.55)) ** 2)
    neb_glow = np.exp(-(dist_bg ** 2) * 1.8)
    bg_base[:, :, 0] = np.clip(neb_glow * 14.0, 0, 255).astype(np.uint8)
    bg_base[:, :, 1] = np.clip(neb_glow * 10.0, 0, 255).astype(np.uint8)
    bg_base[:, :, 2] = np.clip(neb_glow * 36.0, 0, 255).astype(np.uint8)

    # Populate background bokeh stars
    n_stars = profile["stars"]
    star_x = rng.integers(5, W - 5, n_stars)
    star_y = rng.integers(5, H - 5, n_stars)
    star_rad = rng.choice([1, 2, 3], n_stars, p=[0.72, 0.22, 0.06])
    star_cols = [(220, 240, 255), (140, 195, 255), (255, 165, 230), (255, 120, 170)]
    for i in range(n_stars):
        c = star_cols[i % len(star_cols)]
        cv2.circle(bg_base, (int(star_x[i]), int(star_y[i])), int(star_rad[i]), c, -1, cv2.LINE_AA)

    # Select subset of indices for glowing constellation connections
    n_nodes = min(profile["constellation_nodes"], N)
    node_ids = rng.choice(N, n_nodes, replace=False)

    # Setup H.264 MP4 Writer
    writer = None
    out_file = args.output if args.output else f"universe_{current_preset}_{args.quality}.mp4"
    if args.export:
        if HAS_IMAGEIO:
            writer = imageio.get_writer(
                out_file,
                fps=FPS,
                codec="libx264",
                quality=profile["crf_quality"],
                pixelformat="yuv420p",
                macro_block_size=1
            )
            print(f"[EXPORT] Rendering {W}x{H} @ {FPS} FPS ({TOTAL_FRAMES} frames) -> '{out_file}'")
        else:
            print("[WARN] imageio-ffmpeg not found. Falling back to OpenCV mp4v writer.")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_file, fourcc, FPS, (W, H))

    hdr_buffer = np.zeros((H, W, 3), dtype=np.float32)
    fov = 1050.0
    frame_idx = 0
    running = True
    t_start = time.time()

    print(f"[ACTIVE] Preset: {current_preset.upper()} | Particles: {N:,} | Quality: {args.quality.upper()}")
    if not args.export:
        print("[CONTROLS] Keys 1-5: Switch Presets Live | ESC: Quit")

    while running:
        if args.export and frame_idx >= TOTAL_FRAMES:
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif not args.export and event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5):
                    idx_map = {pygame.K_1: 0, pygame.K_2: 1, pygame.K_3: 2, pygame.K_4: 3, pygame.K_5: 4}
                    current_preset = PRESET_LIST[idx_map[event.key]]
                    tx, ty, tz, cr, cg, cb = generate_preset_targets(current_preset, N, rng)
                    pygame.display.set_caption(f"Particle Universe [{current_preset.upper()} | LIVE PREVIEW]")
                    print(f"[PRESET] Switched to: {current_preset.upper()}")

        p = (frame_idx % TOTAL_FRAMES) / float(TOTAL_FRAMES)
        t_sec = p * DURATION

        brightness, attract, explode, const_alpha, cam_dist, pitch, yaw, roll = get_act_parameters(p, args.loop)

        # ------------------------------------------------------------------
        # 1. VECTORIZED ATTRACTION, DIFFERENTIAL SWIRL & REPULSION PHYSICS
        # ------------------------------------------------------------------
        # Differential circular orbital motion (inner core rotates faster than outer arms)
        rad_xy = np.sqrt(tx * tx + ty * ty) + 1.0
        spin_angle = (t_sec * 1.15 * orbital_speed) / (1.0 + rad_xy * 0.0022)
        cos_sp, sin_sp = np.cos(spin_angle), np.sin(spin_angle)

        rot_tx = tx * cos_sp - ty * sin_sp
        rot_ty = tx * sin_sp + ty * cos_sp
        rot_tz = tz + np.sin(rad_xy * 0.02 - t_sec * 3.2) * (18.0 if current_preset == "wave" else 7.0)

        # Swirling attraction from void into formation
        swirl = (1.0 - attract) * 2.8
        c_sw, s_sw = math.cos(swirl), math.sin(swirl)
        vx_s = void_x * c_sw - void_y * s_sw
        vy_s = void_x * s_sw + void_y * c_sw

        px = (1.0 - attract) * vx_s + attract * rot_tx + exp_dir[:, 0] * exp_speed * explode
        py = (1.0 - attract) * vy_s + attract * rot_ty + exp_dir[:, 1] * exp_speed * explode
        pz = (1.0 - attract) * void_z + attract * rot_tz + exp_dir[:, 2] * exp_speed * explode

        # ------------------------------------------------------------------
        # 2. 3D CAMERA ROTATION & PERSPECTIVE PROJECTION
        # ------------------------------------------------------------------
        cy, sy = math.cos(yaw), math.sin(yaw)
        cx, sx = math.cos(pitch), math.sin(pitch)
        cz, sz = math.cos(roll), math.sin(roll)

        # Yaw -> Pitch -> Roll
        x1 = px * cy - pz * sy
        z1 = pz * cy + px * sy
        y2 = py * cx - z1 * sx
        z2 = z1 * cx + py * sx
        x3 = x1 * cz - y2 * sz
        y3 = y2 * cz + x1 * sz

        depth = cam_dist - z2
        valid_z = depth > 80.0
        scale = np.zeros_like(depth)
        scale[valid_z] = fov / depth[valid_z]

        scr_x = (W * 0.5 + x3 * scale).astype(np.int32)
        scr_y = (H * 0.48 + y3 * scale).astype(np.int32)

        # ------------------------------------------------------------------
        # 3. HDR MOTION-TRAIL PERSISTENCE & MICRO-PARTICLE ACCUMULATION
        # ------------------------------------------------------------------
        # Higher persistence during explosion creates dramatic streaking light trails
        trail_decay = 0.56 if explode > 0.1 else 0.42
        hdr_buffer *= trail_decay

        in_bounds = valid_z & (scr_x >= 1) & (scr_x < W - 1) & (scr_y >= 1) & (scr_y < H - 1)
        vx = scr_x[in_bounds]
        vy = scr_y[in_bounds]

        twinkle = 0.78 + 0.22 * np.sin(t_sec * 5.0 + p_phase[in_bounds])
        depth_gain = np.clip(scale[in_bounds] * 1.15, 0.25, 1.85) * brightness * twinkle

        np.add.at(hdr_buffer[:, :, 0], (vy, vx), cr[in_bounds] * depth_gain)
        np.add.at(hdr_buffer[:, :, 1], (vy, vx), cg[in_bounds] * depth_gain)
        np.add.at(hdr_buffer[:, :, 2], (vy, vx), cb[in_bounds] * depth_gain)

        # Tone-map HDR energy: dense galactic arms saturate cleanly into glowing white-pink
        lum = (hdr_buffer[:, :, 0] * 0.4 + hdr_buffer[:, :, 1] * 0.3 + hdr_buffer[:, :, 2] * 0.3)
        white_core = np.clip(lum - 2.4, 0.0, 26.0) * 0.48
        particle_rgb = np.clip((hdr_buffer + white_core[:, :, None]) * 44.0, 0, 255).astype(np.uint8)

        # ------------------------------------------------------------------
        # 4. GLOWING CONSTELLATION CONNECTIONS (VECTORIZED NEAREST PAIRS)
        # ------------------------------------------------------------------
        conn_strength = 0.95 if current_preset == "constellation" else const_alpha
        if conn_strength > 0.05:
            nx = scr_x[node_ids]
            ny = scr_y[node_ids]
            n_ok = in_bounds[node_ids]
            nx, ny = nx[n_ok], ny[n_ok]

            if len(nx) > 4:
                dx = nx[:, None] - nx[None, :]
                dy = ny[:, None] - ny[None, :]
                dist_sq = dx * dx + dy * dy
                max_dist = 155.0 if current_preset == "constellation" else 135.0
                # Upper triangle pairs within threshold distance
                i_idx, j_idx = np.where((dist_sq > 144.0) & (dist_sq < max_dist * max_dist))
                mask_ut = i_idx < j_idx
                for a, b in zip(i_idx[mask_ut][:240], j_idx[mask_ut][:240]):
                    alpha_l = (1.0 - math.sqrt(dist_sq[a, b]) / max_dist) * conn_strength
                    col_l = (int(120 * alpha_l), int(210 * alpha_l), int(255 * alpha_l))
                    cv2.line(particle_rgb, (int(nx[a]), int(ny[a])), (int(nx[b]), int(ny[b])), col_l, 1, cv2.LINE_AA)

        # ------------------------------------------------------------------
        # 5. OPENCV GPU MULTI-PASS OPTICAL BLOOM & FINAL COMPOSITING
        # ------------------------------------------------------------------
        frame_rgb = cv2.add(bg_base, particle_rgb)

        # Pass 1: Tight crisp neon glow
        b1 = cv2.resize(particle_rgb, (W // 4, H // 4), interpolation=cv2.INTER_LINEAR)
        b1 = cv2.GaussianBlur(b1, (9, 9), 0)
        b1 = cv2.resize(b1, (W, H), interpolation=cv2.INTER_LINEAR)
        cv2.addWeighted(frame_rgb, 1.0, b1, 1.15, 0, dst=frame_rgb)

        # Pass 2: Wide cosmic nebula bloom (Medium & High quality)
        if profile["bloom_passes"] >= 2:
            b2 = cv2.resize(particle_rgb, (W // 10, H // 10), interpolation=cv2.INTER_LINEAR)
            b2 = cv2.GaussianBlur(b2, (15, 15), 0)
            b2 = cv2.resize(b2, (W, H), interpolation=cv2.INTER_LINEAR)
            cv2.addWeighted(frame_rgb, 1.0, b2, 0.85, 0, dst=frame_rgb)

        # Pass 3: Ultra-soft atmospheric halo (High quality)
        if profile["bloom_passes"] >= 3:
            b3 = cv2.resize(particle_rgb, (W // 20, H // 20), interpolation=cv2.INTER_LINEAR)
            b3 = cv2.GaussianBlur(b3, (17, 17), 0)
            b3 = cv2.resize(b3, (W, H), interpolation=cv2.INTER_LINEAR)
            cv2.addWeighted(frame_rgb, 1.0, b3, 0.55, 0, dst=frame_rgb)

        # ------------------------------------------------------------------
        # 6. LIVE PREVIEW & H.264 MP4 FRAME WRITE
        # ------------------------------------------------------------------
        preview_rgb = cv2.resize(frame_rgb, (PW, PH), interpolation=cv2.INTER_AREA)
        pygame.surfarray.blit_array(preview_win, preview_rgb.transpose([1, 0, 2]))
        pygame.display.flip()

        if writer is not None:
            if HAS_IMAGEIO:
                writer.append_data(frame_rgb)
            else:
                writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

        frame_idx += 1
        if args.export and frame_idx % FPS == 0:
            pct = int((frame_idx / TOTAL_FRAMES) * 100)
            elapsed = time.time() - t_start
            print(f"  -> Progress: {frame_idx}/{TOTAL_FRAMES} frames ({pct}%) | Elapsed: {elapsed:.1f}s")

    if writer is not None:
        if HAS_IMAGEIO:
            writer.close()
        else:
            writer.release()
        print(f"[SUCCESS] Saved 1080x1920 H.264 Reel to '{out_file}'!")

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cinematic Python Particle Universe Reel Generator (1080x1920)")
    parser.add_argument("--preset", type=str, default="galaxy", choices=PRESET_LIST,
                        help="Visual preset: galaxy, vortex, wave, explosion, constellation")
    parser.add_argument("--quality", type=str, default="medium", choices=["low", "medium", "high"],
                        help="Hardware quality profile: low (fast), medium (balanced), high (cinema)")
    parser.add_argument("--duration", type=float, default=12.0, help="Video duration in seconds")
    parser.add_argument("--fps", type=int, default=None, help="Override target FPS (e.g., 30 or 60)")
    parser.add_argument("--particles", type=int, default=None, help="Override particle count")
    parser.add_argument("--loop", action="store_true", help="Enable smooth seamless loop fade & camera return")
    parser.add_argument("--export", action="store_true", help="Export 1080x1920 H.264 MP4 video file")
    parser.add_argument("--output", type=str, default="", help="Custom output MP4 filename")
    run_generator(parser.parse_args())