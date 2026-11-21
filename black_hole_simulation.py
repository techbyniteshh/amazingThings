import pygame
import numpy as np
import math
import wave
import struct
import os
import subprocess
import sys

# Optional H.264 MP4 & FFmpeg audio muxing support
try:
    import imageio
    import imageio_ffmpeg
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

# ==============================================================================
# CONFIGURATION (1080x1920 VERTICAL REEL @ 60 FPS)
# ==============================================================================
CONFIG = {
    "WIDTH": 1080,
    "HEIGHT": 1920,
    "PREVIEW_SCALE": 0.48,              # Scaled preview window so it fits laptop screens
    "FPS": 60,
    "DURATION": 12.0,                   # Total animation duration in seconds
    "DISK_PARTICLES": 85000,            # Vectorized plasma gas particles in accretion disk
    "STAR_COUNT": 1400,                 # Deep-space background stars
    "EXPORT_MP4": True,                 # True = Render 1080x1920 H.264 MP4 video
    "GENERATE_AUDIO": True,             # True = Synthesize deep cosmic drone & merge into MP4
    "OUTPUT_MP4": "black_hole_simulation.mp4",
    "SHOW_LABEL": True,
    "LABEL_TEXT": "ARTISTIC GRAVITATIONAL LENSING SIMULATION // PYTHON"
}

W, H = CONFIG["WIDTH"], CONFIG["HEIGHT"]
PW, PH = int(W * CONFIG["PREVIEW_SCALE"]), int(H * CONFIG["PREVIEW_SCALE"])
CX, CY = W // 2, int(H * 0.49)

pygame.init()
preview_screen = pygame.display.set_mode((PW, PH), pygame.HWSURFACE | pygame.DOUBLEBUF)
pygame.display.set_caption("Cinematic Black Hole Simulation Generator (1080x1920 @ 60FPS)")
clock = pygame.time.Clock()

master_surf = pygame.Surface((W, H))
hdr_surf = pygame.Surface((W, H))
font_label = pygame.font.SysFont("consolas", 20, bold=True)

# ==============================================================================
# 1. PROCEDURAL SPACE DRONE AUDIO SYNTHESIZER (OPTIONAL SOUND EFFECTS)
# ==============================================================================
def generate_cosmic_drone_wav(filename, duration, sample_rate=44100):
    """Generates a deep cinematic sub-bass gravitational drone & shimmer WAV file."""
    print("[AUDIO] Synthesizing deep-space gravitational audio track...")
    total_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, total_samples, dtype=np.float32)

    # Sub-bass 42Hz + 63Hz fifth + slow LFO modulation matching camera approach
    envelope = np.clip(t / 2.0, 0.0, 1.0) * np.clip((duration - t) / 1.5, 0.0, 1.0)
    intensity_swell = 0.6 + 0.4 * np.sin(np.pi * t / duration)

    sub_bass = np.sin(2.0 * np.pi * 42.0 * t) * 0.45
    harmonic_1 = np.sin(2.0 * np.pi * 63.0 * t + np.sin(t * 1.2) * 0.5) * 0.25
    harmonic_2 = np.sin(2.0 * np.pi * 126.0 * t + np.cos(t * 0.8) * 1.2) * 0.12

    # Subtle filtered cosmic wind noise
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.04, total_samples).astype(np.float32)
    kernel = np.ones(120, dtype=np.float32) / 120.0
    wind = np.convolve(noise, kernel, mode='same') * 3.0

    signal = (sub_bass + harmonic_1 + harmonic_2 + wind) * envelope * intensity_swell
    signal = np.clip(signal * 32767.0, -32767, 32767).astype(np.int16)

    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(signal.tobytes())

# ==============================================================================
# 2. PRE-COMPUTED DEEP SPACE STARFIELD, AMBER DUST & BLUE GALAXY NEBULA
# ==============================================================================
def build_deep_space_background():
    rng = np.random.default_rng(108)
    bg_arr = np.zeros((W, H, 3), dtype=np.float32)

    y_grid, x_grid = np.mgrid[0:H, 0:W].astype(np.float32)

    # Warm amber/brownish cosmic dust glow around the center (matches reference image)
    dist_center = np.sqrt((x_grid - CX) ** 2 + (y_grid - CY) ** 2) / (W * 0.68)
    amber_halo = np.exp(-(dist_center ** 1.8) * 2.2)
    bg_arr[:, :, 0] += amber_halo.T * 42.0
    bg_arr[:, :, 1] += amber_halo.T * 18.0
    bg_arr[:, :, 2] += amber_halo.T * 5.0

    # Distant Blue Galaxy Nebula in Bottom-Right quadrant (matches reference image)
    gx, gy = W * 0.92, H * 0.75
    g_dist = np.sqrt(((x_grid - gx) * 0.9) ** 2 + ((y_grid - gy) * 1.2) ** 2) / 135.0
    blue_neb = np.exp(-(g_dist ** 2) * 1.4)
    bg_arr[:, :, 0] += blue_neb.T * 38.0
    bg_arr[:, :, 1] += blue_neb.T * 85.0
    bg_arr[:, :, 2] += blue_neb.T * 165.0

    # Starfield coordinates, colors (warm white, orange, soft pink, blue) & sizes
    n_stars = CONFIG["STAR_COUNT"]
    sx = rng.integers(4, W - 4, n_stars)
    sy = rng.integers(4, H - 4, n_stars)
    s_size = rng.choice([1, 2, 3], n_stars, p=[0.74, 0.21, 0.05])
    s_palette = [
        (255, 248, 235), (255, 200, 150), (255, 160, 175),
        (170, 210, 255), (255, 225, 190)
    ]

    bg_surf = pygame.Surface((W, H))
    pygame.surfarray.blit_array(bg_surf, np.clip(bg_arr, 0, 255).astype(np.uint8))

    for i in range(n_stars):
        col = s_palette[i % len(s_palette)]
        brightness = rng.uniform(0.35, 1.0)
        c = (int(col[0] * brightness), int(col[1] * brightness), int(col[2] * brightness))
        if s_size[i] == 1:
            bg_surf.set_at((int(sx[i]), int(sy[i])), c)
        else:
            pygame.draw.circle(bg_surf, c, (int(sx[i]), int(sy[i])), int(s_size[i]))
            if s_size[i] == 3:
                # Soft halo around bright foreground stars
                halo_c = (c[0] // 3, c[1] // 3, c[2] // 3)
                pygame.draw.circle(bg_surf, halo_c, (int(sx[i]), int(sy[i])), 6)

    return bg_surf

space_bg_surf = build_deep_space_background()

# ==============================================================================
# 3. KEPLERIAN ACCRETION DISK & GRAVITATIONAL LENSING INITIALIZATION
# ==============================================================================
N = CONFIG["DISK_PARTICLES"]
rng = np.random.default_rng(2026)

# Event Horizon & Photon Sphere Radii (in pixels at reference camera distance)
R_SHADOW = 122.0      # Black hole shadow radius
R_ISCO = 136.0        # Innermost Stable Circular Orbit
R_OUTER = 640.0       # Extended tapering outer plasma wings

# Radial distribution: dense near the hot inner core, tapering out to wide wings
u_rad = rng.uniform(0.0, 1.0, N).astype(np.float32)
r_orbit = (R_ISCO + (R_OUTER - R_ISCO) * (u_rad ** 1.75)).astype(np.float32)

# Normalized radial coordinate [0 = hot inner ISCO, 1 = cool outer tip]
r_norm = (r_orbit - R_ISCO) / (R_OUTER - R_ISCO)

# Initial azimuthal angle around the black hole
theta_0 = rng.uniform(0.0, 2.0 * np.pi, N).astype(np.float32)

# Differential Keplerian angular velocity: omega ~ r^(-1.5)
# Inner particles orbit rapidly; outer particles orbit majestically
omega = (1850.0 / (r_orbit ** 1.38)).astype(np.float32)

# Spiral filament lanes & organic plasma clumping
spiral_arm = np.sin(theta_0 * 3.0 - np.log(r_orbit) * 7.5)
vertical_jitter = rng.normal(0.0, 1.0, N).astype(np.float32) * (2.5 + 14.0 * (1.0 - r_norm) * r_norm)

# Base Plasma Temperature Palette (Inner White-Yellow -> Golden Orange -> Deep Crimson Red)
# Plus subtle high-energy blue-white hot filaments near ISCO
temp = (1.0 - r_norm) ** 1.25
is_blue_filament = (r_norm < 0.14) & (rng.uniform(0, 1, N) < 0.12)

base_r = np.where(is_blue_filament, 0.75, 0.45 + 1.55 * (temp ** 0.7)).astype(np.float32)
base_g = np.where(is_blue_filament, 1.15, 0.06 + 1.15 * (temp ** 1.55)).astype(np.float32)
base_b = np.where(is_blue_filament, 1.85, 0.01 + 0.65 * (temp ** 3.6)).astype(np.float32)

# Modulate density by spiral filaments so dark gas lanes appear like the reference photo
filament_gain = (0.62 + 0.48 * spiral_arm + rng.uniform(-0.15, 0.25, N)).astype(np.float32)
# Taper brightness at extreme outer wing tips
wing_taper = np.clip(1.0 - (r_norm ** 2.2), 0.08, 1.0)
particle_energy = np.clip(filament_gain * wing_taper, 0.05, 1.8).astype(np.float32)

# ==============================================================================
# 4. ANALYTICAL GRAVITATIONAL LENSING & 6-ACT CAMERA CHOREOGRAPHY
# ==============================================================================
def get_camera_choreography(t_sec, duration):
    """
    6-Act Camera Choreography:
    1. Approach from deep space (0 - 2.2s)
    2. Accretion disk emerges & ignites (1.0 - 3.5s)
    3. Differential orbital flow & Doppler beaming
    4. Plasma flare intensification (5.0 - 8.5s)
    5. Slow orbital roll & pitch around the black hole
    6. Final iconic hero shot matching reference image framing (8.5 - 12.0s)
    """
    p = t_sec / duration

    # Smooth approach zoom: starts far (scale 0.52), zooms to dramatic close-up (1.06), settles at 1.0
    if p < 0.22:
        k = p / 0.22
        ease = 1.0 - (1.0 - k) ** 3
        cam_scale = 0.50 + 0.48 * ease
    elif p < 0.75:
        k = (p - 0.22) / 0.53
        cam_scale = 0.98 + 0.08 * math.sin(k * math.pi)
    else:
        cam_scale = 0.98

    # Disk emergence & plasma intensification curve
    disk_ignition = min(1.0, max(0.08, (t_sec - 0.15) / 2.2))
    flare_boost = 1.0 + 0.28 * math.sin(max(0.0, (p - 0.35) / 0.65) * math.pi)

    # Diagonal roll angle: transitions smoothly to the exact -19.5 degree tilt in reference image
    roll_deg = -26.0 + 6.5 * math.sin(p * math.pi * 0.75)
    roll_rad = math.radians(roll_deg)

    # Inclination angle (nearly edge-on ~83.5 deg so the front disk bisects the shadow)
    pitch_rad = math.radians(83.2 + 2.2 * math.cos(t_sec * 0.45))

    return cam_scale, disk_ignition * flare_boost, roll_rad, pitch_rad

# ==============================================================================
# 5. MP4 STREAM WRITER SETUP
# ==============================================================================
writer = None
if CONFIG["EXPORT_MP4"] and HAS_IMAGEIO:
    writer = imageio.get_writer(
        CONFIG["OUTPUT_MP4"],
        fps=CONFIG["FPS"],
        codec="libx264",
        quality=9,
        pixelformat="yuv420p",
        macro_block_size=1
    )
    print(f"[RENDER] Exporting 1080x1920 @ {CONFIG['FPS']} FPS to '{CONFIG['OUTPUT_MP4']}'...")

# ==============================================================================
# 6. MAIN 60 FPS SIMULATION & HDR LENSING RENDER LOOP
# ==============================================================================
hdr_buffer = np.zeros((W, H, 3), dtype=np.float32)
total_frames = int(CONFIG["DURATION"] * CONFIG["FPS"])
frame_idx = 0
running = True

# Pre-compute circular angle array for the razor-thin photon ring
photon_angles = np.linspace(0, 2.0 * np.pi, 720, dtype=np.float32)

while running and frame_idx < total_frames:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

    t = frame_idx / float(CONFIG["FPS"])
    cam_scale, disk_intensity, roll_rad, pitch_rad = get_camera_choreography(t, CONFIG["DURATION"])

    # 1. Advance Keplerian Orbital Angles (inner gas spins much faster than outer wings)
    theta = theta_0 + omega * t

    # Local 3D coordinates in the disk plane
    x_local = r_orbit * np.cos(theta)
    z_local = r_orbit * np.sin(theta)   # z_local > 0 is behind the black hole; z_local < 0 is in front
    y_local = vertical_jitter

    # Project inclined disk before gravitational lensing
    cos_inc = math.cos(pitch_rad)
    sin_inc = math.sin(pitch_rad)
    x_flat = x_local
    y_flat = y_local * sin_inc - z_local * cos_inc

    # --------------------------------------------------------------------------
    # 2. GRAVITATIONAL LENSING APPROXIMATION (EINSTEIN UPPER & LOWER ARCHES)
    # --------------------------------------------------------------------------
    # Light from behind the black hole (z_local > 0) bends over the top and bottom
    # of the Schwarzschild shadow sphere, forming the iconic Gargantua double halo!
    r_2d = np.sqrt(x_flat * x_flat + y_flat * y_flat) + 1e-4
    behind_weight = np.clip(z_local / (r_orbit * 0.85), 0.0, 1.0) ** 1.15

    # Target radius for lensed light wrapping around the shadow rim
    r_lensed_top = R_SHADOW * 1.08 + (r_orbit - R_ISCO) * 0.26
    # Direction unit vector in upper hemisphere as x goes from -r to +r
    arch_angle = np.arccos(np.clip(x_flat / r_orbit, -1.0, 1.0))
    x_arch = r_lensed_top * np.cos(arch_angle)
    y_arch_top = -r_lensed_top * np.sin(arch_angle)  # Upper glowing arch

    # Blend unlensed front disk with gravitationally lensed upper arch for back-side particles
    # Also taper horizontal pinch where the flat disk meets the Einstein ring neck
    neck_pinch = 1.0 - 0.35 * behind_weight * np.exp(-((np.abs(x_flat) - R_SHADOW * 1.45) / 75.0) ** 2)
    x_warped = (1.0 - behind_weight) * x_flat + behind_weight * x_arch
    y_warped = ((1.0 - behind_weight) * y_flat + behind_weight * y_arch_top) * neck_pinch

    # Secondary Under-Side Lensed Arch (light from the underside of the back disk bending below shadow)
    under_mask = (z_local > -20.0) & (r_norm < 0.58)
    x_under_local = x_flat[under_mask]
    r_under_orbit = r_orbit[under_mask]
    r_lensed_bot = R_SHADOW * 1.07 + (r_under_orbit - R_ISCO) * 0.22
    bot_angle = np.arccos(np.clip(x_under_local / r_under_orbit, -1.0, 1.0))
    x_under = r_lensed_bot * np.cos(bot_angle)
    y_under = r_lensed_bot * np.sin(bot_angle) * 0.94  # Lower glowing arch

    # Apply camera scale and diagonal roll (-19.5 deg tilt matching reference image)
    cos_r = math.cos(roll_rad) * cam_scale
    sin_r = math.sin(roll_rad) * cam_scale

    scr_x = (CX + x_warped * cos_r - y_warped * sin_r).astype(np.int32)
    scr_y = (CY + x_warped * sin_r + y_warped * cos_r).astype(np.int32)

    scr_ux = (CX + x_under * cos_r - y_under * sin_r).astype(np.int32)
    scr_uy = (CY + x_under * sin_r + y_under * cos_r).astype(np.int32)

    # --------------------------------------------------------------------------
    # 3. RELATIVISTIC DOPPLER BEAMING & HDR TRAIL ACCUMULATION
    # --------------------------------------------------------------------------
    # Persistence buffer creates smooth glowing gas trails
    hdr_buffer *= 0.44

    # Relativistic Doppler shift: approaching side of disk is slightly brighter
    doppler = 1.0 - 0.24 * (x_local / r_orbit)
    gain = particle_energy * doppler * disk_intensity

    valid = (scr_x >= 1) & (scr_x < W - 1) & (scr_y >= 1) & (scr_y < H - 1)
    vx, vy = scr_x[valid], scr_y[valid]

    np.add.at(hdr_buffer[:, :, 0], (vx, vy), base_r[valid] * gain[valid])
    np.add.at(hdr_buffer[:, :, 1], (vx, vy), base_g[valid] * gain[valid])
    np.add.at(hdr_buffer[:, :, 2], (vx, vy), base_b[valid] * gain[valid])

    # Accumulate Lower Lensed Einstein Arch
    u_valid = (scr_ux >= 1) & (scr_ux < W - 1) & (scr_uy >= 1) & (scr_uy < H - 1)
    uvx, uvy = scr_ux[u_valid], scr_uy[u_valid]
    u_gain = (gain[under_mask][u_valid]) * 0.78

    np.add.at(hdr_buffer[:, :, 0], (uvx, uvy), base_r[under_mask][u_valid] * u_gain)
    np.add.at(hdr_buffer[:, :, 1], (uvx, uvy), base_g[under_mask][u_valid] * u_gain)
    np.add.at(hdr_buffer[:, :, 2], (uvx, uvy), base_b[under_mask][u_valid] * u_gain)

    # --------------------------------------------------------------------------
    # 4. EVENT HORIZON OCCLUSION & RAZOR-THIN PHOTON RING
    # --------------------------------------------------------------------------
    # Convert HDR energy buffer to 8-bit RGB with white-hot core saturation
    lum = (hdr_buffer[:, :, 0] * 0.45 + hdr_buffer[:, :, 1] * 0.40 + hdr_buffer[:, :, 2] * 0.15)
    white_hot = np.clip(lum - 3.2, 0.0, 28.0) * 0.48
    rgb_frame = np.clip((hdr_buffer + white_hot[:, :, None]) * 38.0, 0, 255).astype(np.uint8)
    pygame.surfarray.blit_array(hdr_surf, rgb_frame)

    # Carve out the central Black Hole Shadow (Event Horizon)
    shadow_rad = int(R_SHADOW * cam_scale)
    pygame.draw.circle(hdr_surf, (12, 4, 1), (CX, CY), shadow_rad)
    pygame.draw.circle(hdr_surf, (4, 1, 0), (CX, CY), int(shadow_rad * 0.94))

    # Draw the razor-thin glowing golden-white Photon Ring inside the shadow edge
    photon_rad = int(R_SHADOW * 0.96 * cam_scale)
    pygame.draw.circle(hdr_surf, (255, 195, 110), (CX, CY), photon_rad, 2)

    # Re-draw the FRONT crossing band of the accretion disk over the black hole shadow!
    front_mask = (z_local < -15.0) & valid
    fvx, fvy = scr_x[front_mask], scr_y[front_mask]
    front_hdr = np.zeros((W, H, 3), dtype=np.float32)
    f_gain = gain[front_mask] * 1.25
    np.add.at(front_hdr[:, :, 0], (fvx, fvy), base_r[front_mask] * f_gain)
    np.add.at(front_hdr[:, :, 1], (fvx, fvy), base_g[front_mask] * f_gain)
    np.add.at(front_hdr[:, :, 2], (fvx, fvy), base_b[front_mask] * f_gain)

    f_lum = (front_hdr[:, :, 0] * 0.45 + front_hdr[:, :, 1] * 0.4 + front_hdr[:, :, 2] * 0.15)
    f_white = np.clip(f_lum - 2.5, 0.0, 25.0) * 0.55
    front_rgb = np.clip((front_hdr + f_white[:, :, None]) * 42.0, 0, 255).astype(np.uint8)
    front_surf = pygame.Surface((W, H))
    pygame.surfarray.blit_array(front_surf, front_rgb)
    hdr_surf.blit(front_surf, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

    # --------------------------------------------------------------------------
    # 5. MULTI-PASS OPTICAL BLOOM & DEEP-SPACE COMPOSITING
    # --------------------------------------------------------------------------
    master_surf.blit(space_bg_surf, (0, 0))
    master_surf.blit(hdr_surf, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

    # Pass 1: Tight intense plasma glow
    bloom_1 = pygame.transform.smoothscale(hdr_surf, (W // 3, H // 3))
    bloom_1 = pygame.transform.smoothscale(bloom_1, (W, H))
    master_surf.blit(bloom_1, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

    # Pass 2: Wide atmospheric orange/gold corona
    bloom_2 = pygame.transform.smoothscale(hdr_surf, (W // 10, H // 10))
    bloom_2 = pygame.transform.smoothscale(bloom_2, (W, H))
    master_surf.blit(bloom_2, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

    # Scientific Simulation Disclaimer Watermark
    if CONFIG["SHOW_LABEL"]:
        lbl = font_label.render(CONFIG["LABEL_TEXT"], True, (180, 145, 110))
        lbl.set_alpha(155)
        master_surf.blit(lbl, (W // 2 - lbl.get_width() // 2, H - 70))

    # --------------------------------------------------------------------------
    # 6. LIVE PREVIEW & STREAM TO 1080x1920 H.264 MP4
    # --------------------------------------------------------------------------
    preview_frame = pygame.transform.smoothscale(master_surf, (PW, PH))
    preview_screen.blit(preview_frame, (0, 0))
    pygame.display.flip()

    if writer is not None:
        frame_arr = pygame.surfarray.array3d(master_surf).transpose([1, 0, 2])
        writer.append_data(frame_arr)

    frame_idx += 1
    if frame_idx % 60 == 0:
        print(f"Rendered {frame_idx}/{total_frames} frames ({int(frame_idx / total_frames * 100)}%)")

if writer is not None:
    writer.close()
    print(f"[VIDEO] Saved 1080x1920 video to '{CONFIG['OUTPUT_MP4']}'")

    
    if CONFIG["GENERATE_AUDIO"]:
        wav_filename = "black_hole_amibende.wav"
        temp_mp4 = "temp_video.mp4"
        generate_cosmic_drone_wav(wav_filename, CONFIG["DURATION"])
        try:
            os.replace(CONFIG["OUTPUT_MP4"], temp_mp4)
            ffmpeg_cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", temp_mp4, "-i", wav_filename, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", CONFIG["OUTPUT_MP4"]]

            subprocess.run(ffmpeg_cmd, check= True)

            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.remove(temp_mp4)

            print(f"SUCCESS: Final reel with cosmic drone audio saved to '{CONFIG['OUTPUT_MP4']}'")

        except Exception as e:
            print(f"[INFO] Audio muxing failed: {e}. The video was saved without audio.")


pygame.quit()
sys.exit(0)