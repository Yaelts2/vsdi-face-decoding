import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from functions_scripts.preprocessing_functions import green_gray_magenta
ourCmap = green_gray_magenta()



def show_weight_movie(W_pixel_time,                 # (pixels*pixels, n_windows) e.g. (10000, 106)
                    centers=None,                 # (n_windows,) optional (center frame per window)
                    frame_acc=None,               # (n_windows,) optional
                    trial_acc=None,               # (n_windows,) optional
                    pixels=100,
                    fps=10,
                    cmap=ourCmap,
                    clip_q=99,                    # percentile for symmetric clipping
                    title="Weight maps over time"):
    """
    Interactive viewer for weight maps over time using a slider + Play/Pause button.

    - W_pixel_time: (pixels*pixels, n_windows), each column is one window (flattened image).
    - centers/frame_acc/trial_acc are optional but recommended for annotations.

    Controls:
    - Slider: scrub to any window (rewind/fast-forward)
    - Play/Pause: animate forward
    """

    W = np.asarray(W_pixel_time, dtype=float)
    if W.ndim != 2:
        raise ValueError("W_pixel_time must be 2D: (pixels*pixels, n_windows)")
    n_pix, n_win = W.shape
    if n_pix != pixels * pixels:
        raise ValueError(f"Expected {pixels*pixels} rows, got {n_pix}")

    # center frames and accuracies for annotation 
    if centers is None:
        centers = np.arange(n_win)
    centers = np.asarray(centers)
    if centers.shape[0] != n_win:
        raise ValueError("centers must have length n_windows")

    if frame_acc is not None:
        frame_acc = np.asarray(frame_acc, dtype=float)
        if frame_acc.shape[0] != n_win:
            raise ValueError("frame_acc must have length n_windows")

    if trial_acc is not None:
        trial_acc = np.asarray(trial_acc, dtype=float)
        if trial_acc.shape[0] != n_win:
            raise ValueError("trial_acc must have length n_windows")

    # Global symmetric clip based on percentile of absolute values (ignoring NaNs)
    vals = W[np.isfinite(W)]
    if vals.size == 0:
        raise ValueError("No finite values in W_pixel_time (all NaN?)")
    vmax = np.percentile(np.abs(vals), clip_q)
    vmin = -vmax

    # Figure layout
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.subplots_adjust(bottom=0.22)  # space for widgets

    # Initial frame
    idx0 = 0
    img0 = W[:, idx0].reshape(pixels, pixels)
    im = ax.imshow(img0, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis("off")

    # Title line with accuracy
    def make_status(i):
        parts = [f"win {i+1}/{n_win}", f"center={centers[i]}"]
        if frame_acc is not None:
            parts.append(f"frame_acc={frame_acc[i]:.3f}")
        if trial_acc is not None:
            parts.append(f"trial_acc={trial_acc[i]:.3f}")
        return " | ".join(parts)

    ax.set_title(f"{title}\n{make_status(idx0)}")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Weight")

    # Slider
    ax_slider = plt.axes([0.15, 0.10, 0.70, 0.03])
    slider = Slider(ax_slider, "Window", 0, n_win - 1, valinit=0, valstep=1)

    # Play/Pause button
    ax_btn = plt.axes([0.15, 0.02, 0.18, 0.06])
    btn = Button(ax_btn, "Play")

    # Rewind button (optional but useful)
    ax_rew = plt.axes([0.37, 0.02, 0.18, 0.06])
    btn_rew = Button(ax_rew, "Rewind")

    # Speed display (static)
    ax_spd = plt.axes([0.59, 0.02, 0.26, 0.06])
    ax_spd.axis("off")
    ax_spd.text(0, 0.5, f"fps={fps}  |  clip=±p{clip_q}", va="center")

    playing = {"on": False}

    def render(i):
        i = int(i)
        frame = W[:, i].reshape(pixels, pixels)
        im.set_data(frame)
        ax.set_title(f"{title}\n{make_status(i)}")
        fig.canvas.draw_idle()

    def on_slider(val):
        playing["on"] = False
        btn.label.set_text("Play")
        render(int(val))

    slider.on_changed(on_slider)

    # Timer-based animation (lightweight)
    interval_ms = int(1000 / max(1, fps))
    timer = fig.canvas.new_timer(interval=interval_ms)

    def step_forward():
        if not playing["on"]:
            return
        i = int(slider.val)
        i2 = (i + 1) if (i + 1) < n_win else 0
        slider.set_val(i2)  # triggers render via slider callback

    timer.add_callback(step_forward)

    def on_play(event):
        playing["on"] = not playing["on"]
        btn.label.set_text("Pause" if playing["on"] else "Play")
        if playing["on"]:
            timer.start()
        else:
            timer.stop()

    def on_rewind(event):
        playing["on"] = False
        btn.label.set_text("Play")
        timer.stop()
        slider.set_val(0)

    btn.on_clicked(on_play)
    btn_rew.on_clicked(on_rewind)

    plt.show()
