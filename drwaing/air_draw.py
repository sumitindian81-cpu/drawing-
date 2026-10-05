import cv2
import mediapipe as mp
import numpy as np
import math
import time
from collections import deque

WIDTH, HEIGHT = 1280, 720
WINDOW_NAME = "Air Draw (Pinch = Write) - Pen Mode v5 (Pinch-Center Stable)"

BRUSH_COLOR = (0, 0, 255)
BRUSH_THICKNESS = 6

# --- pinch thresholds (3D ratio) ---
PINCH_ON_RATIO = 0.55
PINCH_OFF_RATIO = 0.72
ON_DEBOUNCE_FRAMES = 2

# --- quality ---
MIN_PALM_PX = 60
INTERP_STEP_PX = 3

# --- pointer stability ---
MAX_STEP_MULT = 0.22          # max pixels/frame = palm_px * this  (0.18-0.30)
DEADZONE_MULT = 0.018         # deadzone pixels = palm_px * this   (0.015-0.03)

# extra smoothing stage
EMA_ALPHA_IDLE = 0.45
EMA_ALPHA_WRITE = 0.22

# --- One Euro filter ---
MIN_CUTOFF = 1.0
BETA = 0.012
D_CUTOFF = 1.0

# --- MP low-light preprocess (optional) ---
USE_PREPROCESS = True
MP_GAMMA = 1.35

try:
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
except AttributeError:
    from mediapipe.python.solutions import hands as mp_hands
    from mediapipe.python.solutions import drawing_utils as mp_draw


def draw_text(img, text, org, scale=0.6, color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 2, cv2.LINE_AA)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class LowPass:
    def __init__(self):
        self.y = None

    def filter(self, x, a):
        if self.y is None:
            self.y = x
        else:
            self.y = a * x + (1 - a) * self.y
        return self.y


def _alpha(cutoff, dt):
    tau = 1.0 / (2 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / max(dt, 1e-6))


class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_f = LowPass()
        self.dx_f = LowPass()
        self.last_x = None
        self.last_t = None

    def reset(self):
        self.x_f.y = None
        self.dx_f.y = None
        self.last_x = None
        self.last_t = None

    def __call__(self, x, t):
        if self.last_t is None:
            self.last_t = t
            self.last_x = x
            return x

        dt = t - self.last_t
        self.last_t = t

        dx = 0.0 if self.last_x is None else (x - self.last_x) / max(dt, 1e-6)
        self.last_x = x

        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = self.dx_f.filter(dx, a_d)

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _alpha(cutoff, dt)
        return self.x_f.filter(x, a)


def preprocess_for_mp(bgr, gamma=1.35):
    # CLAHE + gamma brighten
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    inv = 1.0 / max(gamma, 1e-6)
    table = (np.arange(256) / 255.0) ** inv * 255.0
    table = np.clip(table, 0, 255).astype(np.uint8)
    return cv2.LUT(out, table)


def draw_interpolated_line(canvas, p0, p1, color, thickness, step=3):
    x0, y0 = p0
    x1, y1 = p1
    dist = math.hypot(x1 - x0, y1 - y0)
    if dist <= step:
        cv2.line(canvas, p0, p1, color, thickness, cv2.LINE_AA)
        return
    n = int(dist // step)
    px, py = x0, y0
    for i in range(1, n + 2):
        t = i / (n + 1)
        cx = int(x0 + (x1 - x0) * t)
        cy = int(y0 + (y1 - y0) * t)
        cv2.line(canvas, (px, py), (cx, cy), color, thickness, cv2.LINE_AA)
        px, py = cx, cy


def median_point(buf):
    xs = sorted(p[0] for p in buf)
    ys = sorted(p[1] for p in buf)
    return xs[len(xs)//2], ys[len(ys)//2]


def main():
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    except Exception:
        cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    hands = mp_hands.Hands(
        max_num_hands=1,
        model_complexity=1,              # accuracy better
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WIDTH, HEIGHT)

    canvas = None

    writing = False
    on_count = 0

    # pointer filters
    fx = OneEuroFilter(MIN_CUTOFF, BETA, D_CUTOFF)
    fy = OneEuroFilter(MIN_CUTOFF, BETA, D_CUTOFF)

    # median buffer (raw)
    raw_buf = deque(maxlen=5)

    # final pointer state
    out_x = out_y = None
    last_out_x = last_out_y = None
    prev_draw_x = prev_draw_y = None

    pinch_ratio = 0.0
    palm_px = 0.0
    fps = 0.0
    t_prev = time.time()

    debug = True

    while True:
        cap.grab()
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        if canvas is None:
            canvas = np.zeros_like(frame)

        now = time.time()

        # --- Mediapipe input (same size; preprocess only for MP) ---
        mp_frame = preprocess_for_mp(frame, MP_GAMMA) if USE_PREPROCESS else frame
        rgb = cv2.cvtColor(mp_frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # display dim
        bg = frame.copy()
        cv2.rectangle(bg, (0, 0), (w, h), (0, 0, 0), -1)
        frame_disp = cv2.addWeighted(frame, 0.88, bg, 0.12, 0)

        hand_ok = False

        if results.multi_hand_landmarks:
            hand_ok = True
            hand_lms = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame_disp, hand_lms, mp_hands.HAND_CONNECTIONS)

            lm = hand_lms.landmark

            def dist3(a, b):
                dx = lm[a].x - lm[b].x
                dy = lm[a].y - lm[b].y
                dz = lm[a].z - lm[b].z
                return math.sqrt(dx*dx + dy*dy + dz*dz)

            # palm size in px
            x5, y5 = int(lm[5].x * w), int(lm[5].y * h)
            x17, y17 = int(lm[17].x * w), int(lm[17].y * h)
            palm_px = math.hypot(x5 - x17, y5 - y17)

            if palm_px < MIN_PALM_PX:
                hand_ok = False
            else:
                # pinch ratio (3D)
                palm3 = dist3(5, 17) + 1e-6
                pinch_ratio = dist3(4, 8) / palm3

                # pinch center pen point (MOST STABLE during pinch)
                tx, ty = int(lm[4].x * w), int(lm[4].y * h)   # thumb tip
                ix, iy = int(lm[8].x * w), int(lm[8].y * h)   # index tip
                pen_raw = (int((tx + ix) / 2), int((ty + iy) / 2))

                raw_buf.append(pen_raw)
                pen_med = median_point(raw_buf) if len(raw_buf) >= 3 else pen_raw

                # pinch logic
                if pinch_ratio < PINCH_ON_RATIO:
                    on_count += 1
                else:
                    on_count = 0

                if (not writing) and on_count >= ON_DEBOUNCE_FRAMES:
                    writing = True
                    prev_draw_x = prev_draw_y = None

                if writing and pinch_ratio > PINCH_OFF_RATIO:
                    writing = False
                    prev_draw_x = prev_draw_y = None
                    on_count = 0

                # adaptive deadzone (in pixels)
                deadzone_px = max(2.0, palm_px * DEADZONE_MULT)

                # OneEuro on median point
                fxv = fx(float(pen_med[0]), now)
                fyv = fy(float(pen_med[1]), now)

                if out_x is None:
                    out_x, out_y = fxv, fyv
                    last_out_x, last_out_y = out_x, out_y
                else:
                    # extra EMA stage (more stable while writing)
                    a = EMA_ALPHA_WRITE if writing else EMA_ALPHA_IDLE
                    # micro-jitter deadzone
                    if math.hypot(fxv - out_x, fyv - out_y) < deadzone_px:
                        fxv, fyv = out_x, out_y
                    out_x = (1 - a) * out_x + a * fxv
                    out_y = (1 - a) * out_y + a * fyv

                # speed limiter (prevents sudden big jumps)
                max_step = palm_px * MAX_STEP_MULT
                dx = out_x - last_out_x
                dy = out_y - last_out_y
                dist = math.hypot(dx, dy)
                if dist > max_step:
                    s = max_step / max(dist, 1e-6)
                    out_x = last_out_x + dx * s
                    out_y = last_out_y + dy * s
                last_out_x, last_out_y = out_x, out_y

                out_xi = int(clamp(int(out_x), 0, w - 1))
                out_yi = int(clamp(int(out_y), 0, h - 1))

                # draw
                if writing:
                    if prev_draw_x is not None:
                        draw_interpolated_line(
                            canvas, (prev_draw_x, prev_draw_y), (out_xi, out_yi),
                            BRUSH_COLOR, BRUSH_THICKNESS, step=INTERP_STEP_PX
                        )
                    prev_draw_x, prev_draw_y = out_xi, out_yi
                else:
                    prev_draw_x = prev_draw_y = None

                # cursor + pinch line
                cursor_color = (0, 255, 0) if writing else (0, 0, 255)
                cv2.circle(frame_disp, (out_xi, out_yi), 10, cursor_color, -1)
                cv2.line(frame_disp, (tx, ty), (ix, iy), (0, 255, 255), 2)

                if debug:
                    cv2.circle(frame_disp, pen_raw, 4, (255, 0, 0), -1)   # raw blue
                    cv2.circle(frame_disp, pen_med, 4, (0, 255, 255), -1) # median yellow

        if not hand_ok:
            # safe reset (prevents random lines)
            pinch_ratio = 0.0
            palm_px = 0.0
            writing = False
            on_count = 0
            prev_draw_x = prev_draw_y = None
            out_x = out_y = None
            last_out_x = last_out_y = None
            raw_buf.clear()
            fx.reset()
            fy.reset()

        # merge canvas
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        bg2 = cv2.bitwise_and(frame_disp, frame_disp, mask=mask_inv)
        fg = cv2.bitwise_and(canvas, canvas, mask=mask)
        combined = cv2.add(bg2, fg)

        # FPS
        dt = now - t_prev
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)
        t_prev = now

        status = "WRITING" if writing else "STOPPED"
        detect = "HAND: OK" if hand_ok else "HAND: NOT DETECTED"
        draw_text(combined, "Air Draw: Pinch (Index+Thumb) to Write - Pen Mode v5", (20, 40), 0.75)
        draw_text(combined, "C: Clear | Q: Quit | D: Debug", (20, 75), 0.65)
        draw_text(combined,
                  f"{detect}  pinch3D:{pinch_ratio:.2f}  palm_px:{palm_px:.0f}  FPS:{fps:.1f}",
                  (20, 110), 0.7)
        draw_text(combined,
                  f"deadzone~{DEADZONE_MULT:.3f}*palm  maxStep~{MAX_STEP_MULT:.2f}*palm",
                  (20, 145), 0.6)
        draw_text(combined, f"Mode: {status}", (950, 40), 0.85,
                  (0, 255, 0) if writing else (0, 255, 255))

        cv2.imshow(WINDOW_NAME, combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas[:] = 0
        elif key == ord('d'):
            debug = not debug

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

