"""Shot-aware 16:9 -> 9:16 reframing: find camera cuts, find the speaker's face in each shot,
and build an ffmpeg filter that crops each shot around the face (static crop per shot, like a
human editor would, with no jittery tracking)."""
import re
import subprocess

import cv2
import numpy as np

from .common import ASSETS, FPS, H, W

# YuNet face detector (OpenCV Zoo, MIT licence): small, fast, and handles turned heads and small faces
YUNET = ASSETS / 'models' / 'face_detection_yunet_2023mar.onnx'
_DET = {}
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass


def _detector(w, h):
    if 'yunet' not in _DET:
        _DET['yunet'] = cv2.FaceDetectorYN.create(str(YUNET), '', (w, h), 0.6, 0.3, 5000)
    d = _DET['yunet']
    d.setInputSize((w, h))
    return d


def scene_cuts(video, start, dur, threshold=0.30, min_shot=0.8):
    """Return shot boundaries [(a, b), ...] in seconds relative to `start`."""
    cmd = ['ffmpeg', '-hide_banner', '-ss', f'{start:.3f}', '-t', f'{dur:.3f}', '-i', str(video),
           '-vf', f"scale=320:-2,select='gt(scene,{threshold})',showinfo", '-an', '-f', 'null', '-']
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    times = [float(m) for m in re.findall(r'pts_time:([0-9.]+)', r.stderr)]
    bounds = [0.0]
    for t in times:
        if t - bounds[-1] >= min_shot and dur - t >= min_shot:
            bounds.append(t)
    bounds.append(dur)
    return list(zip(bounds[:-1], bounds[1:]))


def _faces(img):
    """All face boxes (x, y, w, h) in a BGR image."""
    h, w = img.shape[:2]
    _, faces = _detector(w, h).detect(img)
    if faces is None:
        return []
    return [tuple(float(v) for v in f[:4]) for f in faces if f[2] >= 18]


def face_in_shot(cap, t0, t1, samples=5):
    """Median face centre (cx, cy as 0..1) and size (0..1 of frame height), or None."""
    pts = []
    for k in range(samples):
        t = t0 + (t1 - t0) * (k + 0.5) / samples
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        h0, w0 = frame.shape[:2]
        scale = 640 / w0
        small = cv2.resize(frame, (640, int(h0 * scale)))
        faces = _faces(small)
        if faces:
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            sh = small.shape[0]
            pts.append(((x + w / 2) / 640, (y + h / 2) / sh, h / sh))
    if len(pts) < max(2, samples // 3):
        return None
    a = np.array(pts)
    return tuple(float(v) for v in np.median(a, axis=0))


def analyse_range(video, start, dur, src_w, src_h):
    """Shots for one source range: [{'a','b','face': (cx,cy,size)|None}] (times relative to start)."""
    shots = scene_cuts(video, start, dur)
    cap = cv2.VideoCapture(str(video))
    out = []
    for a, b in shots:
        out.append({'a': a, 'b': b, 'face': face_in_shot(cap, start + a, start + b)})
    cap.release()
    return out


def crop_width(src_w, src_h):
    return min(src_w, int(src_h * 9 / 16) // 2 * 2)


def crop_x(cx, src_w, src_h):
    cw = crop_width(src_w, src_h)
    return int(min(max(cx * src_w - cw / 2, 0), src_w - cw)) // 2 * 2


def plan_shots(shots, src_w, src_h, min_fit=1.0):
    """Decide a crop (or 'fit' layout) for each shot. A shot with no face (quote card, slide, screen
    recording, wide b-roll) is shown whole over a blurred background, so no text gets cut off.
    Very short faceless shots (< min_fit s, usually a missed detection) reuse the nearest face."""
    is_vertical = src_w / src_h < 0.75
    known = [s['face'] for s in shots if s['face']]
    plan, last = [], (known[0] if known else None)
    for s in shots:
        face = s['face'] or (last if s['b'] - s['a'] < min_fit else None)
        if s['face']:
            last = s['face']
        if is_vertical:
            mode, x0, fx, fy = 'vertical', 0, (face[0] if face else 0.5) * W, (face[1] if face else 0.4) * H
        elif face is None:
            mode, x0, fx, fy = 'fit', 0, W / 2, H * 0.42
        else:
            cw = crop_width(src_w, src_h)
            x0 = crop_x(face[0], src_w, src_h)
            mode = 'crop'
            fx = (face[0] * src_w - x0) / cw * W
            fy = face[1] * H
        plan.append({**s, 'mode': mode, 'x0': x0, 'focus': [round(fx), round(fy)]})
    return plan


def shot_filter(inp, shot, src_w, src_h, label, fps=FPS):
    """ffmpeg filter chain turning one shot of input `inp` into 1080x1920 [label]."""
    trim = f"trim=start={shot['a']:.3f}:end={shot['b']:.3f},setpts=PTS-STARTPTS"
    tail = f'setsar=1,fps={fps},format=yuv420p'
    if shot['mode'] == 'vertical':
        return (f"[{inp}]{trim},scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,"
                f"crop={W}:{H},{tail}[{label}]")
    if shot['mode'] == 'fit':
        return (f"[{inp}]{trim},split[{label}b][{label}f];"
                f"[{label}b]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
                f"boxblur=30:3,eq=brightness=-0.08[{label}bb];"
                f"[{label}f]scale={W}:-2:flags=lanczos[{label}ff];"
                f"[{label}bb][{label}ff]overlay=0:(H-h)/2*0.85,{tail}[{label}]")
    if shot['mode'] == 'split':
        bx, by, bw, bh = shot['facecam']
        th, gh = shot['top_h'], H - shot['top_h']
        return (f"[{inp}]{trim},split[{label}c][{label}g];"
                f"[{label}c]crop={bw}:{bh}:{bx}:{by},scale={W}:{th}:force_original_aspect_ratio=increase:"
                f"flags=lanczos,crop={W}:{th}[{label}ct];"
                f"[{label}g]crop={shot['gw']}:{src_h}:{shot['gx']}:0,scale={W}:{gh}:"
                f"force_original_aspect_ratio=increase:flags=lanczos,crop={W}:{gh}[{label}gt];"
                f"[{label}ct][{label}gt]vstack,{tail}[{label}]")
    cw = crop_width(src_w, src_h)     # 'crop' (around a face) and 'gameplay' (around crop_x)
    return (f"[{inp}]{trim},crop={cw}:{src_h}:{shot['x0']}:0,scale={W}:{H}:flags=lanczos,"
            f"{tail}[{label}]")


# ---------------- gameplay layouts (no face-led reframing) ----------------

def _even(v):
    return max(2, int(round(v)) // 2 * 2)


def parse_box(text, src_w, src_h):
    """'x,y,w,h' in source pixels, or as 0..1 fractions of the frame -> [x, y, w, h] in pixels."""
    try:
        v = [float(p) for p in text.split(',')]
    except ValueError:
        v = []
    if len(v) != 4 or v[2] <= 0 or v[3] <= 0:
        return None
    if max(v) <= 1.0:
        v = [v[0] * src_w, v[1] * src_h, v[2] * src_w, v[3] * src_h]
    x, y = min(max(v[0], 0), src_w - 16), min(max(v[1], 0), src_h - 16)
    w, h = min(v[2], src_w - x), min(v[3], src_h - y)
    return [int(x) // 2 * 2, int(y) // 2 * 2, _even(w), _even(h)]


def _sample_times(ranges, n):
    total = sum(e - s for s, e in ranges) or 1.0
    out = []
    for s, e in ranges:
        k = max(1, int(round(n * (e - s) / total)))
        out += [s + (e - s) * (i + 0.5) / k for i in range(k)]
    return out


def _edge(profile, lo, hi, default, frame_edge):
    """Strongest persistent line in profile[lo:hi]. When nothing stands out: the frame edge if
    the search reached it (overlay flush with the border), else `default`."""
    n = len(profile)
    at_border = (frame_edge == 0 and lo <= 2) or (frame_edge != 0 and hi >= n - 2)
    lo, hi = max(1, int(lo)), min(n - 1, int(hi))
    if hi - lo >= 3:
        seg = profile[lo:hi]
        i = int(np.argmax(seg))
        if seg[i] > max(6.0, 3.0 * float(np.median(seg)) + 2.0):
            return float(lo + i)
    return float(frame_edge) if at_border else float(default)


def detect_facecam(video, ranges, src_w, src_h, samples=30):
    """Find a streamer's webcam overlay: a face that stays in one place while the game moves.
    Returns ([x, y, w, h] in source pixels, share of frames it was seen in) or (None, share).
    The box edges come from lines that stay put across frames (the overlay's border); where none
    stands out, a 16:9 box around the face is assumed. Always check the result with --preview."""
    cap = cv2.VideoCapture(str(video))
    dw = 960
    dh = int(round(src_h * dw / src_w))
    faces, grads_x, grads_y, seen = [], [], [], 0
    for t in _sample_times(ranges, samples):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        seen += 1
        small = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA)
        for x, y, w, h in _faces(small):
            faces.append((seen, x, y, w, h))
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        grads_x.append(np.abs(np.diff(g, axis=1)))
        grads_y.append(np.abs(np.diff(g, axis=0)))
    cap.release()
    if seen < 4 or not faces:
        return None, 0.0

    # the facecam face: the cluster of same-place faces seen in the most frames
    clusters = []
    for f in faces:
        cx, cy = f[1] + f[3] / 2, f[2] + f[4] / 2
        for c in clusters:
            if abs(cx - c['cx']) < 0.05 * dw and abs(cy - c['cy']) < 0.05 * dw and 0.6 < f[4] / c['h'] < 1.7:
                c['items'].append(f)
                n = len(c['items'])
                c['cx'] += (cx - c['cx']) / n
                c['cy'] += (cy - c['cy']) / n
                c['h'] += (f[4] - c['h']) / n
                break
        else:
            clusters.append({'cx': cx, 'cy': cy, 'h': f[4], 'items': [f]})
    best = max(clusters, key=lambda c: len({i[0] for i in c['items']}))
    share = len({i[0] for i in best['items']}) / seen
    if share < 0.4:
        return None, share
    arr = np.array([i[1:] for i in best['items']])
    fx, fy, fw, fh = (float(v) for v in np.median(arr, axis=0))
    if fh > 0.5 * dh:                       # a full-screen talking head, not an overlay
        return None, share

    # persistent lines: a low percentile over time keeps what never moves (the overlay border)
    px = np.percentile(np.stack(grads_x), 25, axis=0)     # (dh, dw-1): vertical lines
    py = np.percentile(np.stack(grads_y), 25, axis=0)     # (dh-1, dw): horizontal lines
    cam_h = 2.5 * fh
    cam_w = cam_h * 16 / 9
    cxm, cym = fx + fw / 2, fy + fh * 0.55
    # A border runs the full length of the overlay; an object inside the webcam picture (a shelf,
    # a curtain) does not. So a line is scored by its weakest stretch (low percentile along it),
    # first over the rows/columns around the face, then again over the box found so far.
    r0, r1 = max(0, fy - 0.3 * fh), min(dh, fy + 1.6 * fh)
    c0, c1 = max(0, fx - 0.3 * fw), min(dw, fx + 1.3 * fw)
    for _ in range(3):
        col = np.percentile(px[int(r0):max(int(r0) + 2, int(r1))], 20, axis=0)
        row = np.percentile(py[:, int(c0):max(int(c0) + 2, int(c1))], 20, axis=1)
        left = _edge(col, fx - 8.0 * fw, fx - 0.25 * fw, cxm - cam_w / 2, 0)
        right = _edge(col, fx + 1.25 * fw, fx + 9.0 * fw, cxm + cam_w / 2, dw)
        top = _edge(row, fy - 4.0 * fh, fy - 0.15 * fh, cym - cam_h / 2, 0)
        bottom = _edge(row, fy + 1.2 * fh, fy + 7.0 * fh, cym + cam_h / 2, dh)
        left, right = max(0.0, left), min(float(dw), right)
        top, bottom = max(0.0, top), min(float(dh), bottom)
        mx, my = 0.08 * (right - left), 0.08 * (bottom - top)
        r0, r1, c0, c1 = top + my, bottom - my, left + mx, right - mx
    k = src_w / dw
    pad = 3                                  # stay inside the border line
    box = parse_box(f'{left * k + pad},{top * k + pad},{(right - left) * k - 2 * pad},'
                    f'{(bottom - top) * k - 2 * pad}', src_w, src_h)
    return box, share


def layout_shot(dur, layout, src_w, src_h):
    """One whole-range shot for a fixed layout. layout = {'mode': 'gameplay'|'fit'|'split',
    'crop_x': 0..1, 'facecam': [x, y, w, h]}."""
    mode, auto_x = layout['mode'], layout.get('crop_x') is None
    cx = 0.5 if auto_x else float(layout['crop_x'])
    shot = {'a': 0.0, 'b': dur, 'face': None, 'mode': mode, 'x0': 0}
    if mode == 'fit':
        shot['focus'] = [W // 2, round(H * 0.42)]
    elif mode == 'gameplay':                 # fill the phone screen with the middle of the game
        shot['x0'] = crop_x(cx, src_w, src_h)
        shot['focus'] = [W // 2, H // 2]
    else:                                    # split: webcam on top, game below
        bx, by, bw, bh = layout['facecam']
        top_h = min(800, max(480, _even(W * bh / bw)))
        gw = min(src_w, _even(src_h * W / (H - top_h)))
        gx = cx * src_w - gw / 2
        if auto_x and bx < gx + gw and bx + bw > gx:     # the webcam pokes into the game panel:
            if bx + bw / 2 < src_w / 2:                  # slide the game area off it when there is room
                gx = max(gx, min(bx + bw, src_w - gw))
            else:
                gx = min(gx, max(bx - gw, 0))
        gx = int(min(max(gx, 0), src_w - gw)) // 2 * 2
        # zooms scale around the seam, so the line between the two panels never moves
        shot.update(facecam=[bx, by, bw, bh], top_h=top_h, gw=gw, gx=gx, focus=[W // 2, top_h])
    return shot


def preview_image(video, t, shot, src_w, src_h, out_path):
    """Source frame with the boxes drawn, next to the composed 9:16 frame, for a quick visual check."""
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return False
    frame = cv2.resize(frame, (src_w, src_h))

    def fill(img, w, h):
        k = max(w / img.shape[1], h / img.shape[0])
        r = cv2.resize(img, (max(w, int(round(img.shape[1] * k))), max(h, int(round(img.shape[0] * k)))),
                       interpolation=cv2.INTER_AREA)
        x, y = (r.shape[1] - w) // 2, (r.shape[0] - h) // 2
        return r[y:y + h, x:x + w]

    marked, thick = frame.copy(), max(3, src_w // 320)
    if shot['mode'] == 'split':
        bx, by, bw, bh = shot['facecam']
        th = shot['top_h']
        game = (shot['gx'], 0, shot['gw'], src_h)
        out = np.vstack([fill(frame[by:by + bh, bx:bx + bw], W, th),
                         fill(frame[:, game[0]:game[0] + game[2]], W, H - th)])
        cv2.rectangle(marked, (bx, by), (bx + bw, by + bh), (0, 255, 0), thick)
    elif shot['mode'] == 'gameplay':
        game = (shot['x0'], 0, crop_width(src_w, src_h), src_h)
        out = fill(frame[:, game[0]:game[0] + game[2]], W, H)
    else:
        game = (0, 0, src_w, src_h)
        out = np.zeros((H, W, 3), np.uint8)
        f = fill(frame, W, int(W * src_h / src_w))
        y = (H - f.shape[0]) // 2
        out[y:y + f.shape[0]] = f
    cv2.rectangle(marked, (game[0], game[1]), (game[0] + game[2], game[1] + game[3]), (255, 160, 0), thick)
    ph = 960
    left = cv2.resize(marked, (960, int(960 * src_h / src_w)), interpolation=cv2.INTER_AREA)
    canvas = np.full((ph, 960 + 20 + 540, 3), 24, np.uint8)
    canvas[:left.shape[0], :960] = left[:ph]
    canvas[:, 980:] = cv2.resize(out, (540, ph), interpolation=cv2.INTER_AREA)
    return bool(cv2.imwrite(str(out_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 85]))
