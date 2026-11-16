"""See what is on screen at given times: one labelled contact sheet, for picking moments in
videos with little or no speech (gameplay) and for checking a pick before cutting.

    python run.py frames <project> --at 12,85.5,140        # one frame per time
    python run.py frames <project> --range 80,110          # 12 frames spread over a stretch

Writes output/<project>/frames.jpg (downloads the video the first time, like `cut`).
"""
import argparse

import cv2
import numpy as np

from .common import die, fmt_ts, load_json, probe, project_dir
from .cut import ensure_video

MAX_FRAMES, COLS, TILE_W = 16, 4, 400


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('project')
    ap.add_argument('--at', default=None, help='comma list of times in seconds (max 16)')
    ap.add_argument('--range', default=None, help='start,end in seconds: 12 evenly spread frames')
    ap.add_argument('--out', default=None, help='output image (default <project>/frames.jpg)')
    a = ap.parse_args()
    try:
        if a.range:
            s, e = (float(v) for v in a.range.split(','))
            times = [s + (e - s) * k / 11 for k in range(12)]
        else:
            times = [float(v) for v in (a.at or '').split(',') if v.strip()]
    except ValueError:
        times = []
    if not times:
        die('give --at t1,t2,...  or  --range start,end  (seconds)')
    if len(times) > MAX_FRAMES:
        print(f'   note: only the first {MAX_FRAMES} times are shown')
        times = times[:MAX_FRAMES]

    pdir = project_dir(a.project)
    video = ensure_video(pdir, load_json(pdir / 'project.json'))
    info = probe(video)
    tile_h = int(TILE_W * info['height'] / info['width'])
    cap = cv2.VideoCapture(str(video))
    tiles = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, min(t, info['duration'] - 0.1)) * 1000)
        ok, frame = cap.read()
        tile = cv2.resize(frame, (TILE_W, tile_h), interpolation=cv2.INTER_AREA) if ok else \
            np.zeros((tile_h, TILE_W, 3), np.uint8)
        label = f'{t:.1f}s  {fmt_ts(t)}'
        cv2.rectangle(tile, (0, 0), (len(label) * 11 + 10, 26), (0, 0, 0), -1)
        cv2.putText(tile, label, (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    cap.release()
    cols = min(COLS, len(tiles))
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    out = a.out or str(pdir / 'frames.jpg')
    if not cv2.imwrite(out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 80]):
        die(f'could not write {out}')
    print(f'frames: {out}   ({len(times)} frame(s), left to right, top to bottom)')


if __name__ == '__main__':
    main()
