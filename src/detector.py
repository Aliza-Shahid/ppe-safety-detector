"""PPE violation detector: YOLOv8 inference + violation rules (images, video, webcam)."""
import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
from ultralytics import YOLO

VIOLATION_LABELS = {"No-Hardhat": "NO HELMET", "No-Safety-Vest": "NO VEST"}
SHOWN = {"Person", "Hardhat", "Safety-Vest", "No-Hardhat", "No-Safety-Vest"}
GREEN, RED, BLUE = (0, 200, 0), (0, 0, 255), (255, 160, 0)  # BGR colors
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Detection:
    label: str
    conf: float
    box: tuple  # x1, y1, x2, y2


class PPEDetector:
    def __init__(self, weights="models/best.pt", conf=0.4, violation_conf=0.5, imgsz=640):
        self.model = YOLO(weights)
        self.conf = conf                      # minimum confidence to keep a detection
        self.violation_conf = violation_conf  # higher bar to count something as a violation
        self.imgsz = imgsz

    def detect(self, frame):
        r = self.model(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        dets = []
        for b in r.boxes:
            label = r.names[int(b.cls[0])]
            if label in SHOWN:
                x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
                dets.append(Detection(label, float(b.conf[0]), (x1, y1, x2, y2)))
        return dets

    def violations(self, dets):
        return [d for d in dets if d.label in VIOLATION_LABELS and d.conf >= self.violation_conf]

    @staticmethod
    def draw(frame, dets, viols):
        out = frame.copy()
        viol_ids = {id(v) for v in viols}
        for d in dets:
            if id(d) in viol_ids:
                color = RED
            elif d.label in VIOLATION_LABELS:
                continue  # violation class below the violation threshold: don't draw
            elif d.label == "Person":
                color = BLUE
            else:
                color = GREEN
            x1, y1, x2, y2 = d.box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            text = f"{VIOLATION_LABELS.get(d.label, d.label)} {d.conf:.2f}"
            cv2.putText(out, text, (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        if viols:
            kinds = ", ".join(sorted({VIOLATION_LABELS[v.label] for v in viols}))
            cv2.rectangle(out, (0, 0), (out.shape[1], 32), RED, -1)
            cv2.putText(out, f"VIOLATION: {kinds}", (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return out


def process_image(det, path, out_path):
    frame = cv2.imread(str(path))
    dets = det.detect(frame)
    viols = det.violations(dets)
    cv2.imwrite(str(out_path), det.draw(frame, dets, viols))
    kinds = sorted({VIOLATION_LABELS[v.label] for v in viols})
    print(f"{Path(path).name}: {len(dets)} detections, violations: {kinds or 'none'} -> {out_path}")


def process_video(det, source, out_path, skip=5, confirm=3, show=False):
    cap = cv2.VideoCapture(source)
    is_cam = isinstance(source, int)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)) if out_path else None
    dets, viols, streak, events, i, t0 = [], [], 0, [], 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % skip == 0:  # run the model only every `skip` frames (CPU is slow)
            dets = det.detect(frame)
            viols = det.violations(dets)
            streak = streak + 1 if viols else 0
            if streak == confirm:  # violation seen in `confirm` checks in a row
                stamp = (time.time() - t0) if is_cam else i / fps
                kinds = sorted({VIOLATION_LABELS[v.label] for v in viols})
                events.append((round(stamp, 1), kinds))
                print(f"[CONFIRMED VIOLATION] t={stamp:.1f}s {kinds}")
        annotated = det.draw(frame, dets, viols)
        if writer:
            writer.write(annotated)
        if show:
            cv2.imshow("PPE detector (press q to quit)", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        i += 1
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"Done. {i} frames, {len(events)} confirmed violation event(s). Output: {out_path}")
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="image path, video path, or 0 for webcam")
    ap.add_argument("--out", default=None)
    ap.add_argument("--weights", default="models/best.pt")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--viol-conf", type=float, default=0.5)
    ap.add_argument("--skip", type=int, default=5, help="run the model every N frames")
    ap.add_argument("--confirm", type=int, default=3, help="checks in a row to confirm a violation")
    ap.add_argument("--show", action="store_true", help="show a live window")
    a = ap.parse_args()

    Path("outputs").mkdir(exist_ok=True)
    det = PPEDetector(a.weights, a.conf, a.viol_conf)
    if a.source.isdigit():
        out = a.out or "outputs/webcam_pred.mp4"
        process_video(det, int(a.source), out, a.skip, a.confirm, show=True)
    elif Path(a.source).suffix.lower() in IMAGE_EXT:
        out = a.out or f"outputs/{Path(a.source).stem}_pred.jpg"
        process_image(det, a.source, out)
    else:
        out = a.out or f"outputs/{Path(a.source).stem}_pred.mp4"
        process_video(det, a.source, out, a.skip, a.confirm, show=a.show)


if __name__ == "__main__":
    main()