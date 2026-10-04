"""Find frame-by-frame anime character appearances before making clips."""

import argparse
import csv
import math
import tempfile
from pathlib import Path

import numpy as np


VIDEO_EXTENSIONS = ('.mp4', '.mkv', '.mov', '.avi', '.webm')
REPORT_FIELDS = ('frame', 'time_sec', 'status', 'detection_score',
                 'hair_distance', 'embedding_distance', 'reference')
HEAD_SCALE = (1.6, 1.6, 2.3, 1.15)


def _layout(series_dir, character, image_extensions):
    if not isinstance(character, str) or not character or character in ('.', '..') or '/' in character or '\\' in character:
        raise ValueError('character must be one folder name')
    root = Path(series_dir).expanduser()
    if not root.is_dir() or root.is_symlink():
        raise ValueError('series_dir must be a real directory')
    root = root.resolve()
    episodes_dir = root / 'episodes'
    reference_dir = root / 'reference'
    character_dir = reference_dir / character
    results_dir = root / 'results'
    output_dir = results_dir / character
    for directory in (episodes_dir, reference_dir, character_dir):
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError('missing or unsafe input directory: %s' % directory)
    for directory in (results_dir, output_dir):
        if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
            raise ValueError('unsafe output directory: %s' % directory)
    episodes = sorted(p for p in episodes_dir.iterdir() if p.is_file() and
                      p.suffix.lower() in VIDEO_EXTENSIONS and not p.is_symlink())
    if not episodes:
        raise ValueError('no supported episode videos in %s' % episodes_dir)
    if len({p.stem for p in episodes}) != len(episodes):
        raise ValueError('episode filenames must have unique stems')
    if any(p.stem in ('', '.', '..') for p in episodes):
        raise ValueError('episode filenames must have safe stems')
    references = sorted(p for p in character_dir.iterdir() if p.is_file() and
                        p.suffix.lower() in image_extensions and not p.is_symlink())
    if not references:
        raise ValueError('no reference images in %s' % character_dir)
    for episode in episodes:
        target = output_dir / episode.stem
        if target.is_symlink() or (target.exists() and not target.is_dir()):
            raise ValueError('unsafe report directory: %s' % target)
    return episodes, references, output_dir


def _slice_box(image, box):
    x1, y1, x2, y2 = (int(value) for value in box[:4])
    return image[y1:y2, x1:x2]


def _face_crop(image, box):
    x1, y1, x2, y2 = (int(value) for value in box[:4])
    # The detector's lower edge can include neck and clothing.
    return image[y1:y1 + int(0.75 * (y2 - y1)), x1:x2]


def _appearance(image, head_box):
    """Return the head crop and its size-normalized hair color."""
    head = _slice_box(image, head_box)
    if head.size == 0:
        return head, None
    h, w = head.shape[:2]
    hair = head[:max(1, int(0.42 * h)), int(0.16 * w):max(1, int(0.84 * w))]
    return head, _hair_color(hair)


def _hair_color(region):
    if region.size == 0:
        return None
    pixels = region.reshape(-1, 3).astype(np.float32)
    brightness = pixels.mean(axis=1)
    cutoff = max(35.0, float(np.quantile(brightness, 0.45)))
    selected = pixels[brightness >= cutoff]
    if len(selected) < max(6, int(0.005 * len(pixels))):
        return None
    return np.median(selected, axis=0)


def _color_distance(a, b):
    if a is None or b is None:
        return None
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    chroma_a = a / max(float(a.sum()), 1)
    chroma_b = b / max(float(b.sum()), 1)
    chroma = min(1.0, float(np.linalg.norm(chroma_a - chroma_b)) * 1.5)
    lightness = abs(float(a.mean() - b.mean())) / 255.0
    return 0.55 * chroma + 0.45 * lightness


def _cosine_distance(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    length = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 2.0 if length == 0 else 1.0 - float(np.dot(a, b)) / length


def _compare(candidate, references, limits):
    """Hair color takes priority over the supporting embedding match."""
    embedding, hair = candidate
    hair_limit, embedding_limit = limits
    choices = []
    for name, ref_embedding, ref_hair in references:
        hair_distance = _color_distance(hair, ref_hair)
        embedding_distance = _cosine_distance(embedding, ref_embedding)
        if hair_distance is None:
            continue
        score = 0.70 * hair_distance + 0.30 * embedding_distance
        confirmed = (hair_distance <= hair_limit and
                     embedding_distance <= embedding_limit)
        choices.append((not confirmed, score, name, hair_distance, embedding_distance))
    if not choices:
        return 'uncertain', None
    choice = min(choices)
    return ('confirmed' if not choice[0] else 'uncertain'), choice


def _face_heads(frames, detector, filter_boxes, adjust_boxes, min_face_size):
    outputs = detector(frames)
    if isinstance(outputs, tuple):
        boxes, scores = outputs[:2]
        outputs = [np.column_stack((box, score)) for box, score in zip(boxes, scores)]
    all_faces = []
    for frame, raw in zip(frames, outputs):
        faces = filter_boxes(raw, frame.shape[:2], 0.4, min_face_size, 0,
                             (None, '', None, False, False, False), frame, 0)
        heads = adjust_boxes(faces, frame.shape[:2], HEAD_SCALE, False)
        all_faces.append(list(zip(faces, heads)))
    return all_faces


def _load_references(paths, detector, encoder, cv2, filter_boxes, adjust_boxes,
                     min_face_size):
    images = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError('cannot read reference image: %s' % path)
        images.append(image)
    found = _face_heads(images, detector, filter_boxes, adjust_boxes, min_face_size)
    crops = []
    colors = []
    for path, image, pairs in zip(paths, images, found):
        pairs.sort(key=lambda pair: pair[0][4], reverse=True)
        if not pairs or (len(pairs) > 1 and pairs[1][0][4] >= pairs[0][0][4] - 0.1):
            raise ValueError('reference must contain one clear detected head: %s' % path)
        head, hair = _appearance(image, pairs[0][1])
        face = _face_crop(image, pairs[0][0])
        if head.size == 0 or face.size == 0 or hair is None:
            raise ValueError('cannot assess head and hair in reference: %s' % path)
        crops.append(face)
        colors.append((path.name, hair))
    embeddings = encoder(crops)
    if len(embeddings) != len(crops):
        raise ValueError('encoder returned the wrong number of references')
    return [(name, embedding, hair) for (name, hair), embedding
            in zip(colors, embeddings)]


def _report_batch(frames, indexes, times, detector, encoder, references, limits,
                  filter_boxes, adjust_boxes, min_face_size):
    found = _face_heads(frames, detector, filter_boxes, adjust_boxes, min_face_size)
    crops = []
    cues = []
    owners = []
    for frame_index, (frame, pairs) in enumerate(zip(frames, found)):
        for face, head_box in pairs:
            head, hair = _appearance(frame, head_box)
            face_crop = _face_crop(frame, face)
            if head.size and face_crop.size and hair is not None:
                crops.append(face_crop)
                cues.append((hair, face[4]))
                owners.append(frame_index)
    embeddings = encoder(crops) if crops else []
    if len(embeddings) != len(crops):
        raise ValueError('encoder returned the wrong number of faces')
    candidates = [[] for _ in frames]
    for owner, (hair, detection_score), embedding in zip(owners, cues, embeddings):
        status, choice = _compare((embedding, hair), references, limits)
        if choice is not None:
            candidates[owner].append((status, choice, detection_score))
    rows = []
    for frame_index, time_sec, frame_candidates, pairs in zip(indexes, times, candidates, found):
        row = [frame_index, '%.6f' % time_sec, 'absent' if not pairs else 'uncertain',
               '', '', '', '']
        if frame_candidates:
            status, choice, detection_score = min(
                frame_candidates, key=lambda item: (item[0] != 'confirmed', item[1][1]))
            _, _, name, hair_distance, embedding_distance = choice
            if name.lstrip().startswith(('=', '+', '-', '@')):
                name = "'" + name
            row = [frame_index, '%.6f' % time_sec, status, '%.4f' % detection_score,
                   '%.4f' % hair_distance, '%.4f' % embedding_distance, name]
        rows.append(row)
    return rows


def _scan_episode(path, report_path, detector, encoder, references, limits, batch_size,
                  min_face_size, cv2, filter_boxes, adjust_boxes):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError('cannot open episode video: %s' % path)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError('video has no valid frame rate: %s' % path)
    count = 0
    confirmed = 0
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile('w', newline='', dir=str(report_path.parent),
                                         prefix='.matches-', suffix='.tmp',
                                         delete=False) as report:
            temporary_path = Path(report.name)
            writer = csv.writer(report)
            writer.writerow(REPORT_FIELDS)
            while True:
                frames, indexes, times = [], [], []
                for _ in range(batch_size):
                    ok, frame = capture.read()
                    if not ok:
                        break
                    frames.append(frame)
                    indexes.append(count)
                    position_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                    times.append(position_ms / 1000 if position_ms > 0 else count / fps)
                    count += 1
                if not frames:
                    break
                rows = _report_batch(frames, indexes, times, detector, encoder,
                                     references, limits, filter_boxes, adjust_boxes,
                                     min_face_size)
                confirmed += sum(row[2] == 'confirmed' for row in rows)
                writer.writerows(rows)
        if count == 0:
            raise ValueError('episode contains no readable frames: %s' % path)
        temporary_path.replace(report_path)
        temporary_path = None
    finally:
        capture.release()
        if temporary_path is not None:
            temporary_path.unlink()
    return count, confirmed


def find_character(series_dir, character, device=None, batch_size=4, min_face_size=18,
                   hair_threshold=0.20, embedding_threshold=0.90):
    """Write one frame-level ``matches.csv`` per episode; clips are a later phase."""
    if batch_size < 1 or min_face_size < 1:
        raise ValueError('batch_size and min_face_size must be positive')
    if not 0 <= hair_threshold <= 1 or not 0 <= embedding_threshold <= 2:
        raise ValueError('invalid matching threshold')
    from .prep import IMG_EXTENSIONS
    episodes, paths, output_dir = _layout(series_dir, character, IMG_EXTENSIONS)
    import cv2
    from .detection import get_detector_model, filter_boxes, adjust_boxes
    from .grouping import get_encoder_model

    detector = get_detector_model('anime', 'rcnn', device)
    encoder = get_encoder_model('anime', 'default', device)
    references = _load_references(paths, detector, encoder, cv2, filter_boxes,
                                  adjust_boxes, min_face_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    limits = (hair_threshold, embedding_threshold)
    for episode in episodes:
        report_dir = output_dir / episode.stem
        report_dir.mkdir(exist_ok=True)
        report_path = report_dir / 'matches.csv'
        frames, confirmed = _scan_episode(episode, report_path, detector, encoder,
                                           references, limits, batch_size, min_face_size,
                                           cv2, filter_boxes, adjust_boxes)
        print('%s: checked %d frames; %s; report: %s' %
              (episode.name, frames, '%d matches' % confirmed if confirmed else 'no matches',
               report_path))
        reports.append(report_path)
    return reports


def main():
    parser = argparse.ArgumentParser(description='Find an anime character in every episode frame.')
    parser.add_argument('--series-dir', required=True, help='Directory containing episodes/ and reference/<character>/')
    parser.add_argument('--character', required=True, help='Character reference folder name')
    parser.add_argument('--device', help='Torch device, for example cuda:0 or cpu')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--min-face-size', type=int, default=18)
    parser.add_argument('--hair-threshold', type=float, default=0.20)
    parser.add_argument('--embedding-threshold', type=float, default=0.90)
    args = parser.parse_args()
    try:
        find_character(**vars(args))
    except (ValueError, ImportError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
