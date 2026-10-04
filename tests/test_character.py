"""Phase 1 checks that do not require downloading Torch model weights."""

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
try:
    from PIL import Image
except ImportError:
    Image = None


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'character_under_test', str(ROOT / 'src' / 'videotofaces' / 'character.py'))
character = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(character)
REFERENCE_DIR = ROOT / 'anime' / 'hunterxhunter' / 'reference' / 'killua'
HAS_LOCAL_REFERENCES = Image is not None and all(
    (REFERENCE_DIR / name).is_file() for name in ('killua1.png', 'killua2.png'))


def _image(name):
    path = REFERENCE_DIR / name
    return np.asarray(Image.open(str(path)).convert('RGB'))[:, :, ::-1].copy()


class CharacterTests(unittest.TestCase):

    def test_generated_head_cues_and_zoom(self):
        image = np.zeros((160, 160, 3), dtype=np.uint8)
        image[10:58, 35:125] = (240, 240, 240)  # hair
        image[76:87, 61:72] = (90, 60, 50)  # one visible iris
        image[125:160, 30:130] = (90, 80, 70)  # clothing outside the head
        face = (45, 55, 115, 125, .9)
        head = (20, 10, 140, 125, .9)
        crop, hair, eye = character._appearance(image, face, head)
        self.assertEqual(crop.shape[0], 115)
        self.assertGreater(float(hair.mean()), 200)
        self.assertIsNotNone(eye)
        for scale, resized in ((.5, image[::2, ::2]),
                               (2, np.repeat(np.repeat(image, 2, axis=0), 2, axis=1))):
            scaled_face = tuple(int(value * scale) for value in face[:4]) + (.9,)
            scaled_head = tuple(int(value * scale) for value in head[:4]) + (.9,)
            _, scaled_hair, scaled_eye = character._appearance(
                resized, scaled_face, scaled_head)
            self.assertLess(character._color_distance(hair, scaled_hair), .06)
            self.assertLess(character._color_distance(eye, scaled_eye), .06)

    @unittest.skipUnless(HAS_LOCAL_REFERENCES, 'optional local Killua reference images are unavailable')
    def test_reference_head_cues_and_zoom(self):
        first = _image('killua1.png')
        second = _image('killua2.png')
        first_face = (275, 465, 970, 970, .9)
        first_head = (50, 0, 1200, 1110, .9)
        second_face = (213, 65, 430, 229, .9)
        second_head = (120, 0, 480, 247, .9)
        _, hair1, eye1 = character._appearance(first, first_face, first_head)
        head2, hair2, eye2 = character._appearance(second, second_face, second_head)
        self.assertLess(head2.shape[0], second.shape[0])  # clothing is excluded
        self.assertGreater(float(hair1.mean()), 220)
        self.assertGreater(float(hair2.mean()), 190)
        self.assertIsNotNone(eye1)
        self.assertIsNotNone(eye2)  # the visible eye is enough when one is closed
        self.assertLess(character._color_distance(hair1, hair2), .15)
        self.assertLess(character._color_distance(eye1, eye2), .20)

        for scale in (.5, 1.5):
            resized = np.asarray(Image.fromarray(second[:, :, ::-1]).resize(
                (int(second.shape[1] * scale), int(second.shape[0] * scale))))[:, :, ::-1].copy()
            face = tuple(int(value * scale) for value in second_face[:4]) + (.9,)
            head = tuple(int(value * scale) for value in second_head[:4]) + (.9,)
            _, hair, eye = character._appearance(resized, face, head)
            self.assertLess(character._color_distance(hair2, hair), .06)
            self.assertLess(character._color_distance(eye2, eye), .06)

    def test_color_priority_and_multiple_references(self):
        references = [
            ('first.png', np.array([1., 0.]), np.array([240., 240., 240.]),
             np.array([90., 60., 50.])),
            ('second.png', np.array([0., 1.]), np.array([200., 200., 200.]), None),
        ]
        limits = (.20, .20, .9)
        status, choice = character._compare(
            (np.array([0., 1.]), np.array([200., 200., 200.]), None),
            references, limits)
        self.assertEqual((status, choice[2]), ('confirmed', 'second.png'))

        status, _ = character._compare(
            (np.array([1., 0.]), np.array([240., 240., 240.]),
             np.array([50., 60., 90.])), references, limits)
        self.assertEqual(status, 'uncertain')  # wrong eye color gates a good embedding
        status, _ = character._compare(
            (np.array([1., 0.]), np.array([20., 20., 220.]), None),
            references, limits)
        self.assertEqual(status, 'uncertain')  # wrong hair color also gates it
        status, _ = character._compare(
            (np.array([1., 0.]), np.array([240., 240., 240.]), None),
            references, limits)
        self.assertEqual(status, 'confirmed')  # hidden eye is unavailable evidence

    @unittest.skipUnless(HAS_LOCAL_REFERENCES, 'optional local Killua reference images are unavailable')
    def test_supplied_references_load_as_two_head_examples(self):
        paths = [REFERENCE_DIR / name
                 for name in ('killua1.png', 'killua2.png')]

        class FakeCv2:
            @staticmethod
            def imread(path):
                return _image(Path(path).name)

        def detect(images):
            faces = [(275, 465, 970, 970), (213, 65, 430, 229)]
            return ([np.array([face], dtype=float) for face in faces],
                    [np.array([.9]), np.array([.9])], None)

        def filter_boxes(raw, *args):
            return [tuple(row) for row in raw]

        def adjust_boxes(faces, *args):
            return [(50, 0, 1200, 1110, faces[0][4])] if faces[0][0] == 275 else [
                (120, 0, 480, 247, faces[0][4])]

        def encode(heads):
            return np.array([[1., 0.], [0., 1.]])

        references = character._load_references(
            paths, detect, encode, FakeCv2(), filter_boxes, adjust_boxes, 18)
        self.assertEqual([item[0] for item in references], ['killua1.png', 'killua2.png'])
        self.assertTrue(all(item[2] is not None for item in references))
        self.assertTrue(all(item[3] is not None for item in references))

    def test_layout_rejects_unsafe_character_and_output_link(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'hunterxhunter'
            (root / 'episodes').mkdir(parents=True)
            (root / 'reference' / 'killua').mkdir(parents=True)
            (root / 'episodes' / 'episode.mp4').touch()
            (root / 'reference' / 'killua' / 'face.png').touch()
            with self.assertRaises(ValueError):
                character._layout(root, '../outside', ('.png',))
            episodes, references, output = character._layout(root, 'killua', ('.png',))
            self.assertEqual(len(episodes), 1)
            self.assertEqual(len(references), 1)
            self.assertEqual(output, root / 'results' / 'killua')
            outside = Path(folder) / 'outside'
            outside.mkdir()
            (root / 'results').symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                character._layout(root, 'killua', ('.png',))

    def test_frame_report_splits_leave_and_return(self):
        class FakeCapture:
            def __init__(self, frames):
                self.frames = frames
                self.position = 0
                self.released = False

            def isOpened(self):
                return True

            def get(self, prop):
                return 2.0 if prop == 'fps' else (self.position - 1) * 500.0

            def read(self):
                if self.position == len(self.frames):
                    return False, None
                frame = self.frames[self.position]
                self.position += 1
                return True, frame

            def release(self):
                self.released = True

        class FakeCv2:
            CAP_PROP_FPS = 'fps'
            CAP_PROP_POS_MSEC = 'time'

            def __init__(self, capture):
                self.capture = capture

            def VideoCapture(self, path):
                return self.capture

        frames = []
        for index in range(5):
            frame = np.zeros((80, 80, 3), dtype=np.uint8)
            frame[0:27, 20:60] = (240, 240, 240) if index != 4 else (20, 20, 220)
            frame[32:40, 34:44] = (90, 60, 50)
            frame[79, 79, 0] = index
            frames.append(frame)

        def detect(batch):
            boxes = [np.array([[20, 20, 60, 60]], dtype=float) if frame[79, 79, 0] != 2
                     else np.empty((0, 4)) for frame in batch]
            scores = [np.array([.9]) if len(box) else np.array([]) for box in boxes]
            return boxes, scores, None

        def filter_boxes(raw, *args):
            return [tuple(row) for row in raw]

        def adjust_boxes(faces, *args):
            return [(10, 0, 70, 65, face[4]) for face in faces]

        def encode(crops):
            return np.tile([1., 0.], (len(crops), 1))

        _, hair, eye = character._appearance(frames[0], (20, 20, 60, 60, .9),
                                              (10, 0, 70, 65, .9))
        references = [('=killua.png', np.array([1., 0.]), hair, eye)]
        capture = FakeCapture(frames)
        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder) / 'matches.csv'
            count, confirmed = character._scan_episode(
                Path(folder) / 'episode.mp4', report, detect, encode, references,
                (.20, .25, .9), 2, 18, FakeCv2(capture), filter_boxes, adjust_boxes)
            with report.open(newline='') as source:
                rows = list(csv.DictReader(source))
        self.assertEqual(count, 5)
        self.assertEqual(confirmed, 3)
        self.assertTrue(capture.released)
        self.assertEqual([int(row['frame']) for row in rows], list(range(5)))
        self.assertEqual([row['status'] for row in rows],
                         ['confirmed', 'confirmed', 'absent', 'confirmed', 'uncertain'])
        self.assertEqual([float(row['time_sec']) for row in rows], [0., .5, 1., 1.5, 2.])
        self.assertEqual(rows[0]['reference'], "'=killua.png")


if __name__ == '__main__':
    unittest.main()
