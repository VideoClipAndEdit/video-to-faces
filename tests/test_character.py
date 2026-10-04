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

    def test_generated_hair_cue_and_zoom(self):
        image = np.zeros((160, 160, 3), dtype=np.uint8)
        image[55:125, 45:115] = (125, 175, 220)  # face skin
        image[10:58, 35:125] = (240, 240, 240)  # hair
        image[125:160, 30:130] = (90, 80, 70)  # clothing outside the head
        head = (20, 10, 140, 125, .9)
        crop, hair = character._appearance(image, head)
        self.assertEqual(crop.shape[0], 115)
        self.assertGreater(float(hair.mean()), 200)
        for scale, resized in ((.5, image[::2, ::2]),
                               (2, np.repeat(np.repeat(image, 2, axis=0), 2, axis=1))):
            scaled_head = tuple(int(value * scale) for value in head[:4]) + (.9,)
            _, scaled_hair = character._appearance(resized, scaled_head)
            self.assertLess(character._color_distance(hair, scaled_hair), .06)

    @unittest.skipUnless(HAS_LOCAL_REFERENCES, 'optional local Killua reference images are unavailable')
    def test_reference_hair_cues_and_zoom(self):
        first = _image('killua1.png')
        second = _image('killua2.png')
        first_head = (50, 0, 1200, 1110, .9)
        second_head = (120, 0, 480, 247, .9)
        _, hair1 = character._appearance(first, first_head)
        head2, hair2 = character._appearance(second, second_head)
        self.assertLess(head2.shape[0], second.shape[0])  # clothing is excluded
        self.assertGreater(float(hair1.mean()), 220)
        self.assertGreater(float(hair2.mean()), 190)
        self.assertLess(character._color_distance(hair1, hair2), .15)

        for scale in (.5, 1.5):
            resized = np.asarray(Image.fromarray(second[:, :, ::-1]).resize(
                (int(second.shape[1] * scale), int(second.shape[0] * scale))))[:, :, ::-1].copy()
            head = tuple(int(value * scale) for value in second_head[:4]) + (.9,)
            _, hair = character._appearance(resized, head)
            self.assertLess(character._color_distance(hair2, hair), .06)

    def test_generated_references_load_without_local_images(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[35:85, 25:75] = (125, 175, 220)
        image[5:40, 20:80] = (230, 230, 230)
        image[49:54, 37:45] = (90, 60, 50)
        image[78:100, 15:85] = (70, 50, 140)  # clothing reaches the face box
        face = (25, 35, 75, 85)
        head = (15, 0, 85, 100)

        class FakeCv2:
            @staticmethod
            def imread(path):
                return image.copy()

        def detect(images):
            return ([np.array([face], dtype=float) for _ in images],
                    [np.array([.9]) for _ in images], None)

        def filter_boxes(raw, *args):
            return [tuple(row) for row in raw]

        def adjust_boxes(faces, *args):
            return [head + (faces[0][4],)]

        seen = []

        def encode(crops):
            seen.extend(crops)
            return np.array([[1., 0.], [0., 1.]])

        references = character._load_references(
            [Path('portrait.png'), Path('wider.png')], detect, encode,
            FakeCv2(), filter_boxes, adjust_boxes, 18)
        self.assertEqual([item[0] for item in references], ['portrait.png', 'wider.png'])
        self.assertTrue(all(item[2] is not None for item in references))
        self.assertEqual([crop.shape for crop in seen], [(37, 50, 3)] * 2)
        self.assertTrue(all(not np.any(np.all(crop == (70, 50, 140), axis=2)) for crop in seen))

    def test_eye_appearance_does_not_affect_hair_and_reference_match(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[35:85, 25:75] = (125, 175, 220)
        image[5:40, 20:80] = (230, 230, 230)
        head = (15, 0, 85, 90, .9)
        _, reference_hair = character._appearance(image, head)
        references = [('face.png', np.array([1., 0.]), reference_hair)]
        for color in ((10, 10, 10), (40, 35, 85), (50, 130, 50)):
            candidate = image.copy()
            candidate[49:54, 37:45] = color
            _, hair = character._appearance(candidate, head)
            status, _ = character._compare((np.array([1., 0.]), hair),
                                           references, (.2, .9))
            self.assertEqual(status, 'confirmed')

    def test_hair_priority_and_multiple_references(self):
        references = [
            ('first.png', np.array([1., 0.]), np.array([240., 240., 240.])),
            ('second.png', np.array([0., 1.]), np.array([200., 200., 200.])),
        ]
        limits = (.20, .9)
        status, choice = character._compare(
            (np.array([0., 1.]), np.array([200., 200., 200.])),
            references, limits)
        self.assertEqual((status, choice[2]), ('confirmed', 'second.png'))

        status, _ = character._compare(
            (np.array([1., 0.]), np.array([20., 20., 220.])),
            references, limits)
        self.assertEqual(status, 'uncertain')  # wrong hair color gates the embedding
        status, _ = character._compare(
            (np.array([-1., 0.]), np.array([240., 240., 240.])),
            references, limits)
        self.assertEqual(status, 'uncertain')  # hair color alone is insufficient

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

        seen = []

        def encode(heads):
            seen.extend(heads)
            return np.array([[1., 0.], [0., 1.]])

        references = character._load_references(
            paths, detect, encode, FakeCv2(), filter_boxes, adjust_boxes, 18)
        self.assertEqual([item[0] for item in references], ['killua1.png', 'killua2.png'])
        self.assertTrue(all(item[2] is not None for item in references))
        for image, crop, face, head in zip(
                (_image('killua1.png'), _image('killua2.png')), seen,
                ((275, 465, 970, 970), (213, 65, 430, 229)),
                ((50, 0, 1200, 1110), (120, 0, 480, 247))):
            self.assertEqual(crop.shape, character._face_crop(image, face).shape)
            expanded = character._slice_box(image, head)
            black = lambda pixels: np.mean(np.max(pixels, axis=2) < 15)
            self.assertLess(black(crop), black(expanded))
        self.assertLessEqual(seen[0].shape[0], 380)  # ends before the portrait's clothing

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

    def test_invalid_reference_and_frame_without_head(self):
        class FakeCv2:
            @staticmethod
            def imread(path):
                return None

        with self.assertRaisesRegex(ValueError, 'cannot read reference image'):
            character._load_references(
                [Path('invalid.png')], None, None, FakeCv2(), None, None, 18)

        frame = np.zeros((50, 50, 3), dtype=np.uint8)

        def detect(frames):
            return ([np.empty((0, 4)) for _ in frames],
                    [np.empty(0) for _ in frames], None)

        rows = character._report_batch(
            [frame], [0], [0.0], detect, lambda crops: [], [],
            (.2, .9), lambda raw, *args: [], lambda faces, *args: [], 18)
        self.assertEqual(rows[0][2], 'absent')
        self.assertEqual(rows[0][4:], ['', '', ''])

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
            frame[20:60, 20:60] = (125, 175, 220)
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

        encoded_shapes = []
        encoded_batches = []

        def encode(crops):
            encoded_batches.append(len(crops))
            encoded_shapes.extend(crop.shape for crop in crops)
            return np.tile([1., 0.], (len(crops), 1))

        _, hair = character._appearance(frames[0], (10, 0, 70, 65, .9))
        references = [('=killua.png', np.array([1., 0.]), hair)]
        capture = FakeCapture(frames)
        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder) / 'matches.csv'
            count, confirmed = character._scan_episode(
                Path(folder) / 'episode.mp4', report, detect, encode, references,
                (.20, .9), 2, 18, FakeCv2(capture), filter_boxes, adjust_boxes)
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
        self.assertNotIn('eye_distance', rows[0])
        self.assertEqual(encoded_shapes, [(30, 40, 3)] * 4)
        self.assertEqual(encoded_batches, [2, 1, 1])


if __name__ == '__main__':
    unittest.main()
