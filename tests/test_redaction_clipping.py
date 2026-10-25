import itertools
import random
import unittest

from PIL import Image

from doc_page_extractor import (
    Layout, OCRPageResult, clip_layouts_to_unmasked_area,
    create_page_extractor_with_adapter,
)
from doc_page_extractor.structure import build_structured_page


def intersects(a, b):
    return max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])


class TestRedactionClipping(unittest.TestCase):
    def test_captured_page_four_preserves_real_digit(self):
        source = Layout((0, 0, 2474, 3497), '4', polygon=[(0, 0), (2474, 3497)])
        result = clip_layouts_to_unmasked_area(
            [source], [(0, 0, 2481, 2339), (285, 2339, 2188, 3157)],
        )
        self.assertEqual(result[0].det, (0, 3158, 2474, 3497))
        self.assertEqual(result[0].text, '4')
        self.assertIsNone(result[0].polygon)
        self.assertEqual(source.det, (0, 0, 2474, 3497))
        x1, y1, x2, y2 = result[0].det
        self.assertTrue(x1 <= 1228 < 1249 <= x2 and y1 <= 3247 < 3278 <= y2)

    def test_covered_empty_and_untouched_layouts(self):
        untouched = Layout((4, 4, 6, 6), 'keep', polygon=[(4, 4)])
        result = clip_layouts_to_unmasked_area(
            [Layout((0, 0, 3, 3), 'drop'), Layout((2, 2, 2, 3), 'empty'), untouched],
            [(0, 0, 2, 2)],
        )
        self.assertEqual(result, [untouched])
        self.assertIs(result[0], untouched)

    def test_inclusive_mask_edge_is_excluded(self):
        result = clip_layouts_to_unmasked_area([Layout((0, 0, 5, 5), 'x')], [(0, 0, 4, 2)])
        self.assertEqual(result[0].det, (0, 3, 5, 5))

    def test_matches_exhaustive_search_with_overlapping_masks(self):
        rng = random.Random(20261026)
        bounds = list(itertools.combinations(range(7), 2))
        for _ in range(150):
            masks = []
            for _ in range(rng.randrange(1, 6)):
                a, c = rng.choice(bounds)
                b, d = rng.choice(bounds)
                masks.append((a, b, c - 1, d - 1))
            free = []
            for a, c in bounds:
                for b, d in bounds:
                    candidate = (a, b, c, d)
                    if not any(intersects(candidate, (x, y, z + 1, w + 1)) for x, y, z, w in masks):
                        free.append(candidate)
            def key(box):
                a, b, c, d = box
                return ((c-a)*(d-b), -b, -a, -d, -c)
            expected = max(free, key=key) if free else None
            for ordered in (masks, list(reversed(masks))):
                result = clip_layouts_to_unmasked_area([Layout((0, 0, 6, 6), 'x')], ordered)
                self.assertEqual(result[0].det if result else None, expected)

    def test_four_stages_inherit_all_masks_and_rebuild_structure(self):
        class Adapter:
            allows_multi_stage = True
            def __init__(self):
                self.calls = 0
            def download(self, revision):
                pass
            def load(self):
                pass
            def extract_page(self, **kwargs):
                with Image.open(kwargs['image_path']) as actual:
                    if self.calls >= 2:
                        self_outer.assertEqual(actual.getpixel((15, 80)), (255, 255, 255))
                boxes = [[(10, 70, 20, 90)], [(30, 70, 40, 80)],
                         [(10, 80, 20, 90), (50, 80, 60, 85)], [(0, 0, 100, 100)]]
                layouts = [Layout(box, str(self.calls)) for box in boxes[self.calls]]
                self.calls += 1
                return OCRPageResult(layouts, 'fixture', structured=build_structured_page(layouts))
        self_outer = self
        adapter = Adapter()
        original = Image.new('RGB', (100, 100), 'white')
        original.putpixel((15, 80), (0, 0, 0))
        results = list(create_page_extractor_with_adapter(adapter).extract_page_results(
            original, size='tiny', stages=4,
        ))
        self.assertEqual(adapter.calls, 4)
        self.assertEqual(results[0][1].layouts[0].det, (10, 70, 20, 90))
        self.assertEqual(results[2][1].layouts[0].det, (50, 80, 60, 85))
        self.assertEqual(len(results[2][1].layouts), 1)
        self.assertEqual(original.getpixel((15, 80)), (0, 0, 0))
        for _, result in results:
            self.assertEqual(result.structured, build_structured_page(result.layouts))
        # Stage 4 must avoid masks from stages 1, 2 and 3, not just the latest.
        for layout in results[3][1].layouts:
            for mask in [(0, 0, 101, 68), (10, 67, 21, 91),
                         (30, 67, 41, 81), (50, 67, 61, 86)]:
                self.assertFalse(intersects(layout.det, mask))

    def test_rescaled_later_stage_is_clipped_in_original_coordinates(self):
        class Adapter:
            allows_multi_stage = True
            max_image_side = 50
            def __init__(self):
                self.calls = 0
            def download(self, revision):
                pass
            def load(self):
                pass
            def extract_page(self, **kwargs):
                with Image.open(kwargs['image_path']) as image:
                    self_outer.assertEqual(image.size, (50, 50))
                box = (5, 35, 10, 45) if self.calls % 2 == 0 else (0, 0, 50, 50)
                self.calls += 1
                return OCRPageResult([Layout(box, 'x', polygon=[(0, 0), (50, 50)])], 'fixture')
        self_outer = self
        extractor = create_page_extractor_with_adapter(Adapter())
        for _ in range(2):
            results = list(extractor.extract_page_results(
                Image.new('RGB', (100, 100), 'white'), size='tiny', stages=2,
            ))
            self.assertEqual(results[0][1].layouts[0].det, (10, 70, 20, 90))
            self.assertEqual(results[1][1].layouts[0].det, (21, 68, 100, 100))
            self.assertIsNone(results[1][1].layouts[0].polygon)
