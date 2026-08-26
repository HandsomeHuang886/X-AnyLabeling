import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtGui import QColor, QImage

from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.visualization import (
    _qimage_to_bgr_array,
)
from anylabeling.views.labeling.widgets.canvas import Canvas


def test_qimage_to_bgr_array_drops_alpha_and_preserves_channels():
    image = QImage(3, 1, QImage.Format.Format_ARGB32)
    image.setPixelColor(0, 0, QColor(255, 0, 0, 64))
    image.setPixelColor(1, 0, QColor(0, 255, 0, 128))
    image.setPixelColor(2, 0, QColor(0, 0, 255, 255))

    array = _qimage_to_bgr_array(image)

    assert array.shape == (1, 3, 3)
    assert array[0, 0].tolist() == [0, 0, 255]
    assert array[0, 1].tolist() == [0, 255, 0]
    assert array[0, 2].tolist() == [255, 0, 0]


class TestVisualizationObjectIndexes(unittest.TestCase):

    def setUp(self):
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])
        self.canvas = Canvas(parent=None)
        self.canvas.pixmap = QtGui.QPixmap(400, 300)
        self.canvas.pixmap.fill(QtGui.QColor("white"))
        self.canvas.resize(400, 300)

    def tearDown(self):
        self.canvas.close()
        self.app.processEvents()

    @staticmethod
    def make_rectangle(left, top, right, bottom, rgb):
        shape = Shape(label="object", shape_type="rectangle")
        shape.points = [
            QtCore.QPointF(left, top),
            QtCore.QPointF(right, top),
            QtCore.QPointF(right, bottom),
            QtCore.QPointF(left, bottom),
        ]
        shape.line_color = QtGui.QColor(*rgb)
        shape.vertex_fill_color = QtGui.QColor(*rgb)
        shape.hvertex_fill_color = QtGui.QColor(255, 255, 255)
        shape.fill_color = QtGui.QColor(*rgb, 128)
        shape.select_line_color = QtGui.QColor(255, 255, 255)
        shape.select_fill_color = QtGui.QColor(*rgb, 155)
        shape.visible = True
        shape.close()
        return shape

    def _render(self, shapes, show_object_indexes):
        return self.canvas.render_visualization(
            self.canvas.pixmap,
            shapes,
            show_labels=False,
            show_scores=False,
            show_groups=False,
            show_texts=False,
            show_masks=False,
            show_object_indexes=show_object_indexes,
        )

    def test_object_indexes_are_drawn_above_each_box_and_toggleable(self):
        # Scores are intentionally not confidence-sorted; indexes must follow
        # the shape list order, matching the XLSX export row order.
        shapes = [
            self.make_rectangle(30, 30, 100, 80, (255, 0, 0)),
            self.make_rectangle(160, 60, 260, 130, (0, 255, 0)),
        ]
        with_indexes = self._render(shapes, True)
        without_indexes = self._render(shapes, False)

        changed = sum(
            with_indexes.pixelColor(x, y) != without_indexes.pixelColor(x, y)
            for y in range(with_indexes.height())
            for x in range(with_indexes.width())
        )
        self.assertGreater(changed, 0)

        # Each index box (filled with the box's line_color) must appear just
        # above the top-left corner of its rectangle.
        first_rgb = any(
            with_indexes.pixelColor(x, y).name() == "#ff0000"
            for y in range(0, 30)
            for x in range(25, 40)
        )
        second_rgb = any(
            with_indexes.pixelColor(x, y).name() == "#00ff00"
            for y in range(0, 60)
            for x in range(155, 170)
        )
        self.assertTrue(first_rgb)
        self.assertTrue(second_rgb)

    def test_object_indexes_are_clamped_inside_image_bounds(self):
        # A rectangle flush with the image top edge must clamp its index box
        # so it stays inside the image rather than being drawn off-canvas.
        shapes = [self.make_rectangle(10, 0, 60, 40, (200, 0, 200))]
        image = self._render(shapes, True)

        clamped = any(
            image.pixelColor(x, y).name() == "#c800c8"
            for y in range(0, 22)
            for x in range(8, 30)
        )
        self.assertTrue(clamped)
