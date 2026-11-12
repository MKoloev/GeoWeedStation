"""Load mosaic RGB thumbnails in the thread pool, deliver in the GUI thread."""
from PySide6.QtCore import QRunnable

from geo_weed_station.core.image_io import read_image, read_ms5_image, ms5_rgb_preview
from geo_weed_station.core.map_engine.mosaic import make_thumbnail


class ThumbnailJob(QRunnable):
    def __init__(self, loader):
        super().__init__()
        self.loader = loader

    def run(self):
        if self.loader.callback() is None:
            return
        try:
            path = self.loader.path
            is_ms5 = False
            if path.suffix.lower() in {'.tif', '.tiff'}:
                import tifffile
                with tifffile.TiffFile(path) as tif:
                    series = tif.series[0]
                    is_ms5 = (len(series.shape) == 3 and (
                        (series.axes in {'YXS', 'YXC'} and series.shape[-1] == 5)
                        or (series.axes in {'SYX', 'CYX', 'QYX', 'IYX'} and series.shape[0] == 5)))
            rgb = ms5_rgb_preview(read_ms5_image(path)) if is_ms5 else read_image(path)
            thumbnail = make_thumbnail(rgb)
            thumbnail.setflags(write=False)
            self.loader.result.emit(thumbnail)
        except Exception:
            # A missing/corrupt raster leaves the footprint outline visible.
            self.loader.result.emit(None)
