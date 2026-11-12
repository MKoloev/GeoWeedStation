"""MS5 data contracts, geometry, classification and real ONNX integration."""
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import tifffile

from geo_weed_station.config.settings import DetectorSettings
from geo_weed_station.core.exceptions import ImageReadError, DetectionError
from geo_weed_station.core.image_io import read_ms5_image, ms5_rgb_preview
from geo_weed_station.core.onnx_detector import OnnxDetector, _decode_yolox, nms
from geo_weed_station.core.source_factory import ImageSource, SourceFactory
from geo_weed_station.workers.batch_image_worker import BatchImageWorker

ROOT = Path(__file__).resolve().parents[1]


def write_ms5(path, array, planar=False, xmp=None):
    extra = [] if xmp is None else [(700, 'B', len(xmp.encode()), xmp.encode(), False)]
    tifffile.imwrite(path, array.transpose(2, 0, 1) if planar else array,
                     photometric='minisblack', planarconfig='separate' if planar else 'contig',
                     metadata={'axes': 'SYX' if planar else 'YXS'}, extratags=extra)
    return path


@pytest.mark.parametrize('dtype,maximum', [(np.uint8, 255), (np.uint16, 65535), (np.float32, 1)])
@pytest.mark.parametrize('planar', [False, True])
def test_band_order_and_normalization(tmp_path, dtype, maximum, planar):
    image = np.broadcast_to(np.array([0, .2, .4, .7, 1]) * maximum, (12, 20, 5)).astype(dtype)
    p = write_ms5(tmp_path/'ms.tif', image, planar)
    actual = read_ms5_image(p)
    np.testing.assert_allclose(actual, image.astype(np.float32)/maximum, atol=1e-7)
    assert actual.flags.c_contiguous
    np.testing.assert_array_equal(ms5_rgb_preview(actual), (actual[..., :3]*255).astype(np.uint8))


@pytest.mark.parametrize('bad', ['rgb', 'nan', 'range', 'signed'])
def test_invalid_ms5_rejected(tmp_path, bad):
    image = np.zeros((12, 20, 5), np.float32)
    if bad == 'rgb':
        image = image[..., :3]
    elif bad == 'nan':
        image[0, 0, 4] = np.nan
    elif bad == 'range':
        image[0, 0, 3] = 2
    else:
        image = image.astype(np.int16)
    p = write_ms5(tmp_path/'bad.tif', image)
    with pytest.raises(ImageReadError):
        read_ms5_image(p)


def test_rgb_source_and_video_rejected(tmp_path):
    p = tmp_path/'rgb.jpg'
    cv2.imwrite(str(p), np.zeros((12, 20, 3), np.uint8))
    with pytest.raises(ImageReadError, match='TIFF'):
        ImageSource(p, input_profile='ms5').load(ImageSource(p).entries()[0])
    with pytest.raises(ImageReadError, match='видео'):
        SourceFactory.create('video', p, input_profile='ms5')


def synthetic_detector(tmp_path, classification='c06'):
    meta = tmp_path/'ms5.json'
    meta.write_text('{"model_type":"yolox","input_profile":"ms5","classes":{"0":"plant"}}')
    det = OnnxDetector(DetectorSettings(classes_path=meta, classifier_type=classification))
    outputs = [np.full((1, 6, s, s), -30, np.float32) for s in (80, 40, 20)]
    # Box [160,160,480,480] in resized 640x640 space.
    outputs[0][0, :, 40, 40] = [0, 0, np.log(40), np.log(40), 20, 20]
    calls = {}
    def run(_, feed):
        calls['detector'] = next(iter(feed.values()))
        return outputs
    det._session = SimpleNamespace(run=run)
    det._input_name = 'ms_images'
    def classify(_, feed):
        calls['classifier'] = next(iter(feed.values()))
        logits = np.full((1, 24), -10, np.float32)
        logits[0, 2] = 10
        return [logits]
    if classification != 'none':
        det._classifier_session = SimpleNamespace(run=classify)
        det._cls_input_name = 'images'
    return det, calls


def test_resize_inverse_geometry_and_rgb_crop(tmp_path):
    det, calls = synthetic_detector(tmp_path)
    image = np.broadcast_to(np.array([.1,.2,.3,.8,.9], np.float32), (100, 200, 5)).copy()
    rgb = ms5_rgb_preview(image)
    result = det.detect(image, classification_image=rgb)
    assert len(result) == 1
    assert result[0].bbox.model_dump() == pytest.approx(dict(x=.25,y=.25,w=.5,h=.5))
    assert result[0].class_id == 2
    assert result[0].class_prob == 1
    assert calls['detector'].shape == (1,5,640,640)
    np.testing.assert_allclose(calls['detector'][0,:,0,0], image[0,0])
    assert calls['classifier'].shape == (1,3,224,224)
    expected = (rgb[0,0]/255.0-np.array([.485,.456,.406]))/np.array([.229,.224,.225])
    np.testing.assert_allclose(calls['classifier'][0,:,0,0], expected, atol=1e-6)
    with pytest.raises(DetectionError, match='совмещённый'):
        det.detect(image, classification_image=rgb[:10])


def test_batch_preserves_xmp_and_emits_rgb_only(tmp_path):
    from tests.helpers.make_xmp_image import SAMPLE_XMP_LOCAL_ARUCO
    path = write_ms5(tmp_path/'frame.tif', np.full((100,200,5), .5, np.float32),
                     xmp=SAMPLE_XMP_LOCAL_ARUCO)
    source = ImageSource(path, input_profile='ms5')
    det, _ = synthetic_detector(tmp_path, 'none')
    det.load = lambda: None
    worker = BatchImageWorker(source, detector_factory=lambda: det)
    received, finished = [], []
    worker.frame_processed.connect(received.append)
    worker.batch_finished.connect(lambda ok, failed: finished.append((ok,failed)))
    worker.process(source.entries(), True)
    assert finished == [(1,0)]
    payload = received[0]
    assert payload.has_georef()
    assert payload.detector_image is None
    assert payload.image.shape == (100,200,3)
    assert not payload.image.flags.writeable
    assert payload.detections[0].class_name == 'plant'
    from geo_weed_station.core.map_engine.weed_map import MapEngine
    from geo_weed_station.core.map_engine.projector import NadirProjector
    from geo_weed_station.config.settings import MapSettings
    engine = MapEngine(projector=NadirProjector(), settings=MapSettings())
    engine.ingest_frame(payload)
    assert engine.statistics().total_weeds == 1
    assert engine.weed_map.footprints[0].thumbnail.shape[2] == 3
    from geo_weed_station.core.annotation_io import save_annotations_to_folder, load_annotations_file
    saved = save_annotations_to_folder(tmp_path, {str(path): payload.detections}, source.entries())
    assert load_annotations_file(saved, [path])[str(path)][0].class_name == 'plant'


def test_cancel_batch_does_not_decode(tmp_path):
    source = SimpleNamespace(load=lambda entry: pytest.fail('cancelled load'))
    worker = BatchImageWorker(source)
    worker.cancel()
    worker.process([SimpleNamespace(path='unused')], True)


@pytest.mark.parametrize('kind', ['c06', 'c07'])
def test_real_models_two_stage(tmp_path, kind):
    model = ROOT/'models/d04_ms5_plant.onnx'
    if not model.exists():
        pytest.skip('Local D04 weights are not installed')
    det = OnnxDetector(DetectorSettings(model_path=model,
        classes_path=model.with_suffix('.json'), classifier_type=kind), providers=['CPUExecutionProvider'])
    try:
        det.load()
        assert det._classifier_session is not None
        image = np.full((80,120,5), .5, np.float32)
        # Real D04 inference, then force one box to exercise the real RGB classifier.
        result = det.detect(image)
        assert isinstance(result, list)
        outputs = [np.full((1,6,s,s), -30, np.float32) for s in (80,40,20)]
        outputs[0][0,:,40,40] = [0,0,np.log(40),np.log(40),20,20]
        real_session = det._session
        det._session = SimpleNamespace(run=lambda *_: outputs)
        result = det.detect(image)
        assert len(result) == 1 and result[0].class_prob is not None
        assert (result[0].stage_name is not None) == (kind == 'c07')
        det._session = real_session
    finally:
        det.close()


def test_real_d04_matches_reference_resize_and_decode():
    """Independent D04 reference math: stretch, logits, grid, exp, NMS, normalized boxes."""
    model = ROOT/'models/d04_ms5_plant.onnx'
    if not model.exists():
        pytest.skip('Local D04 weights are not installed')
    det = OnnxDetector(DetectorSettings(model_path=model, classes_path=model.with_suffix('.json')),
                       providers=['CPUExecutionProvider'])
    try:
        det.load()
        image = np.random.default_rng(42).random((120,200,5), dtype=np.float32)
        reference_input = np.stack([cv2.resize(image[...,c], (640,640)) for c in range(5)])[None]
        raw = det._session.run(None, {det._input_name: reference_input})
        boxes, scores = [], []
        for output, stride in zip(raw, (8,16,32)):
            reg = output[0,:4]
            score = (1/(1+np.exp(-np.clip(output[0,4],-80,80)))) * (1/(1+np.exp(-np.clip(output[0,5],-80,80))))
            y, x = np.indices(score.shape)
            cx, cy = (reg[0]+x)*stride, (reg[1]+y)*stride
            w, h = np.exp(reg[2])*stride, np.exp(reg[3])*stride
            boxes.append(np.stack([cx-w/2,cy-h/2,cx+w/2,cy+h/2],-1).reshape(-1,4))
            scores.append(score.ravel())
        boxes = np.concatenate(boxes).astype(np.float32)
        scores = np.concatenate(scores)
        # Always exercise actual boxes even if the synthetic image has no confident plants.
        det.confidence_threshold = float(np.max(scores)) * .90
        selected = scores >= det.confidence_threshold
        boxes, scores = boxes[selected], scores[selected]
        xywh = boxes.copy(); xywh[:,2:] -= xywh[:,:2]
        indices = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), det.confidence_threshold,
                                  det.nms_iou_threshold)
        keep = np.asarray(indices).reshape(-1)
        expected = np.clip(boxes[keep]/640.0, 0, 1)
        expected[:,2:] -= expected[:,:2]
        actual = det.detect(image)
        assert len(actual) == len(expected) > 0
        np.testing.assert_allclose([[d.bbox.x,d.bbox.y,d.bbox.w,d.bbox.h] for d in actual], expected, atol=2e-6)
        np.testing.assert_allclose([d.confidence for d in actual], scores[keep], atol=6e-5)
    finally:
        det.close()


def test_wrong_onnx_input_is_rejected(tmp_path):
    from tests.helpers.make_dummy_onnx import create_dummy_yolo_onnx
    model = create_dummy_yolo_onnx(tmp_path/'rgb.onnx')
    det = OnnxDetector(DetectorSettings(model_path=model, input_profile='ms5'), providers=['CPUExecutionProvider'])
    with pytest.raises(DetectionError, match='MS5 ожидает'):
        det.load()
    assert not det.is_loaded


def test_classifier_failure_is_not_silently_reported_as_plant(tmp_path):
    det, _ = synthetic_detector(tmp_path)
    def fail(*args):
        raise RuntimeError('classifier failure')
    det._classifier_session = SimpleNamespace(run=fail)
    with pytest.raises(DetectionError, match='классификации'):
        det.detect(np.full((100,200,5), .5, np.float32))


def test_standalone_main_window_import_has_no_cycle():
    import os
    import subprocess
    import sys
    env = dict(os.environ, PYTHONPATH=str(ROOT/'src'))
    subprocess.run([sys.executable, '-c', 'from geo_weed_station.ui.main_window import MainWindow'],
                   cwd=ROOT, env=env, check=True, capture_output=True, timeout=20)
