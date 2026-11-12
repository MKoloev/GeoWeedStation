import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from pipeline import Detector,load_settings,positions,frame_gsd,fuse,nms
from metrics import evaluate,ground_truth


class EmptyBackend:
    max_batch=4
    def predict(self,tiles):
        self.first_pixel=tiles[0,0,0].tolist()
        return [np.full((len(tiles),6,h,h),-30,np.float32) for h in (80,40,20)]
    def close(self):pass


class BundleTests(unittest.TestCase):
    def test_exact_edge_grid(self):
        self.assertEqual(positions(8192,4211,.25),[0,3158,3981])
        self.assertEqual(positions(6144,4211,.25),[0,1933])

    def test_camera_rule_uses_only_frame_width(self):
        cfg=load_settings()
        self.assertAlmostEqual(frame_gsd(8192,cfg),.0002736)
        self.assertAlmostEqual(frame_gsd(1280,cfg),.0002736*6.4)
        cfg['gsd_policy']='fixed'
        self.assertAlmostEqual(frame_gsd(1280,cfg),.0002736)

    def test_frame_detector_never_opens_annotation_or_any_file(self):
        cfg=load_settings();cfg.update(gsd_policy='fixed',source_gsd=.0018)
        backend=EmptyBackend();detector=Detector(backend,cfg,progress=False)
        frame=np.zeros((640,640,3),np.uint8);frame[:]=[10,20,30]
        with patch('builtins.open',side_effect=AssertionError('Frame inference must not read files')):
            out=detector.detect(frame)
        self.assertEqual(out['boxes_xyxy'].shape,(0,4))
        self.assertEqual(backend.first_pixel,[30,20,10])

    def test_containment_preserves_separate_small_detection(self):
        cfg=load_settings()
        fine=(np.array([[10,10,20,20],[200,200,220,220]],np.float32),np.array([.9,.9],np.float32))
        coarse=(np.array([[0,0,100,100]],np.float32),np.array([.9],np.float32))
        b,s=fuse(fine,coarse,.0036,cfg)
        self.assertEqual(len(b),2)
        self.assertTrue(any(np.array_equal(v,[200,200,220,220]) for v in b))

    def test_empty_frame_fp_is_counted(self):
        p=[(np.array([[0,0,10,10]],np.float32),np.array([.9],np.float32)),
           (np.array([[0,0,10,10]],np.float32),np.array([.8],np.float32))]
        t=[np.array([[0,0,10,10]],np.float32),np.empty((0,4),np.float32)]
        m=evaluate(p,t)
        self.assertEqual((m['tp'],m['fp'],m['fn']),(1,1,0));self.assertAlmostEqual(m['f1'],2/3)

    def test_anomalous_gt_correction_is_evaluation_only(self):
        record={'declared_size':(8192,6144),'boxes':[[0,0,640,640]]}
        np.testing.assert_array_equal(ground_truth(record,1280,960),[[0,0,100,100]])

    def test_metrics_opens_gt_only_after_all_frame_predictions(self):
        import cv2
        import cli
        import metrics as metric_module
        calls=[]
        class FakeDetector:
            settings={}
            backend=SimpleNamespace(runtime=None,device='test')
            def detect(self,frame):
                calls.append('predict')
                return dict(boxes_xyxy=np.empty((0,4),np.float32),scores=np.empty(0,np.float32),image_size=[frame.shape[1],frame.shape[0]])
            def close(self):pass
        original=metric_module.dataset_records
        def read_gt(root):
            self.assertEqual(calls,['predict','predict'])
            calls.append('gt')
            return original(root)
        with tempfile.TemporaryDirectory(prefix='rgb_metric_gt_order_') as tmp:
            root=Path(tmp);images=root/'GAS_202608281553_032_web';images.mkdir()
            annotation=root/'GAS_202608281553_032_mark_web';annotation.mkdir()
            for name in ['a.png','b.png']:cv2.imwrite(str(images/name),np.zeros((10,10,3),np.uint8))
            coco=dict(images=[dict(id=i,file_name=n,width=10,height=10) for i,n in enumerate(['a.png','b.png'])],
                      annotations=[dict(image_id=0,bbox=[1,1,3,3])])
            (annotation/'annotations_coco.json').write_text(json.dumps(coco))
            with patch('sys.argv',['evaluate_pt.py','--dataset',str(root),'--output',str(root/'out')]), \
                 patch('cli.create_detector',return_value=FakeDetector()), \
                 patch('metrics.dataset_records',side_effect=read_gt):
                cli.main_evaluate('pt')
            report=json.loads((root/'out/metrics.json').read_text())
            self.assertEqual(report['metrics']['fn'],1)
            self.assertEqual(calls,['predict','predict','gt'])

if __name__=='__main__':unittest.main()
