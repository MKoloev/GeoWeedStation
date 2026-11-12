"""Lazy model-format adapters. No annotations are read by any backend."""
import hashlib
import json
import platform
import tempfile
from pathlib import Path
import numpy as np
from pipeline import ROOT


def checked(code,operation):
    if code!=0:raise RuntimeError(f'{operation} failed: code {code}')


def verify_model(path):
    expected=json.loads((ROOT/'models.sha256.json').read_text()).get(path.name)
    if expected and hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise ValueError(f'Model checksum mismatch: {path}. Keep the matching weights and settings together.')


class PTBackend:
    max_batch=4
    def __init__(self,device='auto',threads=4,**kwargs):
        import torch
        from model_architecture import build_yolox_m
        torch.set_num_threads(threads)
        self.device=device if device!='auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        path=ROOT/'model.pt';verify_model(path)
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        self.model=build_yolox_m(pretrained=False)
        self.model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        self.model=self.model.to(self.device).eval();self.torch=torch
        print(f'PyTorch FP32: {self.device}',flush=True)
    def predict(self,rgb_tiles):
        t=self.torch.from_numpy(np.ascontiguousarray(rgb_tiles.transpose(0,3,1,2))).float()/255
        with self.torch.inference_mode():outputs=self.model(t.to(self.device))
        return [o.float().cpu().numpy() for o in outputs]
    def close(self):pass


class ONNXBackend:
    max_batch=1
    def __init__(self,device='auto',threads=4,**kwargs):
        import onnxruntime as ort
        path=ROOT/'model.onnx';verify_model(path)
        options=ort.SessionOptions();options.intra_op_num_threads=threads;options.inter_op_num_threads=1
        providers=['CPUExecutionProvider']
        if device=='cuda':
            if 'CUDAExecutionProvider' not in ort.get_available_providers():raise RuntimeError('ONNX Runtime CUDA provider is not installed')
            providers=['CUDAExecutionProvider','CPUExecutionProvider']
        self.session=ort.InferenceSession(str(path),sess_options=options,providers=providers)
        if device=='cuda' and 'CUDAExecutionProvider' not in self.session.get_providers():raise RuntimeError('ONNX CUDA initialization failed; CPU fallback was rejected')
        self.device='cuda' if 'CUDAExecutionProvider' in self.session.get_providers() else 'cpu'
        self.input_name=self.session.get_inputs()[0].name
        print(f'ONNX FP32: {self.session.get_providers()}',flush=True)
    def predict(self,rgb_tiles):
        x=np.ascontiguousarray(rgb_tiles.transpose(0,3,1,2),dtype=np.float32)/255
        return self.session.run(None,{self.input_name:x})
    def close(self):pass


class RKNNBackend:
    max_batch=1
    def __init__(self,runtime='auto',target='rk3588',device_id=None,**kwargs):
        path=ROOT/'model.rknn';verify_model(path);self.temp=None;self.api=None
        self.runtime=runtime if runtime!='auto' else ('lite' if platform.machine().lower() in ('aarch64','arm64') else 'simulator')
        try:
            if self.runtime=='lite':
                from rknnlite.api import RKNNLite
                self.api=RKNNLite();checked(self.api.load_rknn(str(path)),'load_rknn')
                checked(self.api.init_runtime(),'RKNNLite init_runtime')
            elif self.runtime=='toolkit':
                from rknn.api import RKNN
                self.api=RKNN(verbose=False);checked(self.api.load_rknn(str(path)),'load_rknn')
                checked(self.api.init_runtime(target=target,device_id=device_id),'Toolkit hardware init_runtime')
            elif self.runtime=='simulator':
                from rknn.api import RKNN
                onnx=ROOT/'model.onnx';verify_model(onnx)
                files=sorted((ROOT/'calibration').glob('*.png'))
                if len(files)!=200:raise FileNotFoundError('Simulator requires the included 200 calibration/*.png files')
                provenance=json.loads((ROOT/'calibration/provenance.json').read_text())
                expected={f'{row["index"]:04d}.png':row['sha256'] for row in provenance}
                for f in files:
                    if hashlib.sha256(f.read_bytes()).hexdigest()!=expected.get(f.name):
                        raise ValueError(f'Calibration checksum mismatch: {f.name}')
                self.temp=tempfile.TemporaryDirectory(prefix='best_rgb_rknn_sim_')
                temp=Path(self.temp.name);paths=[]
                for i,f in enumerate(files):
                    link=temp/f'{i:04d}.png';link.symlink_to(f.resolve());paths.append(str(link))
                listing=temp/'dataset.txt';listing.write_text('\n'.join(paths)+'\n')
                print('RKNN simulator: rebuilding the INT8 graph from ONNX and the original 200 calibration crops. This can take several minutes.',flush=True)
                self.api=RKNN(verbose=False)
                checked(self.api.config(target_platform=target,optimization_level=3,mean_values=[[0,0,0]],std_values=[[255,255,255]],
                            quantized_dtype='asymmetric_quantized-8'),'RKNN config')
                checked(self.api.load_onnx(model=str(onnx)),'load_onnx')
                checked(self.api.build(do_quantization=True,dataset=str(listing)),'INT8 build')
                checked(self.api.init_runtime(),'simulator init_runtime')
            else:raise ValueError('runtime must be auto, lite, toolkit or simulator')
            print(f'RKNN INT8 runtime: {self.runtime}',flush=True)
        except BaseException:
            self.close();raise
    def predict(self,rgb_tiles):
        # std=255 is inside the graph: no float conversion or division here.
        x=np.ascontiguousarray(rgb_tiles,dtype=np.uint8)
        if self.runtime=='lite':out=self.api.inference(inputs=[x])
        else:out=self.api.inference(inputs=[x],data_format=['nhwc'],inputs_pass_through=[0])
        if out is None:raise RuntimeError('RKNN inference returned no outputs')
        return out
    def close(self):
        if self.api is not None:self.api.release();self.api=None
        if self.temp is not None:self.temp.cleanup();self.temp=None


def make_backend(format,**kwargs):
    return {'pt':PTBackend,'onnx':ONNXBackend,'rknn':RKNNBackend}[format](**kwargs)
