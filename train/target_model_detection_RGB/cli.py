"""Six thin launchers share this CLI and exactly the same Detector.detect(frame)."""
import argparse
import json
import time
from pathlib import Path
import cv2
from tqdm import tqdm
from pipeline import ROOT,Detector,load_settings
from backends import make_backend

DEFAULT_DATASET='/media/workerfit/SP PHD U3/ТГУ/соревнования/Екатеринбург 2026/datasets/GeosAero_hitech_dataset'


def parser(format,evaluation):
    p=argparse.ArgumentParser(description=f'{format.upper()}: '+('fixed-parameter metrics' if evaluation else 'frame-only inference'))
    p.add_argument('--settings',type=Path,default=ROOT/f'settings_{format}.json')
    p.add_argument('--source-gsd',type=float,help='Actual input metres/pixel: disables reference-width scaling')
    if format=='rknn':
        p.add_argument('--runtime',default='auto',choices=['auto','lite','toolkit','simulator'])
        p.add_argument('--target',default='rk3588');p.add_argument('--device-id')
        p.set_defaults(device='auto',threads=4)
    else:
        p.add_argument('--device',default='auto',choices=['auto','cpu','cuda'])
        p.add_argument('--threads',type=int,default=4)
        p.set_defaults(runtime='auto',target='rk3588',device_id=None)
    if evaluation:
        p.add_argument('--dataset',type=Path,default=Path(DEFAULT_DATASET))
        p.add_argument('--output',type=Path,default=ROOT/'results'/f'metrics_{format}')
        p.add_argument('--limit',type=int,help='Debug subset only; omit for full metrics')
    else:
        p.add_argument('image',type=Path,help='Any input image; annotations are never opened')
        p.add_argument('--output',type=Path,default=ROOT/'results'/f'detection_{format}.jpg')
    return p


def create_detector(format,args):
    cfg=load_settings(args.settings)
    if args.source_gsd is not None:
        if args.source_gsd<=0:raise ValueError('--source-gsd must be positive')
        cfg['source_gsd']=args.source_gsd;cfg['gsd_policy']='fixed'
    print(f'Loading {format.upper()} model...',flush=True)
    backend=make_backend(format,device=args.device,threads=args.threads,runtime=args.runtime,target=args.target,device_id=args.device_id)
    return Detector(backend,cfg,progress=True)


def main_infer(format):
    a=parser(format,False).parse_args();detector=create_detector(format,a)
    try:
        frame=cv2.imread(str(a.image))
        if frame is None:raise OSError(f'Cannot read image: {a.image}')
        start=time.monotonic();result=detector.detect(frame);seconds=time.monotonic()-start
        for box,score in zip(result['boxes_xyxy'],result['scores']):
            x1,y1,x2,y2=map(round,box)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,f'plant {score:.3f}',(x1,max(16,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,0),2)
        a.output.parent.mkdir(parents=True,exist_ok=True)
        if not cv2.imwrite(str(a.output),frame):raise OSError(a.output)
        payload=dict(image=str(a.image.resolve()),format=format,image_size=result['image_size'],source_gsd=result['source_gsd'],
                     boxes_xyxy=result['boxes_xyxy'].tolist(),scores=result['scores'].tolist(),settings=detector.settings,seconds=seconds)
        a.output.with_suffix('.json').write_text(json.dumps(payload,indent=2))
        print(f'Saved (overwritten): {a.output}\nBoxes: {len(result["scores"])}; time: {seconds:.2f}s',flush=True)
    finally:detector.close()


def main_evaluate(format):
    # Enumerate input frames without consulting COCO or any GT metadata.
    a=parser(format,True).parse_args()
    image_dir=a.dataset/'GAS_202608281553_032_web'
    files=sorted(p for p in image_dir.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png','.tif','.tiff','.bmp'))
    if not files:raise ValueError(f'No images in {image_dir}')
    if a.limit is not None:
        if a.limit<1:raise ValueError('--limit must be positive')
        files=files[:a.limit]
    detector=create_detector(format,a);a.output.mkdir(parents=True,exist_ok=True)
    predictions=[];frames=[];start=time.monotonic()
    try:
        with (a.output/'predictions.jsonl').open('w') as stream:
            for path in tqdm(files,desc=f'{format.upper()}: dataset images',dynamic_ncols=True):
                frame=cv2.imread(str(path))
                if frame is None:raise OSError(path)
                # Exactly the same one-argument call as ordinary inference.
                result=detector.detect(frame)
                b,s=result['boxes_xyxy'],result['scores']
                predictions.append((b,s));frames.append((path,result['image_size']))
                stream.write(json.dumps(dict(image=str(path),boxes_xyxy=b.tolist(),scores=s.tolist()))+'\n')
        print('All frame predictions are complete. Reading annotations only to calculate metrics...',flush=True)
        from metrics import dataset_records,ground_truth,assign,evaluate
        by_name={r['path'].name:r for r in dataset_records(a.dataset)}
        truth=[];per_image=[]
        for (path,(width,height)),(b,s) in zip(frames,predictions):
            if path.name not in by_name:raise ValueError(f'No GT record for {path.name}; cannot score this image')
            rec=by_name[path.name];gt=ground_truth(rec,width,height);truth.append(gt)
            tp=int(assign(b,s,gt).sum())
            per_image.append(dict(image_id=rec['id'],image=str(path),tp=tp,fp=len(b)-tp,fn=len(gt)-tp))
        m=evaluate(predictions,truth)
        report=dict(format=format,runtime=getattr(detector.backend,'runtime',None),device=getattr(detector.backend,'device',a.device),
                    images=len(files),gt=sum(len(g) for g in truth),iou_threshold=.5,metrics=m,settings=detector.settings,
                    seconds=time.monotonic()-start,protocol='fixed settings; frame-only inference; GT read only after all predictions')
        (a.output/'metrics.json').write_text(json.dumps(report,indent=2));(a.output/'per_image.json').write_text(json.dumps(per_image,indent=2))
        print(f"F1={m['f1']*100:.2f}% | P={m['precision']*100:.2f}% R={m['recall']*100:.2f}% | TP={m['tp']} FP={m['fp']} FN={m['fn']}",flush=True)
        print('Saved:',a.output/'metrics.json',flush=True)
    finally:detector.close()
