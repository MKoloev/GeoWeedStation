"""Frame-only RGB inference. This module never reads annotations or image paths."""
import json
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

ROOT=Path(__file__).resolve().parent


def load_settings(path=None):
    return json.loads(Path(path or ROOT/'settings_pt.json').read_text())


def iou_matrix(a,b):
    a=np.asarray(a,np.float32).reshape(-1,4);b=np.asarray(b,np.float32).reshape(-1,4)
    intersection=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2])).prod(2)
    area_a=np.maximum(0,a[:,2:]-a[:,:2]).prod(1);area_b=np.maximum(0,b[:,2:]-b[:,:2]).prod(1)
    return intersection/np.maximum(area_a[:,None]+area_b[None]-intersection,1e-12)


def nms(boxes,scores,threshold):
    order=np.argsort(-scores,kind='stable');kept=[]
    while len(order):
        i=order[0];kept.append(i)
        if len(order)==1:break
        rest=order[1:];overlap=iou_matrix(boxes[i:i+1],boxes[rest])[0]
        order=rest[overlap<=threshold]
    return np.asarray(kept,np.int64)


def filter_nms(boxes,scores,threshold):
    valid=np.isfinite(boxes).all(1)&np.isfinite(scores)&((boxes[:,2:]-boxes[:,:2])>0).all(1)
    boxes=boxes[valid];scores=scores[valid];keep=nms(boxes,scores,threshold)
    return boxes[keep],scores[keep]


def positions(length,size,overlap):
    end=max(0,length-size);step=max(1,round(size*(1-overlap)))
    return sorted(set(list(range(0,end+1,step))+[end]))


def frame_gsd(width,settings):
    # Fixed camera calibration; no filename, image id or annotation lookup.
    if settings['gsd_policy']=='fixed':return float(settings['source_gsd'])
    if settings['gsd_policy']=='reference_width':return float(settings['source_gsd'])*settings['reference_width']/width
    raise ValueError('gsd_policy must be fixed or reference_width')


def windows(width,height,size,overlap):
    return [(x,y,size) for y in positions(height,size,overlap) for x in positions(width,size,overlap)]


def tile_rgb(rgb,x,y,size,settings):
    crop=rgb[y:y+size,x:x+size]
    padded=np.full((size,size,3),settings['pad_value'],np.uint8)
    padded[:crop.shape[0],:crop.shape[1]]=crop
    return cv2.resize(padded,(settings['resolution'],settings['resolution']),interpolation=cv2.INTER_LINEAR)


def canonical_heads(outputs):
    heads=[]
    for raw in outputs:
        a=np.asarray(raw)
        if not np.issubdtype(a.dtype,np.floating):
            raise TypeError('Backend returned integer outputs. Request dequantized float outputs; do not decode native int8 values.')
        if a.ndim==3:a=a[None]
        if a.ndim!=4:raise ValueError(f'Unsupported head shape: {a.shape}')
        if a.shape[1]==6:pass
        elif a.shape[-1]==6:a=a.transpose(0,3,1,2)
        else:raise ValueError(f'Expected 6 raw channels, got {a.shape}')
        heads.append(a.astype(np.float32,copy=False))
    heads.sort(key=lambda a:-a.shape[2])
    if len(heads)!=3 or [a.shape[2:] for a in heads]!=[(80,80),(40,40),(20,20)]:
        raise ValueError(f'Expected raw stride 8/16/32 heads, got {[a.shape for a in heads]}')
    return heads


def sigmoid(x):
    return (1/(1+np.exp(-np.clip(x,-80,80)))).astype(np.float32)


def decode(outputs,floor):
    heads=canonical_heads(outputs);decoded=[]
    for batch in range(heads[0].shape[0]):
        boxes=[];scores=[]
        for out,stride in zip(heads,(8,16,32)):
            a=out[batch];h,w=a.shape[1:]
            score=sigmoid(a[4])*sigmoid(a[5]);mask=score>=floor
            y,x=np.meshgrid(np.arange(h,dtype=np.float32),np.arange(w,dtype=np.float32),indexing='ij')
            reg=a[:4,mask].T;grid=np.stack((x[mask],y[mask]),1)
            xy=(reg[:,:2]+grid)*stride;wh=np.exp(np.clip(reg[:,2:],-16,16))*stride
            boxes.append(np.concatenate((xy-wh/2,xy+wh/2),1));scores.append(score[mask])
        decoded.append((np.concatenate(boxes).astype(np.float32),np.concatenate(scores).astype(np.float32)))
    return decoded


def fuse(fine,coarse,gsd,cfg):
    fb,fs=fine;cb,cs=coarse;eps=cfg.get('score_epsilon',0)
    fi=np.where(fs>=cfg['fine_confidence']-eps)[0];ci=np.where(cs>=cfg['coarse_confidence']-eps)[0]
    if cfg['containment']>0 and len(fi) and len(ci):
        c=cb[ci];f=fb[fi];ca=np.prod(c[:,2:]-c[:,:2],1);fa=np.prod(f[:,2:]-f[:,:2],1)
        large=(c[:,2:]-c[:,:2]).max(1)*gsd/cfg['coarse_gsd']>=cfg['min_large_px']
        inter=np.maximum(0,np.minimum(c[:,None,2:],f[None,:,2:])-np.maximum(c[:,None,:2],f[None,:,:2])).prod(2)
        remove=((inter/np.maximum(fa[None],1e-9)>=cfg['containment']) &
                (fa[None]<=ca[:,None]*cfg['max_area_ratio']) & large[:,None]).any(0)
        fi=fi[~remove]
    c=cb[ci];f=fb[fi];cscore=cs[ci];fscore=fs[fi]
    mode=cfg['box_fusion']
    if len(c) and len(f) and mode!='nms':
        overlap=iou_matrix(c,f)
        if mode=='coarse_first':
            mask=overlap.max(0)<cfg['global_nms'];f=f[mask];fscore=fscore[mask]
        elif mode=='weighted':
            used=set();new_boxes=[];new_scores=[]
            for i in np.argsort(-cscore,kind='stable'):
                available=[j for j in range(len(f)) if j not in used]
                j=max(available,key=lambda j:overlap[i,j]) if available else -1
                if j>=0 and overlap[i,j]>=cfg['global_nms']:
                    new_boxes.append((c[i]*cscore[i]+f[j]*fscore[j])/max(cscore[i]+fscore[j],1e-9))
                    new_scores.append(max(cscore[i],fscore[j]));used.add(j)
                else:new_boxes.append(c[i]);new_scores.append(cscore[i])
            available=[j for j in range(len(f)) if j not in used]
            c=np.asarray(new_boxes,np.float32).reshape(-1,4);cscore=np.asarray(new_scores,np.float32)
            f=f[available];fscore=fscore[available]
        else:raise ValueError(f'Unknown fusion mode: {mode}')
    return filter_nms(np.concatenate((c,f)),np.concatenate((cscore,fscore)),cfg['global_nms'])


class Detector:
    def __init__(self,backend,settings,progress=True):
        self.backend=backend;self.settings=dict(settings);self.progress=progress

    def _scale(self,rgb,gsd,target_gsd,name):
        cfg=self.settings;h,w=rgb.shape[:2];size=max(1,round(cfg['resolution']*target_gsd/gsd))
        coords=windows(w,h,size,cfg['overlap']);all_boxes=[];all_scores=[]
        batch_size=min(cfg['inference_batch_size'],self.backend.max_batch)
        bar=tqdm(total=len(coords),desc=f'{name}: tiles {size}px -> 640',disable=not self.progress,dynamic_ncols=True,mininterval=1)
        try:
            for start in range(0,len(coords),batch_size):
                group=coords[start:start+batch_size]
                tiles=np.stack([tile_rgb(rgb,*c,cfg) for c in group])
                for (b,s),(x,y,size) in zip(decode(self.backend.predict(tiles),cfg['candidate_floor']),group):
                    b,s=filter_nms(b,s,cfg['preliminary_tile_nms'])
                    b*=size/cfg['resolution'];b+=np.array([x,y,x,y],np.float32)
                    b[:,[0,2]]=b[:,[0,2]].clip(0,w);b[:,[1,3]]=b[:,[1,3]].clip(0,h)
                    all_boxes.append(b);all_scores.append(s);bar.update(1)
        finally:bar.close()
        return filter_nms(np.concatenate(all_boxes),np.concatenate(all_scores),cfg['local_nms'])

    def detect(self,frame_bgr):
        """Only a decoded frame enters this method; GT and file paths are absent."""
        if frame_bgr is None or frame_bgr.ndim!=3 or frame_bgr.shape[2]!=3 or frame_bgr.dtype!=np.uint8:
            raise ValueError('Expected a nonempty uint8 BGR HxWx3 frame')
        h,w=frame_bgr.shape[:2]
        if h==0 or w==0:raise ValueError('Empty image')
        gsd=frame_gsd(w,self.settings);rgb=cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB)
        fine=self._scale(rgb,gsd,self.settings['fine_gsd'],'fine')
        coarse=self._scale(rgb,gsd,self.settings['coarse_gsd'],'coarse')
        b,s=fuse(fine,coarse,gsd,self.settings)
        return dict(boxes_xyxy=b,scores=s,image_size=[w,h],source_gsd=gsd,candidates={'fine':fine,'coarse':coarse})

    def close(self):self.backend.close()
