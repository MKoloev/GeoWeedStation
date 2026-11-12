"""Annotations and evaluation live here, separately from the frame-only detector."""
import json
from pathlib import Path
import numpy as np
from pipeline import iou_matrix


def dataset_records(root):
    root=Path(root)
    coco=json.loads((root/'GAS_202608281553_032_mark_web/annotations_coco.json').read_text())
    boxes={r['id']:[] for r in coco['images']}
    for a in coco['annotations']:
        x,y,w,h=map(float,a['bbox'])
        if w<=0 or h<=0:raise ValueError(f'Invalid annotation {a}')
        boxes[a['image_id']].append([x,y,x+w,y+h])
    return [dict(id=i['id'],path=root/'GAS_202608281553_032_web'/i['file_name'],
                 declared_size=(i['width'],i['height']),boxes=boxes[i['id']]) for i in coco['images']]


def ground_truth(record,width,height):
    dw,dh=record['declared_size'];sx,sy=width/dw,height/dh
    if abs(sx-sy)>1e-6:raise ValueError(f'Nonuniform annotation scale: {record["path"]}')
    b=np.asarray(record['boxes'],np.float32).reshape(-1,4)*np.array([sx,sy,sx,sy],np.float32)
    b[:,[0,2]]=b[:,[0,2]].clip(0,width);b[:,[1,3]]=b[:,[1,3]].clip(0,height)
    return b


def assign(boxes,scores,gt,threshold=.5):
    iou=iou_matrix(boxes,gt);tp=np.zeros(len(boxes),bool);used=set()
    for i in np.argsort(-scores,kind='stable'):
        available=[j for j in range(len(gt)) if j not in used]
        if available:
            j=max(available,key=lambda j:iou[i,j])
            if iou[i,j]>=threshold:tp[i]=True;used.add(j)
    return tp


def ap_at(predictions,truth,threshold):
    total=sum(len(g) for g in truth);scores=[];tp=[]
    for (b,s),g in zip(predictions,truth):scores.extend(s);tp.extend(assign(b,s,g,threshold))
    if not total or not scores:return 0.
    scores=np.asarray(scores);order=np.argsort(-scores,kind='stable');tp=np.asarray(tp)[order].cumsum()
    recall=tp/total;precision=tp/np.arange(1,len(tp)+1)
    mrec=np.r_[0.,recall,1.];mpre=np.r_[1.,precision,0.]
    mpre=np.maximum.accumulate(mpre[::-1])[::-1];indices=np.where(mrec[1:]!=mrec[:-1])[0]
    return float(np.sum((mrec[indices+1]-mrec[indices])*mpre[indices+1]))


def evaluate(predictions,truth):
    total=sum(len(g) for g in truth);count=sum(len(b) for b,s in predictions)
    tp=sum(int(assign(b,s,g).sum()) for (b,s),g in zip(predictions,truth));fp=count-tp;fn=total-tp
    p=tp/max(count,1);r=tp/max(total,1)
    aps=[ap_at(predictions,truth,float(t)) for t in np.arange(.5,1,.05)]
    return dict(tp=tp,fp=fp,fn=fn,precision=p,recall=r,f1=2*tp/max(total+count,1),ap50=aps[0],map50_95=float(np.mean(aps)))
