"""Reproducible evaluation of original GeosAero frames and annotated plant crops.

No random sampling, generated background labels, training fallback, or threshold search.
Classification metrics use ground-truth crops, not detector proposals.
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.ops import nms
from tqdm import tqdm

from common.models.resnet import build_resnet50, build_resnet50_dual_head
from common.models.yolox import build_yolox_m
from common.metrics import box_iou_np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path('/media/workerfit/SP PHD U3/ТГУ/соревнования/Екатеринбург 2026/datasets/GeosAero_hitech_dataset')
CATEGORIES = ['background', 'Осот, Бодяк, Латук', 'Вьюнок', 'Щирица (виды)',
              'Маревые (Марь, Лебеда, Кохия, Солянка)', 'Канатник, просвирник, гибискус',
              'Крестоцветные (Капустные)', 'Горец (виды)', 'Злаковые', 'Полынь, Ромашка', 'Гвоздичные']
STAGES = ['background', 'всходы', '2-4 листа', 'взрослое растение', 'переросшие']
STAGE_MAP = {'1-я_пара': 1, '1-я_пара листьев': 1, 'появление_отпрысков': 1,
             '2-я_пара': 2, '2-я_пара листьев': 2, '3-5_пара': 2, '3-5_пара листьев': 2,
             '3-5_лист': 2, 'розетка': 3, 'возобновление_вегетации': 3,
             'переросшие': 4, 'переросший': 4}
# Species output index -> compatible GeosAero category. This does not infer species GT.
SPECIES_GROUP = [0, 8, 8, 8, 8, 8, 4, 4, 4, 11, 9, 11, 11, 2, 5, 11, 6, 11, 11, 11, 11, 11, 11, 11]
# Only these explicitly annotated species are in the C06 vocabulary.
EXACT_SPECIES = {'Вьюнок полевой': 13}
SPECIES_NAMES = ['S000: Background (Фон / Не растение)', 'S001: Zea mays (Кукуруза сахарная)', 'S002: Lolium rigidum (Плевел жесткий)', 'S003: Sorghum halepense (Гумай (сорго алеппское))', 'S004: Avena fatua (Овсюг обыкновенный)', 'S005: Hordeum vulgare (Ячмень обыкновенный)', 'S006: Atriplex patula (Лебеда раскидистая)', 'S007: Chenopodium album (Марь белая)', 'S008: Salsola kali (Солянка русская (курай))', 'S009: Beta vulgaris (Свёкла обыкновенная)', 'S010: Matricaria discoidea (Ромашка пахучая (безъязычковая))', 'S011: Onopordum acanthium (Татарник колючий)', 'S012: Silybum marianum (Расторопша пятнистая)', 'S013: Convolvulus arvensis (Вьюнок полевой (берёзка))', 'S014: Malva sylvestris (Мальва лесная (просвирник))', 'S015: Gossypium hirsutum (Хлопчатник обыкновенный)', 'S016: Sinapis arvensis (Горчица полевая)', 'S017: Cyperus rotundus (Сыть круглая)', 'S018: Solanum lycopersicum (Томат обыкновенный)', 'S019: Solanum nigrum (Паслён чёрный)', 'S020: Solanum elaeagnifolium (Паслён линейнолистный)', 'S021: Datura ferox (Дурман свирепый)', 'S022: Datura stramonium (Дурман обыкновенный)', 'S023: Portulaca oleracea (Портулак огородный)']


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def classification_metrics(targets, probabilities, names):
    y = np.asarray(targets, dtype=int)
    prob = np.asarray(probabilities, dtype=float)
    if not len(y):
        return {'samples': 0, 'status': 'no evaluable labels'}
    pred = prob.argmax(1)
    n = len(names)
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    tp = cm.diagonal().astype(float)
    support, predicted = cm.sum(1), cm.sum(0)
    precision = np.divide(tp, predicted, out=np.zeros(n), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(n), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(n), where=precision + recall > 0)
    present = support > 0
    acc = float((pred == y).mean())
    expected = float(np.dot(support.astype(float), predicted) / len(y)**2)
    mcc_den = float(np.sqrt((len(y)**2 - np.dot(predicted.astype(float), predicted)) *
                            (len(y)**2 - np.dot(support.astype(float), support))))
    result = dict(samples=len(y), correct=int((pred == y).sum()), accuracy=acc,
                  error_rate=1-acc, balanced_accuracy=float(recall[present].mean()),
                  macro_policy='mean over ground-truth-present classes; zero_division=0',
                  macro_precision=float(precision[present].mean()), macro_recall=float(recall[present].mean()),
                  macro_f1=float(f1[present].mean()), macro_f1_all_output_classes=float(f1.mean()),
                  weighted_precision=float(precision @ support / len(y)),
                  weighted_recall=float(recall @ support / len(y)), weighted_f1=float(f1 @ support / len(y)),
                  micro_precision=acc, micro_recall=acc, micro_f1=acc,
                  cohen_kappa=(acc-expected)/(1-expected) if expected < 1 else None,
                  mcc=float((tp.sum()*len(y)-np.dot(support.astype(float), predicted))/mcc_den) if mcc_den else None,
                  log_loss=float(-np.log(np.clip(prob[np.arange(len(y)), y], 1e-15, 1)).mean()),
                  brier_multiclass=float(((prob-np.eye(n)[y])**2).sum(1).mean()),
                  confusion_matrix=cm.tolist(), class_names=names,
                  per_class={name: dict(precision=float(precision[i]), recall=float(recall[i]), f1=float(f1[i]),
                                        support=int(support[i]), tp=int(tp[i]), fp=int(predicted[i]-tp[i]),
                                        fn=int(support[i]-tp[i])) for i, name in enumerate(names)})
    for k in (2, 3, 5):
        if k <= n:
            result[f'top_{k}_accuracy'] = float((np.argsort(-prob, axis=1, kind='stable')[:, :k] == y[:, None]).any(1).mean())
    return result


def match_detections(predictions, ground_truth, threshold):
    scores, matched = [], []
    for (boxes, conf), gt in zip(predictions, ground_truth):
        order = np.argsort(-conf, kind='stable')
        overlaps = box_iou_np(boxes[order], gt)
        used = np.zeros(len(gt), dtype=bool)
        for i, idx in enumerate(order):
            available = np.where(used, -1, overlaps[i])
            best = int(available.argmax()) if len(gt) else -1
            ok = best >= 0 and available[best] >= threshold
            if ok:
                used[best] = True
            scores.append(float(conf[idx]))
            matched.append(int(ok))
    order = np.argsort(-np.asarray(scores), kind='stable')
    return np.asarray(scores)[order], np.asarray(matched, dtype=int)[order]


def detection_metrics(predictions, ground_truth, confidence):
    total = sum(map(len, ground_truth))
    result = dict(images=len(ground_truth), gt=total, confidence_threshold=confidence,
                  matching='score-ordered one-to-one greedy; IoU >= threshold',
                  ap_protocol='all-point precision envelope and 101 recall points, no maxDets truncation',
                  ap_by_iou={}, ap101_by_iou={})
    for threshold in np.round(np.arange(.5, 1, .05), 2):
        scores, tp = match_detections(predictions, ground_truth, threshold)
        ct = tp.cumsum()
        recalls = ct / max(total, 1)
        precisions = ct / np.maximum(np.arange(1, len(tp)+1), 1)
        mr = np.r_[0, recalls, 1]
        mp = np.r_[0, precisions, 0]
        mp = np.maximum.accumulate(mp[::-1])[::-1]
        indices = np.where(mr[1:] != mr[:-1])[0]
        ap = float(((mr[indices+1]-mr[indices])*mp[indices+1]).sum())
        ap101 = float(np.mean([precisions[recalls >= r].max(initial=0) for r in np.linspace(0, 1, 101)]))
        result['ap_by_iou'][f'{threshold:.2f}'] = ap
        result['ap101_by_iou'][f'{threshold:.2f}'] = ap101
        if threshold == .5:
            mask = scores >= confidence
            true_pos = int(tp[mask].sum())
            false_pos = int(mask.sum()) - true_pos
            false_neg = total - true_pos
            p = true_pos / max(true_pos + false_pos, 1)
            r = true_pos / max(total, 1)
            result.update(tp=true_pos, fp=false_pos, fn=false_neg, precision=p, recall=r,
                          f1=2*p*r/(p+r) if p+r else 0, predictions=int(mask.sum()))
    result.update(ap50=result['ap_by_iou']['0.50'], ap75=result['ap_by_iou']['0.75'],
                  map50_95=float(np.mean(list(result['ap_by_iou'].values()))),
                  map101_50_95=float(np.mean(list(result['ap101_by_iou'].values()))))
    return result


@torch.inference_mode()
def detect(model, rgb, args):
    h, w = rgb.shape[:2]
    if args.det_mode == 'resize':
        patches = [(cv2.resize(rgb, (640, 640)), (0, 0, w/640, h/640))]
    else:
        # Normal 8192px frame -> ~1229px at the training reference GSD.
        scale = args.gsd_scale * 8192 / w
        sw, sh = max(1, round(w*scale)), max(1, round(h*scale))
        scaled = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        positions = lambda size: sorted(set(list(range(0, max(size-640, 0)+1, 512)) + [max(size-640, 0)]))
        patches = []
        for y in positions(sh):
            for x in positions(sw):
                patch = scaled[y:y+640, x:x+640]
                patch = cv2.copyMakeBorder(patch, 0, 640-patch.shape[0], 0, 640-patch.shape[1], cv2.BORDER_CONSTANT, value=0)
                patches.append((patch, (x* w/sw, y*h/sh, w/sw, h/sh)))
    boxes, scores = [], []
    for patch, (ox, oy, sx, sy) in patches:
        tensor = torch.from_numpy(np.ascontiguousarray(patch.transpose(2, 0, 1))).float()[None].to(args.device)/255
        outputs = model(tensor)
        for out, stride in zip(outputs, (8, 16, 32)):
            p = out[0]
            conf = p[4].sigmoid()*p[5].sigmoid()
            mask = conf >= args.score_floor
            yy, xx = torch.meshgrid(torch.arange(p.shape[1], device=p.device), torch.arange(p.shape[2], device=p.device), indexing='ij')
            reg = p[:4, mask].T
            xy = (reg[:, :2] + torch.stack((xx[mask], yy[mask]), 1))*stride
            wh = reg[:, 2:].clamp(-16, 16).exp()*stride
            b = torch.cat((xy-wh/2, xy+wh/2), 1)
            b = b * b.new_tensor([sx, sy, sx, sy]) + b.new_tensor([ox, oy, ox, oy])
            b[:, [0, 2]] = b[:, [0, 2]].clamp(0, w)
            b[:, [1, 3]] = b[:, [1, 3]].clamp(0, h)
            good = torch.isfinite(b).all(1) & ((b[:, 2:]-b[:, :2]) > 0).all(1)
            boxes.append(b[good]); scores.append(conf[mask][good])
    boxes, scores = torch.cat(boxes), torch.cat(scores)
    keep = nms(boxes, scores, args.nms_iou)
    return boxes[keep].cpu().numpy(), scores[keep].cpu().numpy()


def main(model_id):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--checkpoint', type=Path, default=ROOT.parent/'runs'/model_id/'best.pt')
    parser.add_argument('--output-dir', type=Path, default=ROOT/model_id/'geosaero_evaluation')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--confidence', type=float, default=.25)
    parser.add_argument('--score-floor', type=float, default=.001)
    parser.add_argument('--nms-iou', type=float, default=.5)
    parser.add_argument('--det-mode', choices=['resize', 'tiles'], default='resize')
    parser.add_argument('--gsd-scale', type=float, default=.15)
    args = parser.parse_args()
    if not 0 <= args.score_floor <= args.confidence <= 1:
        parser.error('Require 0 <= score-floor <= confidence <= 1')
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    torch.manual_seed(0)
    mark = args.dataset/'GAS_202608281553_032_mark_web'
    coco_file = mark/'annotations_coco.json'
    csv_file = mark/'detections.csv'
    coco = json.loads(coco_file.read_text())
    rows = list(csv.DictReader(csv_file.open(encoding='utf-8')))
    rows_by_image, annotations = defaultdict(list), defaultdict(list)
    for i, row in enumerate(rows):
        row['row_id'] = i
        rows_by_image[row['image_file']].append(row)
    for ann in coco['annotations']:
        annotations[ann['image_id']].append(ann)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if model_id.startswith('d01'):
        model = build_yolox_m(in_channels=3, num_classes=1, pretrained=False)
    elif model_id.startswith('c06'):
        model = build_resnet50(num_classes=24, pretrained=False)
        if ckpt.get('species_to_idx', {f'S{i:03d}': i for i in range(24)}) != {f'S{i:03d}': i for i in range(24)}:
            raise ValueError('Unexpected species output mapping')
    else:
        model = build_resnet50_dual_head(num_category_classes=11, num_stage_classes=5, pretrained=False)
        if ckpt.get('category_to_idx', dict(zip(CATEGORIES, range(11)))) != dict(zip(CATEGORIES, range(11))):
            raise ValueError('Unexpected category output mapping')
        if ckpt.get('stage_to_idx', dict(zip(STAGES, range(5)))) != dict(zip(STAGES, range(5))):
            raise ValueError('Unexpected stage output mapping')
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.to(args.device).eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize([.485,.456,.406], [.229,.224,.225])])
    predictions, gt, records, probabilities, stage_probabilities = [], [], [], [], []
    sizes, input_digest = [], hashlib.sha256()
    started = time.time()
    with torch.inference_mode(), (args.output_dir/'predictions.jsonl').open('w', encoding='utf-8') as output:
        for image_meta in tqdm(coco['images'], desc=model_id):
            path = args.dataset/'GAS_202608281553_032_web'/Path(image_meta['file_name']).name
            input_digest.update(path.name.encode()); input_digest.update(bytes.fromhex(sha256(path)))
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise OSError(f'Cannot read {path}')
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            sx, sy = w/image_meta['width'], h/image_meta['height']
            sizes.append(dict(image=path.name, actual=[w,h], annotation=[image_meta['width'], image_meta['height']]))
            if model_id.startswith('d01'):
                boxes = []
                for ann in annotations[image_meta['id']]:
                    x,y,bw,bh = ann['bbox']
                    boxes.append([max(0,x*sx), max(0,y*sy), min(w,(x+bw)*sx), min(h,(y+bh)*sy)])
                gt.append(np.asarray(boxes, dtype=np.float32).reshape(-1,4))
                pb, ps = detect(model, rgb, args)
                predictions.append((pb, ps))
                output.write(json.dumps(dict(image=path.name, boxes_xyxy=pb.tolist(), scores=ps.tolist()))+'\n')
            else:
                image_rows = rows_by_image[path.name]
                if not image_rows:
                    continue
                crops = []
                for row in image_rows:
                    x,y,bw,bh = [float(row[k]) for k in ('x','y','w','h')]
                    x1,y1,x2,y2 = round(x*sx),round(y*sy),round((x+bw)*sx),round((y+bh)*sy)
                    x1,y1 = max(0,min(w-1,x1)),max(0,min(h-1,y1))
                    x2,y2 = max(x1+1,min(w,x2)),max(y1+1,min(h,y2))
                    crops.append(tf(Image.fromarray(rgb[y1:y2,x1:x2])))
                batch = torch.stack(crops).to(args.device)
                raw = model(batch)
                if model_id.startswith('c06'):
                    probs = raw.softmax(1).cpu().numpy()
                    stage_probs = [None]*len(probs)
                else:
                    probs = raw[0].softmax(1).cpu().numpy()
                    stage_probs = raw[1].softmax(1).cpu().numpy()
                for row, prob, sp in zip(image_rows, probs, stage_probs):
                    records.append(row); probabilities.append(prob)
                    item = dict(row_id=row['row_id'], image=path.name, label=row['label_name'],
                                species=row['Вид'], stage=row['Фаза роста'], probabilities=prob.tolist())
                    if sp is not None:
                        stage_probabilities.append(sp)
                        item['stage_probabilities'] = sp.tolist()
                    output.write(json.dumps(item, ensure_ascii=False)+'\n')
    report = dict(model_id=model_id, checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha256(args.checkpoint),
                  checkpoint_epoch=ckpt.get('epoch'), dataset=str(args.dataset.resolve()),
                  annotations_sha256=sha256(coco_file), detections_csv_sha256=sha256(csv_file),
                  image_content_manifest_sha256=input_digest.hexdigest(),
                  images=len(coco['images']), annotated_objects=len(coco['annotations']),
                  protocol='all original dataset images; not a held-out test; possible training overlap',
                  device=args.device, seconds=time.time()-started, input_sizes=sizes)
    if model_id.startswith('d01'):
        report.update(preprocessing=args.det_mode, resolution=640, score_floor=args.score_floor,
                      nms_iou=args.nms_iou, gsd_scale=args.gsd_scale if args.det_mode == 'tiles' else None,
                      annotation_completeness='partial_plant: FP and precision are relative to supplied annotation',
                      metrics=detection_metrics(predictions, gt, args.confidence))
    else:
        probs = np.asarray(probabilities)
        targets = [CATEGORIES.index(r['label_name']) for r in records]
        report['preprocessing'] = 'GT bbox crop, PIL bilinear resize 224x224, RGB /255, ImageNet normalization; no synthetic background'
        if model_id.startswith('c06'):
            # Map the top-1 species prediction; summing group scores would change the classifier decision.
            group_pred = np.asarray(SPECIES_GROUP)[probs.argmax(1)]
            group_names = CATEGORIES + ['outside GeosAero active categories']
            group_onehot = np.eye(len(group_names))[group_pred]
            gm = classification_metrics(targets, group_onehot, group_names)
            for key in ['log_loss','brier_multiclass','top_2_accuracy','top_3_accuracy','top_5_accuracy']:
                gm.pop(key, None)
            report['category_projection'] = gm
            valid = [i for i,r in enumerate(records) if r['Вид'] in EXACT_SPECIES]
            report['species_metrics'] = classification_metrics([EXACT_SPECIES[records[i]['Вид']] for i in valid], probs[valid], SPECIES_NAMES)
            report['species_coverage'] = dict(evaluable=len(valid), unspecified=sum(r['Вид']=='NotSpecified' for r in records),
                out_of_vocabulary=[dict(row_id=r['row_id'], species=r['Вид']) for r in records if r['Вид']!='NotSpecified' and r['Вид'] not in EXACT_SPECIES])
            report['species_to_category'] = SPECIES_GROUP
        else:
            starget = [STAGE_MAP[r['Фаза роста']] for r in records]
            sp = np.asarray(stage_probabilities)
            report['category_metrics'] = classification_metrics(targets, probs, CATEGORIES)
            report['stage_metrics'] = classification_metrics(starget, sp, STAGES)
            report['joint_accuracy'] = float(((probs.argmax(1)==targets) & (sp.argmax(1)==starget)).mean())
            report['mean_head_accuracy'] = (report['category_metrics']['accuracy']+report['stage_metrics']['accuracy'])/2
    dump(args.output_dir/'metrics.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('input_sizes',)}, ensure_ascii=False, indent=2), flush=True)
    print('Saved:', args.output_dir/'metrics.json', flush=True)
