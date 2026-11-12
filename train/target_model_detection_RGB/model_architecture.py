import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional


def get_activation(name="silu"):
    if name == "silu":
        return nn.SiLU(inplace=True)
    elif name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "lrelu":
        return nn.LeakyReLU(0.1, inplace=True)
    else:
        raise AttributeError(f"Unsupported act {name}")


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5, act="silu"):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class SPPBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), act="silu"):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes])
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=act)

    def forward(self, x):
        x = self.conv1(x)
        features = [x] + [m(x) for m in self.m]
        return self.conv2(torch.cat(features, dim=1))


class CSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, act="silu"):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, act=act) for _ in range(n)]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class CSPDarknet(nn.Module):
    def __init__(self, dep_mul=0.67, wid_mul=0.75, in_channels=3, act="silu"):
        super().__init__()
        base_channels = int(wid_mul * 64)
        base_depth = max(round(dep_mul * 3), 1)

        # Standard 3x3 Conv stride 2 (hardware NPU friendly)
        self.stem = BaseConv(in_channels, base_channels, 3, stride=2, act=act)

        # Dark2: stride 4
        self.dark2 = nn.Sequential(
            BaseConv(base_channels, base_channels * 2, 3, stride=2, act=act),
            CSPLayer(base_channels * 2, base_channels * 2, n=base_depth, act=act),
        )

        # Dark3: stride 8
        self.dark3 = nn.Sequential(
            BaseConv(base_channels * 2, base_channels * 4, 3, stride=2, act=act),
            CSPLayer(base_channels * 4, base_channels * 4, n=base_depth * 3, act=act),
        )

        # Dark4: stride 16
        self.dark4 = nn.Sequential(
            BaseConv(base_channels * 4, base_channels * 8, 3, stride=2, act=act),
            CSPLayer(base_channels * 8, base_channels * 8, n=base_depth * 3, act=act),
        )

        # Dark5: stride 32
        self.dark5 = nn.Sequential(
            BaseConv(base_channels * 8, base_channels * 16, 3, stride=2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, act=act),
            CSPLayer(base_channels * 16, base_channels * 16, n=base_depth, shortcut=False, act=act),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        out3 = self.dark3(x)   # stride 8
        out4 = self.dark4(out3) # stride 16
        out5 = self.dark5(out4) # stride 32
        return out3, out4, out5


class YOLOPAFPN(nn.Module):
    def __init__(self, dep_mul=0.67, wid_mul=0.75, in_channels=3, act="silu"):
        super().__init__()
        self.backbone = CSPDarknet(dep_mul, wid_mul, in_channels=in_channels, act=act)
        base_channels = int(wid_mul * 64)
        base_depth = max(round(dep_mul * 3), 1)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # Top-down
        self.lateral_conv0 = BaseConv(base_channels * 16, base_channels * 8, 1, 1, act=act)
        self.C3_p4 = CSPLayer(base_channels * 16, base_channels * 8, n=base_depth, shortcut=False, act=act)
        self.reduce_conv1 = BaseConv(base_channels * 8, base_channels * 4, 1, 1, act=act)
        self.C3_p3 = CSPLayer(base_channels * 8, base_channels * 4, n=base_depth, shortcut=False, act=act)

        # Bottom-up
        self.bu_conv2 = BaseConv(base_channels * 4, base_channels * 4, 3, 2, act=act)
        self.C3_n3 = CSPLayer(base_channels * 8, base_channels * 8, n=base_depth, shortcut=False, act=act)
        self.bu_conv1 = BaseConv(base_channels * 8, base_channels * 8, 3, 2, act=act)
        self.C3_n4 = CSPLayer(base_channels * 16, base_channels * 16, n=base_depth, shortcut=False, act=act)

    def forward(self, x):
        out3, out4, out5 = self.backbone(x)

        # Top-down
        fpn_out0 = self.lateral_conv0(out5)
        f_out0 = self.upsample(fpn_out0)
        f_out0 = torch.cat([f_out0, out4], dim=1)
        f_out0 = self.C3_p4(f_out0)

        fpn_out1 = self.reduce_conv1(f_out0)
        f_out1 = self.upsample(fpn_out1)
        f_out1 = torch.cat([f_out1, out3], dim=1)
        pan_out2 = self.C3_p3(f_out1) # stride 8

        # Bottom-up
        p_online1 = self.bu_conv2(pan_out2)
        p_online1 = torch.cat([p_online1, fpn_out1], dim=1)
        pan_out1 = self.C3_n3(p_online1) # stride 16

        p_online0 = self.bu_conv1(pan_out1)
        p_online0 = torch.cat([p_online0, fpn_out0], dim=1)
        pan_out0 = self.C3_n4(p_online0) # stride 32

        return pan_out2, pan_out1, pan_out0


class YOLOXHead(nn.Module):
    """
    Decoupled Head optimized for RKNN NPU deployment:
    In export mode: returns [out_stride8, out_stride16, out_stride32]
    In training mode: returns dictionary or tuple of raw feature map predictions.
    """
    def __init__(self, num_classes=1, wid_mul=0.75, act="silu", strides=(8, 16, 32)):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        base_channels = int(wid_mul * 64)
        in_channel_list = [base_channels * 4, base_channels * 8, base_channels * 16]
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()

        feat_channels = int(256 * wid_mul)

        for in_c in in_channel_list:
            self.stems.append(BaseConv(in_c, feat_channels, 1, 1, act=act))
            self.cls_convs.append(nn.Sequential(
                BaseConv(feat_channels, feat_channels, 3, 1, act=act),
                BaseConv(feat_channels, feat_channels, 3, 1, act=act),
            ))
            self.reg_convs.append(nn.Sequential(
                BaseConv(feat_channels, feat_channels, 3, 1, act=act),
                BaseConv(feat_channels, feat_channels, 3, 1, act=act),
            ))
            self.cls_preds.append(nn.Conv2d(feat_channels, num_classes, 1, 1, 0))
            self.reg_preds.append(nn.Conv2d(feat_channels, 4, 1, 1, 0))
            self.obj_preds.append(nn.Conv2d(feat_channels, 1, 1, 1, 0))

    def forward(self, feats):
        outputs = []
        for i, feat in enumerate(feats):
            feat = self.stems[i](feat)
            cls_feat = self.cls_convs[i](feat)
            reg_feat = self.reg_convs[i](feat)

            cls_output = self.cls_preds[i](cls_feat) # [B, num_classes, H, W]
            reg_output = self.reg_preds[i](reg_feat) # [B, 4, H, W]
            obj_output = self.obj_preds[i](reg_feat) # [B, 1, H, W]

            # [B, 4 + 1 + num_classes, H, W]
            out = torch.cat([reg_output, obj_output, cls_output], dim=1)
            outputs.append(out)
        return outputs


class YOLOX(nn.Module):
    """
    YOLOX detection model.
    Variants:
      YOLOX-S: dep_mul=0.33, wid_mul=0.50
      YOLOX-M: dep_mul=0.67, wid_mul=0.75
      YOLOX-L: dep_mul=1.00, wid_mul=1.00
    """
    def __init__(self, dep_mul=0.67, wid_mul=0.75, in_channels=3, num_classes=1, act="silu", export_mode=False):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.export_mode = export_mode
        self.strides = (8, 16, 32)
        self.backbone = YOLOPAFPN(dep_mul=dep_mul, wid_mul=wid_mul, in_channels=in_channels, act=act)
        self.head = YOLOXHead(num_classes=num_classes, wid_mul=wid_mul, act=act, strides=self.strides)

    def forward(self, x):
        feats = self.backbone(x)
        outputs = self.head(feats)
        return outputs


def init_yolox_weights(model: nn.Module):
    """
    Стандартная инициализация весов в стиле RetinaNet / YOLOX.
    Инициализирует conv веса kaiming uniform, а bias классификационных и объектных голов
    устанавливает в logit(prior_prob=0.01) ~= -4.6, чтобы в начале обучения вероятность
    фона была 99%, предотвращая взрыв ложных срабатываний и дестабилизацию градиентов.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    prior_prob = 0.01
    bias_val = -math.log((1 - prior_prob) / prior_prob)
    if hasattr(model, "head"):
        for cp in model.head.cls_preds:
            nn.init.normal_(cp.weight, std=0.01)
            nn.init.constant_(cp.bias, bias_val)
        for op in model.head.obj_preds:
            nn.init.normal_(op.weight, std=0.01)
            nn.init.constant_(op.bias, bias_val)


def load_yolox_pretrained_weights(model: nn.Module, in_channels: int = 3, weights_path: Optional[str] = None) -> bool:
    """
    Загрузка официальных предобученных весов COCO для YOLOX-M с адаптацией:
    1. Перенос весов слоя Focus из оригинального YOLOX в стандартный BaseConv stem (3x3 stride 2),
       что сохраняет полную совместимость с NPU RKNN.
    2. Кросс-модальная адаптация (Weight Inflation) для N > 3 каналов (4, 5, 7 каналов):
       - Видимые RGB каналы получают предобученные RGB фильтры.
       - Спектральные каналы (RE, NIR) инициализируются усреднением RGB фильтров (структурно-текстурные градиенты).
       - Масштабирование весов первого слоя на коэффициент (3 / in_channels) для строгого сохранения дисперсии активаций.
       - Все последующие слои (Dark2-Dark5, SPP, PAFPN, Reg/Obj heads) идентичны на 100% и получают предобученные веса.
    3. Головы классификации cls_preds адаптируются под целевое число классов (1 класс: сорняк/культура).
    """
    # Standalone: no parent-project paths, downloads or hidden cache fallback.
    if weights_path is None or not os.path.isfile(weights_path):
        raise FileNotFoundError("Pass a local official COCO yolox_m.pth file")
    ckpt_file = weights_path

    try:
        ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            # Native D01/standalone checkpoint: architecture must match exactly.
            state_dict = ckpt["model_state_dict"]
            if not isinstance(state_dict, dict) or not state_dict:
                raise ValueError("Empty or invalid model_state_dict")
            model.load_state_dict(state_dict, strict=True)
            print(f"✅ [YOLOX-M] Загружен checkpoint {os.path.basename(ckpt_file)}: "
                  f"{len(state_dict)} тензоров (strict=True); оптимизатор и эпоха сброшены.")
            return True
        state_dict = ckpt["model"] if "model" in ckpt else ckpt

        # Адаптация весов Focus (48, 12, 3, 3) к Conv2d stem (48, in_channels, 3, 3)
        if "backbone.backbone.stem.conv.conv.weight" in state_dict:
            w_focus = state_dict["backbone.backbone.stem.conv.conv.weight"]  # [48, 12, 3, 3]
            out_c = w_focus.shape[0]
            # Восстанавливаем 3-канальный RGB фильтр усреднением 4 субпиксельных позиций
            w_rgb = torch.zeros(out_c, 3, 3, 3, dtype=w_focus.dtype, device=w_focus.device)
            for c in range(3):
                w_rgb[:, c] = (w_focus[:, c] + w_focus[:, c + 3] + w_focus[:, c + 6] + w_focus[:, c + 9]) / 4.0

            # Создаем расширенный тензор весов под целевое число входных каналов
            w_inflated = torch.zeros(out_c, in_channels, 3, 3, dtype=w_focus.dtype, device=w_focus.device)
            mean_rgb = w_rgb.mean(dim=1, keepdim=True)  # [out_c, 1, 3, 3]

            if in_channels == 3:
                w_inflated[:, 0:3] = w_rgb
            elif in_channels == 4:  # [MS_R, MS_G, MS_RE, MS_NIR]
                w_inflated[:, 0] = w_rgb[:, 0]      # MS_R -> R
                w_inflated[:, 1] = w_rgb[:, 1]      # MS_G -> G
                w_inflated[:, 2] = mean_rgb[:, 0]  # MS_RE -> mean(RGB)
                w_inflated[:, 3] = mean_rgb[:, 0]  # MS_NIR -> mean(RGB)
                w_inflated = w_inflated * (3.0 / 4.0)
            elif in_channels == 5:  # [R, G, B, RE, NIR]
                w_inflated[:, 0:3] = w_rgb
                w_inflated[:, 3] = mean_rgb[:, 0]  # RE -> mean(RGB)
                w_inflated[:, 4] = mean_rgb[:, 0]  # NIR -> mean(RGB)
                w_inflated = w_inflated * (3.0 / 5.0)
            elif in_channels == 7:  # [RGB_R, RGB_G, RGB_B, MS_R, MS_G, MS_RE, MS_NIR]
                w_inflated[:, 0:3] = w_rgb
                w_inflated[:, 3] = w_rgb[:, 0]      # MS_R -> R
                w_inflated[:, 4] = w_rgb[:, 1]      # MS_G -> G
                w_inflated[:, 5] = mean_rgb[:, 0]  # MS_RE -> mean(RGB)
                w_inflated[:, 6] = mean_rgb[:, 0]  # MS_NIR -> mean(RGB)
                w_inflated = w_inflated * (3.0 / 7.0)
            else:
                for i in range(in_channels):
                    w_inflated[:, i] = w_rgb[:, i % 3]
                w_inflated = w_inflated * (3.0 / in_channels)

            state_dict["backbone.backbone.stem.conv.weight"] = w_inflated
            state_dict["backbone.backbone.stem.bn.weight"] = state_dict["backbone.backbone.stem.conv.bn.weight"]
            state_dict["backbone.backbone.stem.bn.bias"] = state_dict["backbone.backbone.stem.conv.bn.bias"]
            state_dict["backbone.backbone.stem.bn.running_mean"] = state_dict["backbone.backbone.stem.conv.bn.running_mean"]
            state_dict["backbone.backbone.stem.bn.running_var"] = state_dict["backbone.backbone.stem.conv.bn.running_var"]
            state_dict["backbone.backbone.stem.bn.num_batches_tracked"] = state_dict["backbone.backbone.stem.conv.bn.num_batches_tracked"]

        # Загружаем только слои с совпадающей размерностью
        model_state = model.state_dict()
        clean_dict = {}
        for k, v in state_dict.items():
            if k in model_state and model_state[k].shape == v.shape:
                clean_dict[k] = v

        if not clean_dict:
            raise ValueError("No matching model tensors: incompatible checkpoint format or architecture")
        missing, unexpected = model.load_state_dict(clean_dict, strict=False)
        print(f"✅ [YOLOX-M] Успешно загружено {len(clean_dict)} тензоров из {os.path.basename(ckpt_file)} (каналов: {in_channels}).")
        return True
    except Exception as e:
        print(f"⚠️ [YOLOX-M] Ошибка при загрузке предобученных весов: {e}")
        return False


def build_yolox_m(in_channels=3, num_classes=1, export_mode=False, pretrained=True, weights_path=None):
    model = YOLOX(dep_mul=0.67, wid_mul=0.75, in_channels=in_channels, num_classes=num_classes, export_mode=export_mode)
    init_yolox_weights(model)
    if pretrained and not export_mode:
        load_yolox_pretrained_weights(model, in_channels=in_channels, weights_path=weights_path)
    return model


def build_yolox_l(in_channels=3, num_classes=1, export_mode=False, pretrained=False, weights_path=None):
    model = YOLOX(dep_mul=1.00, wid_mul=1.00, in_channels=in_channels, num_classes=num_classes, export_mode=export_mode)
    init_yolox_weights(model)
    return model

