"""Helper for generating a lightweight dummy ONNX detection model for unit testing."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _encode_varint(value: int) -> bytes:
    bits = value & 0x7F
    value >>= 7
    res = bytearray()
    while value:
        res.append(0x80 | bits)
        bits = value & 0x7F
        value >>= 7
    res.append(bits)
    return bytes(res)


def _encode_tag(field_number: int, wire_type: int) -> bytes:
    return _encode_varint((field_number << 3) | wire_type)


def _encode_length_delimited(field_number: int, data: bytes | str) -> bytes:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return _encode_tag(field_number, 2) + _encode_varint(len(data)) + data


def _encode_int64(field_number: int, val: int) -> bytes:
    return _encode_tag(field_number, 0) + _encode_varint(val)


def _make_dim(dim_value: int) -> bytes:
    return _encode_length_delimited(1, _encode_int64(1, dim_value))


def _make_shape(dims: list[int]) -> bytes:
    dims_bytes = b"".join(_make_dim(d) for d in dims)
    # In TypeProto.Tensor, shape is field 2
    return _encode_length_delimited(2, dims_bytes)


def _make_type_proto_tensor(elem_type: int, dims: list[int]) -> bytes:
    tensor_fields = _encode_int64(1, elem_type) + _make_shape(dims)
    # In TypeProto, tensor_type is field 1
    return _encode_length_delimited(1, tensor_fields)


def _make_value_info(name: str, elem_type: int, dims: list[int]) -> bytes:
    type_proto = _make_type_proto_tensor(elem_type, dims)
    return _encode_length_delimited(1, name) + _encode_length_delimited(2, type_proto)


def _make_tensor_proto(name: str, dims: list[int], data_bytes: bytes) -> bytes:
    res = bytearray()
    for d in dims:
        res += _encode_int64(1, d)
    res += _encode_int64(2, 1)  # FLOAT
    res += _encode_length_delimited(8, name)
    res += _encode_length_delimited(9, data_bytes)
    return bytes(res)


def _make_attr_tensor(name: str, tensor_proto: bytes) -> bytes:
    res = bytearray()
    res += _encode_length_delimited(1, name)
    res += _encode_int64(20, 4)  # AttributeType::TENSOR = 4
    res += _encode_length_delimited(5, tensor_proto)
    return bytes(res)


def _make_constant_node(out_name: str, attr_tensor: bytes) -> bytes:
    res = bytearray()
    res += _encode_length_delimited(2, out_name)
    res += _encode_length_delimited(3, "const_node")
    res += _encode_length_delimited(4, "Constant")
    res += _encode_length_delimited(5, attr_tensor)
    return bytes(res)


def _make_graph(name: str, node: bytes, inputs: list[bytes], outputs: list[bytes]) -> bytes:
    res = bytearray()
    res += _encode_length_delimited(1, node)
    res += _encode_length_delimited(2, name)
    for inp in inputs:
        res += _encode_length_delimited(11, inp)
    for out in outputs:
        res += _encode_length_delimited(12, out)
    return bytes(res)


def _make_opset(version: int) -> bytes:
    return _encode_length_delimited(8, _encode_int64(2, version))


def _make_model(graph: bytes) -> bytes:
    res = bytearray()
    res += _encode_int64(1, 8)  # ir_version 8
    res += _make_opset(17)  # opset 17
    res += _encode_length_delimited(7, graph)
    return bytes(res)


def create_dummy_yolo_onnx(path: Path) -> Path:
    """Create a minimal valid ONNX file producing fixed YOLOv8 detection proposals.
    
    Output tensor shape: (1, 6, 8400)
    Proposal 0: weed_dicot at center (cx=320, cy=320, w=100, h=80, score=0.92)
    Proposal 1: weed_monocot at (cx=120, cy=180, w=60, h=50, score=0.88)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    out_arr = np.zeros((1, 6, 8400), dtype=np.float32)

    # Box 1: weed_dicot (class 0)
    out_arr[0, 0, 0] = 320.0
    out_arr[0, 1, 0] = 320.0
    out_arr[0, 2, 0] = 100.0
    out_arr[0, 3, 0] = 80.0
    out_arr[0, 4, 0] = 0.92
    out_arr[0, 5, 0] = 0.05

    # Box 2: weed_monocot (class 1)
    out_arr[0, 0, 1] = 120.0
    out_arr[0, 1, 1] = 180.0
    out_arr[0, 2, 1] = 60.0
    out_arr[0, 3, 1] = 50.0
    out_arr[0, 4, 1] = 0.08
    out_arr[0, 5, 1] = 0.88

    t_proto = _make_tensor_proto("val", [1, 6, 8400], out_arr.tobytes())
    attr = _make_attr_tensor("value", t_proto)
    node = _make_constant_node("output0", attr)
    inp = _make_value_info("images", 1, [1, 3, 640, 640])
    out = _make_value_info("output0", 1, [1, 6, 8400])
    graph = _make_graph("dummy_yolo", node, [inp], [out])
    model_bytes = _make_model(graph)

    with open(path, "wb") as f:
        f.write(model_bytes)

    return path
