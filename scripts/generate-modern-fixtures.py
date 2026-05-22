#!/usr/bin/env python3
"""Generate local HDF5 C-library fixtures."""

from pathlib import Path
import os
import zlib

import h5py
import numpy as np


OUT = Path("tests/data/hdf5_ref")

GLOBAL_HEAP_MAGIC = b"GCOL"


def write_fixed_array(path: Path) -> None:
    data = np.arange(100, dtype=np.int32)
    with h5py.File(path, "w", libver="latest") as f:
        f.create_dataset("fixed_array", data=data, chunks=(10,))


def write_fixed_array_deflate_parallel_threshold_tail(path: Path) -> None:
    chunk = 2048
    length = chunk * 8 + 17
    data = (np.arange(length, dtype=np.int32) * 5) - 11
    with h5py.File(path, "w", libver="latest") as f:
        f.create_dataset(
            "fixed_array_deflate_parallel_threshold_tail",
            data=data,
            chunks=(chunk,),
            compression="gzip",
            compression_opts=4,
        )


def write_fixed_array_deflate_mask_parallel_fallback(path: Path) -> None:
    chunk = 2048
    length = chunk * 8
    data = (np.arange(length, dtype=np.int32) * 2) - 31
    with h5py.File(path, "w", libver="latest") as f:
        ds = f.create_dataset(
            "fixed_array_deflate_mask_parallel_fallback",
            shape=(length,),
            dtype=np.int32,
            chunks=(chunk,),
            compression="gzip",
            compression_opts=4,
        )
        first = np.asarray(data[:chunk], dtype="<i4")
        ds.id.write_direct_chunk((0,), first.tobytes(), filter_mask=0b1)
        ds[chunk:] = data[chunk:]


def write_fixed_array_3d_edges(path: Path) -> None:
    data = np.arange(5 * 7 * 4, dtype=np.int32).reshape(5, 7, 4)
    with h5py.File(path, "w", libver="latest") as f:
        f.create_dataset("fixed_array_3d_edges", data=data, chunks=(2, 3, 2))


def write_paged_fixed_array(path: Path) -> None:
    data = np.arange(4096, dtype=np.int32)
    with h5py.File(path, "w", libver="latest") as f:
        f.create_dataset("fixed_array_paged", data=data, chunks=(1,))


def write_paged_fixed_array_sparse(path: Path) -> None:
    with h5py.File(path, "w", libver="latest") as f:
        ds = f.create_dataset(
            "fixed_array_paged_sparse",
            shape=(4096,),
            dtype=np.int32,
            chunks=(1,),
            fillvalue=-3,
        )
        ds[0] = 11
        ds[2048] = 22
        ds[4095] = 33


def write_extensible_array(path: Path) -> None:
    data = np.arange(80, dtype=np.float64)
    with h5py.File(path, "w", libver="latest") as f:
        ds = f.create_dataset(
            "extensible_array",
            shape=(0,),
            maxshape=(None,),
            chunks=(20,),
            dtype="f8",
        )
        ds.resize((data.shape[0],))
        ds[...] = data


def write_extensible_array_deflate_parallel_threshold_tail(path: Path) -> None:
    chunk = 2048
    length = chunk * 8 + 17
    data = (np.arange(length, dtype=np.int32) * 7) - 13
    with h5py.File(path, "w", libver="latest") as f:
        ds = f.create_dataset(
            "extensible_array_deflate_parallel_threshold_tail",
            shape=(0,),
            maxshape=(None,),
            chunks=(chunk,),
            dtype=np.int32,
            compression="gzip",
            compression_opts=4,
        )
        ds.resize((data.shape[0],))
        ds[...] = data


def write_extensible_array_deflate_mask_parallel_fallback(path: Path) -> None:
    chunk = 2048
    length = chunk * 8
    data = (np.arange(length, dtype=np.int32) * 11) - 19
    with h5py.File(path, "w", libver="latest") as f:
        ds = f.create_dataset(
            "extensible_array_deflate_mask_parallel_fallback",
            shape=(0,),
            maxshape=(None,),
            chunks=(chunk,),
            dtype=np.int32,
            compression="gzip",
            compression_opts=4,
        )
        ds.resize((length,))
        first = np.asarray(data[:chunk], dtype="<i4")
        ds.id.write_direct_chunk((0,), first.tobytes(), filter_mask=0b1)
        ds[chunk:] = data[chunk:]


def write_extensible_array_2d_unlimited_edges(path: Path) -> None:
    data = np.arange(5 * 7, dtype=np.int32).reshape(5, 7)
    with h5py.File(path, "w", libver="latest") as f:
        ds = f.create_dataset(
            "extensible_array_2d_unlimited_edges",
            shape=(0, 7),
            maxshape=(None, 7),
            chunks=(2, 3),
            dtype=np.int32,
        )
        ds.resize(data.shape)
        ds[...] = data


def write_extensible_array_spillover(path: Path) -> None:
    data = np.arange(4096, dtype=np.float64)
    with h5py.File(path, "w", libver="latest") as f:
        ds = f.create_dataset(
            "extensible_array_spillover",
            shape=(0,),
            maxshape=(None,),
            chunks=(1,),
            dtype="f8",
        )
        ds.resize((data.shape[0],))
        ds[...] = data


def write_extensible_array_sparse_transitions(path: Path) -> None:
    with h5py.File(path, "w", libver="latest") as f:
        ds = f.create_dataset(
            "extensible_array_sparse_transitions",
            shape=(0,),
            maxshape=(None,),
            chunks=(1,),
            dtype=np.int32,
            fillvalue=-4,
        )
        ds.resize((4096,))
        for idx in [0, 1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 511, 512, 1023, 1024, 2047, 2048, 4095]:
            ds[idx] = idx


def write_v2_btree(path: Path) -> None:
    data = np.arange(8 * 8, dtype=np.int32).reshape(8, 8)
    with h5py.File(path, "w", libver="latest") as f:
        ds = f.create_dataset(
            "btree_v2",
            shape=(0, 0),
            maxshape=(None, None),
            chunks=(4, 4),
            dtype="i4",
        )
        ds.resize(data.shape)
        ds[...] = data


def write_v2_btree_internal(path: Path) -> None:
    data = np.arange(80 * 80, dtype=np.int32).reshape(80, 80)
    with h5py.File(path, "w", libver="latest") as f:
        ds = f.create_dataset(
            "btree_v2_internal",
            shape=(0, 0),
            maxshape=(None, None),
            chunks=(1, 1),
            dtype="i4",
        )
        ds.resize(data.shape)
        ds[...] = data


def write_v2_btree_deep_internal(path: Path) -> None:
    data = np.arange(160 * 160, dtype=np.int32).reshape(160, 160)
    with h5py.File(path, "w", libver="latest") as f:
        ds = f.create_dataset(
            "btree_v2_deep_internal",
            shape=(0, 0),
            maxshape=(None, None),
            chunks=(1, 1),
            dtype="i4",
        )
        ds.resize(data.shape)
        ds[...] = data


def write_v2_btree_filtered_mask(path: Path) -> None:
    with h5py.File(path, "w", libver="latest") as f:
        ds = f.create_dataset(
            "btree_v2_filtered_mask",
            shape=(0, 0),
            maxshape=(None, None),
            chunks=(2, 2),
            dtype=np.int32,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
            fillvalue=-1,
        )
        ds.resize((4, 4))
        raw_chunk = np.array([[1, 2], [5, 6]], dtype="<i4")
        ds.id.write_direct_chunk((0, 0), raw_chunk.tobytes(), filter_mask=0b11)
        ds[2:4, 2:4] = np.array([[11, 12], [15, 16]], dtype=np.int32)


def write_filtered_implicit(path: Path) -> None:
    data = np.arange(64, dtype=np.int16)
    with h5py.File(path, "w", libver="latest") as f:
        f.create_dataset(
            "filtered_chunked",
            data=data,
            chunks=(16,),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )


def write_implicit_2d_edge_chunks(path: Path) -> None:
    data = np.arange(5 * 7, dtype=np.int32).reshape(5, 7)
    with h5py.File(path, "w", libver="latest") as f:
        f.create_dataset("implicit_2d_edge", data=data, chunks=(2, 3))


def write_sparse_chunked_fill_value(path: Path) -> None:
    with h5py.File(path, "w", libver="latest") as f:
        ds = f.create_dataset(
            "sparse_chunked_fill",
            shape=(4, 6),
            dtype=np.int32,
            chunks=(2, 3),
            fillvalue=-7,
        )
        ds[0:2, 0:3] = np.arange(6, dtype=np.int32).reshape(2, 3)


def write_filtered_chunk_filter_mask(path: Path) -> None:
    with h5py.File(path, "w", libver="latest") as f:
        ds = f.create_dataset(
            "filtered_chunk_filter_mask",
            shape=(4, 6),
            dtype=np.int32,
            chunks=(2, 3),
            fillvalue=-7,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        raw_chunk = np.arange(6, dtype="<i4").reshape(2, 3)
        ds.id.write_direct_chunk((0, 0), raw_chunk.tobytes(), filter_mask=0b11)
        ds[2:4, 3:6] = (np.arange(6, dtype=np.int32).reshape(2, 3) + 100)


def write_filtered_single_chunk_filter_mask(path: Path) -> None:
    with h5py.File(path, "w", libver="latest") as f:
        ds = f.create_dataset(
            "filtered_single_chunk_filter_mask",
            shape=(2, 3),
            dtype=np.int32,
            chunks=(2, 3),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        raw_chunk = np.arange(6, dtype="<i4").reshape(2, 3)
        ds.id.write_direct_chunk((0, 0), raw_chunk.tobytes(), filter_mask=0b11)


def shuffle_bytes(data: bytes, element_size: int) -> bytes:
    if element_size <= 1 or not data:
        return data
    n_elements = len(data) // element_size
    out = bytearray(len(data))
    for elem_idx in range(n_elements):
        for byte_idx in range(element_size):
            out[byte_idx * n_elements + elem_idx] = data[elem_idx * element_size + byte_idx]
    return bytes(out)


def fletcher32(data: bytes) -> int:
    sum1 = 0
    sum2 = 0
    pos = 0
    remaining = len(data) // 2
    while remaining:
        batch = min(remaining, 360)
        remaining -= batch
        for _ in range(batch):
            value = (data[pos] << 8) | data[pos + 1]
            sum1 += value
            sum2 += sum1
            pos += 2
        sum1 = (sum1 & 0xFFFF) + (sum1 >> 16)
        sum2 = (sum2 & 0xFFFF) + (sum2 >> 16)
    if len(data) % 2:
        sum1 += data[pos] << 8
        sum2 += sum1
        sum1 = (sum1 & 0xFFFF) + (sum1 >> 16)
        sum2 = (sum2 & 0xFFFF) + (sum2 >> 16)
    sum1 = (sum1 & 0xFFFF) + (sum1 >> 16)
    sum2 = (sum2 & 0xFFFF) + (sum2 >> 16)
    return ((sum2 << 16) | sum1) & 0xFFFFFFFF


def write_filtered_middle_filter_mask(path: Path) -> None:
    with h5py.File(path, "w", libver="latest") as f:
        ds = f.create_dataset(
            "filtered_middle_filter_mask",
            shape=(6,),
            dtype=np.int32,
            chunks=(6,),
            shuffle=True,
            compression="gzip",
            compression_opts=4,
            fletcher32=True,
        )
        raw_chunk = np.arange(6, dtype="<i4").tobytes()
        shuffled = shuffle_bytes(raw_chunk, 4)
        checksum = fletcher32(shuffled).to_bytes(4, "little")
        ds.id.write_direct_chunk((0,), shuffled + checksum, filter_mask=0b010)


def write_multi_filter_orders(path: Path) -> None:
    int_data = np.arange(32, dtype=np.int32)
    nbit_data = np.arange(64, dtype=np.int32)
    with h5py.File(path, "w", libver="latest") as f:
        f.create_dataset(
            "scaleoffset_shuffle_deflate",
            data=int_data,
            chunks=(16,),
            scaleoffset=0,
            shuffle=True,
            compression="gzip",
            compression_opts=4,
        )
        f.create_dataset(
            "shuffle_deflate_fletcher",
            data=int_data,
            chunks=(16,),
            shuffle=True,
            compression="gzip",
            compression_opts=4,
            fletcher32=True,
        )

        space = h5py.h5s.create_simple(nbit_data.shape)
        dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        dcpl.set_chunk((16,))
        dcpl.set_filter(5, 0, ())
        dcpl.set_deflate(4)
        dset = h5py.h5d.create(f.id, b"nbit_deflate", h5py.h5t.STD_I32LE, space, dcpl=dcpl)
        dset.write(h5py.h5s.ALL, h5py.h5s.ALL, nbit_data)


def write_fletcher32_corrupt(path: Path) -> None:
    raw_chunk = np.arange(6, dtype="<i4").tobytes()
    bad_checksum = (0x12345678).to_bytes(4, "little")
    with h5py.File(path, "w", libver="latest") as f:
        plain = f.create_dataset(
            "fletcher32_corrupt",
            shape=(6,),
            dtype=np.int32,
            chunks=(6,),
            fletcher32=True,
        )
        plain.id.write_direct_chunk((0,), raw_chunk + bad_checksum, filter_mask=0)

        filtered = f.create_dataset(
            "deflate_fletcher32_corrupt",
            shape=(6,),
            dtype=np.int32,
            chunks=(6,),
            compression="gzip",
            compression_opts=4,
            fletcher32=True,
        )
        filtered.id.write_direct_chunk((0,), zlib.compress(raw_chunk, 4) + bad_checksum, filter_mask=0)


def write_nbit_filter(path: Path) -> None:
    data = np.arange(100, dtype=np.int32)
    with h5py.File(path, "w") as f:
        space = h5py.h5s.create_simple(data.shape)
        dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        dcpl.set_chunk((20,))
        dcpl.set_filter(5, 0, ())
        dset = h5py.h5d.create(f.id, b"nbit_i32", h5py.h5t.STD_I32LE, space, dcpl=dcpl)
        dset.write(h5py.h5s.ALL, h5py.h5s.ALL, data)


def write_nbit_filter_be(path: Path) -> None:
    data = np.arange(100, dtype=">i4")
    with h5py.File(path, "w") as f:
        space = h5py.h5s.create_simple(data.shape)
        dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        dcpl.set_chunk((20,))
        dcpl.set_filter(5, 0, ())
        dset = h5py.h5d.create(f.id, b"nbit_be_i32", h5py.h5t.STD_I32BE, space, dcpl=dcpl)
        dset.write(h5py.h5s.ALL, h5py.h5s.ALL, data)


def write_nbit_parity_vectors(path: Path) -> None:
    def create_nbit_dataset(file_id, name: bytes, data, h5type, chunks) -> None:
        space = h5py.h5s.create_simple(data.shape)
        dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        dcpl.set_chunk(chunks)
        dcpl.set_filter(5, 0, ())
        dset = h5py.h5d.create(file_id, name, h5type, space, dcpl=dcpl)
        dset.write(h5py.h5s.ALL, h5py.h5s.ALL, data)

    signed = np.array([-32768, -257, -1, 0, 1, 255, 1024, 32767], dtype="<i2")
    unsigned = np.array([0, 1, 255, 256, 1024, 32768, 65535], dtype="<u2")
    floats = np.array([-0.0, 1.5, -2.25, 123.5, np.inf, -np.inf], dtype="<f4")
    compound_dtype = np.dtype([("code", "<i2"), ("count", "<u2"), ("score", "<f4")])
    compound = np.array(
        [(-7, 3, 1.25), (12, 65530, -4.5), (-1024, 42, 32.0)],
        dtype=compound_dtype,
    )

    with h5py.File(path, "w") as f:
        create_nbit_dataset(f.id, b"nbit_i16_signed", signed, h5py.h5t.STD_I16LE, (4,))
        create_nbit_dataset(f.id, b"nbit_u16_unsigned", unsigned, h5py.h5t.STD_U16LE, (4,))
        create_nbit_dataset(f.id, b"nbit_f32", floats, h5py.h5t.IEEE_F32LE, (3,))
        create_nbit_dataset(
            f.id,
            b"nbit_compound_members",
            compound,
            h5py.h5t.py_create(compound_dtype),
            (2,),
        )


def write_scaleoffset_filter(path: Path) -> None:
    data = np.arange(100, dtype=np.int32)
    with h5py.File(path, "w") as f:
        f.create_dataset("scaleoffset_i32", data=data, chunks=(20,), scaleoffset=0)
        floats = (np.arange(40, dtype=np.float32) / np.float32(10.0)) + np.float32(1.25)
        f.create_dataset("scaleoffset_f32", data=floats, chunks=(10,), scaleoffset=2)


def write_scaleoffset_filter_be(path: Path) -> None:
    data = np.arange(100, dtype=">i4")
    with h5py.File(path, "w") as f:
        f.create_dataset("scaleoffset_be_i32", data=data, chunks=(20,), scaleoffset=0)


def write_scaleoffset_parity_vectors(path: Path) -> None:
    signed = np.array([-120, -17, -1, 0, 5, 63, 127], dtype=np.int16)
    unsigned = np.array([1000, 1001, 1003, 1007, 1015, 1023], dtype=np.uint16)
    zero_minbits = np.full((8,), -42, dtype=np.int32)
    f32 = np.array([-1.25, -0.5, 0.0, 1.25, 3.5], dtype=np.float32)
    f64 = np.array([-100.125, -1.5, 0.25, 12.75, 2048.5], dtype=np.float64)
    with h5py.File(path, "w") as f:
        f.create_dataset("scaleoffset_i16_signed", data=signed, chunks=(4,), scaleoffset=0)
        f.create_dataset("scaleoffset_u16_minbits", data=unsigned, chunks=(3,), scaleoffset=0)
        f.create_dataset(
            "scaleoffset_i32_zero_minbits",
            data=zero_minbits,
            chunks=(4,),
            scaleoffset=0,
        )
        f.create_dataset("scaleoffset_f32_dscale", data=f32, chunks=(3,), scaleoffset=2)
        f.create_dataset("scaleoffset_f64_dscale", data=f64, chunks=(3,), scaleoffset=3)


def write_integer_conversion_vectors(path: Path) -> None:
    signed = np.array([-129, -1, 0, 1, 127, 128, 255, 256, 32767], dtype=np.int16)
    unsigned = np.array([0, 1, 127, 128, 255, 256, 32767, 32768, 65535], dtype=np.uint16)
    with h5py.File(path, "w") as f:
        f.create_dataset("i16_conversion", data=signed)
        f.create_dataset("u16_conversion", data=unsigned)


def write_float_conversion_vectors(path: Path) -> None:
    floats32 = np.array(
        [-np.inf, -129.75, -1.5, -0.0, 0.0, 1.5, 127.25, 128.75, np.inf, np.nan],
        dtype=np.float32,
    )
    floats64 = floats32.astype(np.float64)
    ints = np.array([-129, -1, 0, 1, 127, 128, 255, 32767], dtype=np.int16)
    uints = np.array([0, 1, 127, 128, 255, 256, 32767, 65535], dtype=np.uint16)
    with h5py.File(path, "w") as f:
        f.create_dataset("f32_conversion", data=floats32)
        f.create_dataset("f64_conversion", data=floats64)
        f.create_dataset("be_f32_conversion", data=floats32.astype(">f4"))
        f.create_dataset("i16_to_float_conversion", data=ints)
        f.create_dataset("u16_to_float_conversion", data=uints)


def write_fixed_string_cases(path: Path) -> None:
    def create_fixed_string(file_id, name: bytes, values: list[bytes], size: int, strpad: int, cset: int) -> None:
        tid = h5py.h5t.C_S1.copy()
        tid.set_size(size)
        tid.set_strpad(strpad)
        tid.set_cset(cset)
        space = h5py.h5s.create_simple((len(values),))
        dset = h5py.h5d.create(file_id, name, tid, space)
        dset.write(h5py.h5s.ALL, h5py.h5s.ALL, np.array(values, dtype=f"S{size}"), mtype=tid)

    with h5py.File(path, "w") as f:
        create_fixed_string(
            f.id,
            b"null_padded",
            [b"hi", b"a b", b"trail "],
            8,
            h5py.h5t.STR_NULLPAD,
            h5py.h5t.CSET_ASCII,
        )
        create_fixed_string(
            f.id,
            b"space_padded",
            [b"hi", b"a b", b"trail "],
            8,
            h5py.h5t.STR_SPACEPAD,
            h5py.h5t.CSET_ASCII,
        )
        create_fixed_string(
            f.id,
            b"null_terminated",
            [b"hi", b"a b", b"trail "],
            8,
            h5py.h5t.STR_NULLTERM,
            h5py.h5t.CSET_ASCII,
        )
        create_fixed_string(
            f.id,
            b"utf8_fixed",
            ["å".encode(), "猫".encode(), "hi".encode()],
            8,
            h5py.h5t.STR_NULLPAD,
            h5py.h5t.CSET_UTF8,
        )


def write_vlen_string_cases(path: Path) -> None:
    dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "vlen_utf8_strings",
            data=np.array(["", "猫", "å", "alpha"], dtype=object),
            dtype=dtype,
        )
        f.create_dataset(
            "vlen_global_heap_edges",
            data=np.array(["dup", "dup", "long-" + ("x" * 96)], dtype=object),
            dtype=dtype,
        )
        null_ds = f.create_dataset(
            "vlen_null_descriptor",
            data=np.array(["will_be_null", "kept"], dtype=object),
            dtype=dtype,
        )
        null_offset = null_ds.id.get_offset()

    with path.open("r+b") as fh:
        fh.seek(null_offset)
        fh.write(b"\x00" * 16)


def write_opaque_cases(path: Path) -> None:
    with h5py.File(path, "w") as f:
        tid = h5py.h5t.create(h5py.h5t.OPAQUE, 4)
        tid.set_tag(b"hdf5-pure-rust opaque fixture")
        space = h5py.h5s.create_simple((3,))
        dset = h5py.h5d.create(f.id, b"opaque_tagged", tid, space)
        payload = np.array([b"abcd", b"\x00\x01\x02\x03", b"wxyz"], dtype="|V4")
        dset.write(h5py.h5s.ALL, h5py.h5s.ALL, payload, mtype=tid)


def write_reference_cases(path: Path) -> None:
    with h5py.File(path, "w") as f:
        target = f.create_dataset("target", data=np.arange(6, dtype=np.int32).reshape(2, 3))
        group = f.create_group("target_group")

        object_refs = np.empty(3, dtype=h5py.ref_dtype)
        object_refs[0] = target.ref
        object_refs[1] = group.ref
        object_refs[2] = h5py.Reference()
        f.create_dataset("object_refs", data=object_refs)

        region_refs = np.empty(2, dtype=h5py.regionref_dtype)
        region_refs[0] = target.regionref[0:2, 1:3]
        region_refs[1] = h5py.RegionReference()
        f.create_dataset("region_refs", data=region_refs)


def write_time_cases(path: Path) -> None:
    with h5py.File(path, "w") as f:
        data32 = np.array([0, 1, 2_147_483_647], dtype="<u4")
        space32 = h5py.h5s.create_simple(data32.shape)
        dset32 = h5py.h5d.create(f.id, b"unix_d32le", h5py.h5t.UNIX_D32LE, space32)
        dset32.write(h5py.h5s.ALL, h5py.h5s.ALL, data32, mtype=h5py.h5t.UNIX_D32LE)

        data64 = np.array([0, 1, 4_102_444_800], dtype=">u8")
        space64 = h5py.h5s.create_simple(data64.shape)
        dset64 = h5py.h5d.create(f.id, b"unix_d64be", h5py.h5t.UNIX_D64BE, space64)
        dset64.write(h5py.h5s.ALL, h5py.h5s.ALL, data64, mtype=h5py.h5t.UNIX_D64BE)


def write_enum_conversion_cases(path: Path) -> None:
    enum_u16be = h5py.enum_dtype(
        {"LOW": 1, "MID": 258, "HIGH": 4095},
        basetype=">u2",
    )
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "enum_u16be",
            data=np.array([1, 258, 4095], dtype=">u2"),
            dtype=enum_u16be,
        )


def write_fractal_heap_cases(path: Path) -> None:
    with h5py.File(path, "w", libver="latest") as f:
        group = f.create_group("many_links")
        for idx in range(80):
            group.create_dataset(f"link_{idx:03}", data=np.arange(4, dtype=np.int32))
        dset = f.create_dataset("many_attrs", data=np.arange(8, dtype=np.int32))
        for idx in range(80):
            dset.attrs[f"attr_{idx:03}"] = np.arange(16, dtype=np.int64)


def write_dense_group_cases(path: Path) -> None:
    with h5py.File(path, "w", libver="latest") as f:
        target = f.create_dataset("target", data=np.arange(4, dtype=np.int32))

        deep = f.create_group("name_index_deep")
        for idx in range(4096):
            deep[f"link_{idx:04}"] = target

        tracked = f.create_group("creation_order_tracked", track_order=True)
        for idx in reversed(range(64)):
            tracked[f"tracked_{idx:02}"] = target

        untracked = f.create_group("creation_order_untracked", track_order=False)
        for idx in reversed(range(64)):
            untracked[f"untracked_{idx:02}"] = target


def write_attribute_cases(path: Path) -> None:
    with h5py.File(path, "w", libver="latest") as f:
        compact_gcpl = h5py.h5p.create(h5py.h5p.GROUP_CREATE)
        compact_gcpl.set_attr_phase_change(64, 32)
        compact_id = h5py.h5g.create(f.id, b"large_compact_attrs", gcpl=compact_gcpl)
        compact = h5py.Group(compact_id)
        compact.attrs["large_i32"] = np.arange(256, dtype=np.int32)

        tracked = f.create_group("dense_attrs_tracked", track_order=True)
        for idx in range(32):
            tracked.attrs[f"attr_{idx:02}"] = np.array([idx, idx + 100], dtype=np.int32)

        untracked = f.create_group("dense_attrs_untracked", track_order=False)
        for idx in range(32):
            untracked.attrs[f"attr_{idx:02}"] = np.array([idx, idx + 200], dtype=np.int32)

        vlen_holder = f.create_group("vlen_attr_holder")
        vlen_holder.attrs.create(
            "vlen_strings",
            np.array(["", "alpha", "猫"], dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )


def write_link_edge_cases(path: Path) -> None:
    with h5py.File(path, "w", libver="latest") as f:
        f.create_dataset("target", data=np.arange(3, dtype=np.int32))
        f.create_group("猫_group")
        f["å_link"] = f["target"]
        f["external_å"] = h5py.ExternalLink("målfil.h5", "/dåta")


def write_complex_compound(path: Path) -> None:
    nested = np.dtype([("a", "<i4"), ("b", "<f8")])
    dtype = np.dtype(
        [
            ("nested", nested),
            ("arr", "<i2", (3,)),
            ("vlen", h5py.vlen_dtype(np.dtype("<i4"))),
            ("ref", h5py.ref_dtype),
        ]
    )

    with h5py.File(path, "w", libver="latest") as f:
        target = f.create_dataset("target", data=np.arange(3, dtype=np.int32))
        data = np.empty(2, dtype=dtype)
        data["nested"]["a"] = [7, 8]
        data["nested"]["b"] = [1.5, 2.5]
        data["arr"] = [[1, 2, 3], [4, 5, 6]]
        data["vlen"][0] = np.array([10, 11], dtype=np.int32)
        data["vlen"][1] = np.array([20, 21, 22], dtype=np.int32)
        data["ref"] = [target.ref, target.ref]
        f.create_dataset("compound_complex", data=data)


def write_compound_layout_cases(path: Path) -> None:
    padded_reordered = np.dtype(
        {
            "names": ["second", "first", "last"],
            "formats": ["<i2", "<i4", "u1"],
            "offsets": [4, 0, 8],
            "itemsize": 12,
        }
    )
    nested_vlen = np.dtype(
        [
            ("tag", "<i2"),
            ("seq", h5py.vlen_dtype(np.dtype("<i4"))),
        ]
    )
    outer = np.dtype([("nested_vlen", nested_vlen), ("id", "u1")])

    with h5py.File(path, "w", libver="latest") as f:
        padded = np.zeros(2, dtype=padded_reordered)
        padded["first"] = [1000, 2000]
        padded["second"] = [10, 20]
        padded["last"] = [7, 8]
        f.create_dataset("padded_reordered", data=padded)

        nested = np.empty(2, dtype=outer)
        nested["nested_vlen"]["tag"] = [3, 4]
        nested["nested_vlen"]["seq"][0] = np.array([1, 2], dtype=np.int32)
        nested["nested_vlen"]["seq"][1] = np.array([5, 6, 7], dtype=np.int32)
        nested["id"] = [9, 10]
        f.create_dataset("nested_vlen", data=nested)


def write_array_datatype_cases(path: Path) -> None:
    compound_array = np.dtype([("grid", "<i2", (2, 3)), ("id", "u1")])

    with h5py.File(path, "w", libver="latest") as f:
        array_type = h5py.h5t.array_create(h5py.h5t.STD_I16LE, (2, 3))
        space = h5py.h5s.create_simple((2,))
        dset = h5py.h5d.create(f.id, b"array_i16_2x3", array_type, space)
        payload = np.arange(12, dtype="<i2").reshape(2, 2, 3)
        dset.write(h5py.h5s.ALL, h5py.h5s.ALL, payload, mtype=array_type)

        data = np.zeros(2, dtype=compound_array)
        data["grid"][0] = [[1, 2, 3], [4, 5, 6]]
        data["grid"][1] = [[7, 8, 9], [10, 11, 12]]
        data["id"] = [13, 14]
        f.create_dataset("compound_array2d", data=data)


def write_compact_read_cases(path: Path) -> None:
    compound_dtype = np.dtype([("x", "<f8"), ("label", "<i4")])
    compound_data = np.array((1.5, 7), dtype=compound_dtype)

    with h5py.File(path, "w", libver="latest") as f:
        compact_dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        compact_dcpl.set_layout(h5py.h5d.COMPACT)

        zero_space = h5py.h5s.create_simple((0,))
        h5py.h5d.create(
            f.id,
            b"compact_zero",
            h5py.h5t.STD_I32LE,
            zero_space,
            dcpl=compact_dcpl,
        )

        compound_dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        compound_dcpl.set_layout(h5py.h5d.COMPACT)
        f.create_dataset(
            "compact_compound_scalar",
            shape=(),
            dtype=compound_dtype,
            data=compound_data,
            dcpl=compound_dcpl,
        )


def write_external_raw_storage(path: Path, raw_path: Path) -> None:
    cwd = Path.cwd()
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        os.chdir(OUT)
        with h5py.File(path.name, "w") as f:
            ds = f.create_dataset(
                "external_raw",
                shape=(4,),
                dtype=np.int32,
                external=[(raw_path.name, 0, h5py.h5f.UNLIMITED)],
            )
            ds[...] = np.array([1, 2, 3, 4], dtype=np.int32)
    finally:
        os.chdir(cwd)


def write_undefined_storage_address(path: Path) -> None:
    with h5py.File(path, "w", libver="latest") as f:
        space = h5py.h5s.create_simple((4,))
        dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        dcpl.set_alloc_time(h5py.h5d.ALLOC_TIME_LATE)
        dcpl.set_fill_time(h5py.h5d.FILL_TIME_IFSET)
        dcpl.set_fill_value(np.array(-5, dtype=np.int32))
        h5py.h5d.create(f.id, b"late_fill", h5py.h5t.STD_I32LE, space, dcpl=dcpl)


def write_late_fill_time_never(path: Path) -> None:
    with h5py.File(path, "w", libver="latest") as f:
        space = h5py.h5s.create_simple((4,))
        dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        dcpl.set_alloc_time(h5py.h5d.ALLOC_TIME_LATE)
        dcpl.set_fill_time(h5py.h5d.FILL_TIME_NEVER)
        dcpl.set_fill_value(np.array(-5, dtype=np.int32))
        h5py.h5d.create(f.id, b"late_never", h5py.h5t.STD_I32LE, space, dcpl=dcpl)


def write_v1_btree_3d_chunks(path: Path) -> None:
    data = np.arange(4 * 5 * 6, dtype=np.int32).reshape(4, 5, 6)
    with h5py.File(path, "w", libver="earliest") as f:
        f.create_dataset("btree_v1_3d", data=data, chunks=(2, 2, 3))


def write_v1_btree_deflate_parallel_threshold_tail(path: Path) -> None:
    chunk = 2048
    length = chunk * 8 + 17
    data = (np.arange(length, dtype=np.int32) * 3) - 7
    with h5py.File(path, "w", libver="earliest") as f:
        f.create_dataset(
            "btree_v1_deflate_parallel_threshold_tail",
            data=data,
            chunks=(chunk,),
            compression="gzip",
            compression_opts=4,
        )


def write_v1_btree_sparse_nonmonotonic(path: Path) -> None:
    with h5py.File(path, "w", libver="earliest") as f:
        ds = f.create_dataset(
            "btree_v1_sparse_nonmonotonic",
            shape=(6, 6),
            dtype=np.int32,
            chunks=(2, 2),
            fillvalue=-9,
        )
        ds[4:6, 4:6] = np.array([[44, 45], [54, 55]], dtype=np.int32)
        ds[0:2, 0:2] = np.array([[0, 1], [10, 11]], dtype=np.int32)
        ds[2:4, 2:4] = np.array([[22, 23], [32, 33]], dtype=np.int32)


def write_v1_btree_full_leaf_gap(path: Path) -> None:
    chunk = 5
    with h5py.File(path, "w", libver="earliest") as f:
        ds = f.create_dataset(
            "btree_v1_full_leaf_gap",
            shape=(chunk * 65,),
            dtype=np.int32,
            chunks=(chunk,),
            fillvalue=-1,
        )
        ds[: chunk * 64] = np.arange(chunk * 64, dtype=np.int32)


def write_virtual_all(vds_path: Path, source_path: Path) -> None:
    data = np.arange(24, dtype=np.int32).reshape(4, 6)
    with h5py.File(source_path, "w", libver="latest") as f:
        f.create_dataset("source", data=data)

    space = h5py.h5s.create_simple(data.shape)
    source_space = h5py.h5s.create_simple(data.shape)
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dcpl.set_virtual(
        space,
        source_path.name.encode("utf-8"),
        b"/source",
        source_space,
    )

    with h5py.File(vds_path, "w", libver="latest") as f:
        h5py.h5d.create(f.id, b"vds_all", h5py.h5t.STD_I32LE, space, dcpl=dcpl)


def write_virtual_same_file(path: Path) -> None:
    data = np.arange(12, dtype=np.int32).reshape(3, 4)
    space = h5py.h5s.create_simple(data.shape)
    source_space = h5py.h5s.create_simple(data.shape)
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dcpl.set_virtual(space, b".", b"/source", source_space)

    with h5py.File(path, "w", libver="latest") as f:
        f.create_dataset("source", data=data)
        h5py.h5d.create(f.id, b"vds_same_file", h5py.h5t.STD_I32LE, space, dcpl=dcpl)


def write_virtual_mixed_all_regular(vds_path: Path, source_path: Path) -> None:
    data = np.arange(6, dtype=np.int32).reshape(2, 3)
    with h5py.File(source_path, "w", libver="latest") as f:
        f.create_dataset("source", data=data)

    dataset_space = h5py.h5s.create_simple((4, 6))
    virtual_space = h5py.h5s.create_simple((4, 6))
    virtual_space.select_hyperslab((1, 2), (1, 1), block=data.shape)
    source_space = h5py.h5s.create_simple(data.shape)
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dcpl.set_virtual(
        virtual_space,
        source_path.name.encode("utf-8"),
        b"/source",
        source_space,
    )

    with h5py.File(vds_path, "w", libver="latest") as f:
        h5py.h5d.create(
            f.id,
            b"vds_mixed_all_regular",
            h5py.h5t.STD_I32LE,
            dataset_space,
            dcpl=dcpl,
        )


def write_virtual_fill_value(vds_path: Path, source_path: Path) -> None:
    data = np.arange(6, dtype=np.int32).reshape(2, 3)
    with h5py.File(source_path, "w", libver="latest") as f:
        f.create_dataset("source", data=data)

    dataset_space = h5py.h5s.create_simple((4, 6))
    virtual_space = h5py.h5s.create_simple((4, 6))
    virtual_space.select_hyperslab((1, 2), (1, 1), block=data.shape)
    source_space = h5py.h5s.create_simple(data.shape)
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dcpl.set_fill_value(np.array(-7, dtype=np.int32))
    dcpl.set_virtual(
        virtual_space,
        source_path.name.encode("utf-8"),
        b"/source",
        source_space,
    )

    with h5py.File(vds_path, "w", libver="latest") as f:
        h5py.h5d.create(
            f.id,
            b"vds_fill_value",
            h5py.h5t.STD_I32LE,
            dataset_space,
            dcpl=dcpl,
        )


def write_virtual_f64(vds_path: Path, source_path: Path) -> None:
    data = (np.arange(12, dtype=np.float64).reshape(3, 4) / 2.0) + 0.25
    with h5py.File(source_path, "w", libver="latest") as f:
        f.create_dataset("source", data=data)

    space = h5py.h5s.create_simple(data.shape)
    source_space = h5py.h5s.create_simple(data.shape)
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dcpl.set_virtual(
        space,
        source_path.name.encode("utf-8"),
        b"/source",
        source_space,
    )

    with h5py.File(vds_path, "w", libver="latest") as f:
        h5py.h5d.create(f.id, b"vds_f64", h5py.h5t.IEEE_F64LE, space, dcpl=dcpl)


def write_virtual_scalar(vds_path: Path, source_path: Path) -> None:
    with h5py.File(source_path, "w", libver="latest") as f:
        f.create_dataset("source", data=np.array(42, dtype=np.int32))

    space = h5py.h5s.create(h5py.h5s.SCALAR)
    source_space = h5py.h5s.create(h5py.h5s.SCALAR)
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dcpl.set_virtual(space, source_path.name.encode("utf-8"), b"/source", source_space)

    with h5py.File(vds_path, "w", libver="latest") as f:
        h5py.h5d.create(f.id, b"vds_scalar", h5py.h5t.STD_I32LE, space, dcpl=dcpl)


def write_virtual_zero_sized(vds_path: Path, source_path: Path) -> None:
    with h5py.File(source_path, "w", libver="latest") as f:
        f.create_dataset(
            "source",
            shape=(0, 4),
            maxshape=(None, 4),
            dtype=np.int32,
            chunks=(1, 4),
        )

    space = h5py.h5s.create_simple((0, 4), (h5py.h5s.UNLIMITED, 4))
    source_space = h5py.h5s.create_simple((0, 4), (h5py.h5s.UNLIMITED, 4))
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dcpl.set_virtual(space, source_path.name.encode("utf-8"), b"/source", source_space)

    with h5py.File(vds_path, "w", libver="latest") as f:
        h5py.h5d.create(f.id, b"vds_zero_sized", h5py.h5t.STD_I32LE, space, dcpl=dcpl)


def write_virtual_null(vds_path: Path, source_path: Path) -> None:
    null_space = h5py.h5s.create(h5py.h5s.NULL)
    with h5py.File(source_path, "w", libver="latest") as f:
        h5py.h5d.create(f.id, b"source", h5py.h5t.STD_I32LE, null_space)

    space = h5py.h5s.create(h5py.h5s.NULL)
    source_space = h5py.h5s.create(h5py.h5s.NULL)
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dcpl.set_virtual(space, source_path.name.encode("utf-8"), b"/source", source_space)

    with h5py.File(vds_path, "w", libver="latest") as f:
        h5py.h5d.create(f.id, b"vds_null", h5py.h5t.STD_I32LE, space, dcpl=dcpl)


def write_virtual_rank_mismatch(vds_path: Path, source_path: Path) -> None:
    data = np.arange(6, dtype=np.int32)
    with h5py.File(source_path, "w", libver="latest") as f:
        f.create_dataset("source", data=data)

    dataset_space = h5py.h5s.create_simple((2, 3))
    virtual_space = h5py.h5s.create_simple((2, 3))
    source_space = h5py.h5s.create_simple((6,))
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dcpl.set_virtual(
        virtual_space,
        source_path.name.encode("utf-8"),
        b"/source",
        source_space,
    )

    with h5py.File(vds_path, "w", libver="latest") as f:
        h5py.h5d.create(
            f.id,
            b"vds_rank_mismatch",
            h5py.h5t.STD_I32LE,
            dataset_space,
            dcpl=dcpl,
        )


def write_virtual_overlap(vds_path: Path, source_a_path: Path, source_b_path: Path) -> None:
    with h5py.File(source_a_path, "w", libver="latest") as f:
        f.create_dataset("source", data=np.array([1, 2, 3, 4], dtype=np.int32))
    with h5py.File(source_b_path, "w", libver="latest") as f:
        f.create_dataset("source", data=np.array([9, 8], dtype=np.int32))

    dataset_space = h5py.h5s.create_simple((4,))
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)

    virtual_a = h5py.h5s.create_simple((4,))
    source_a = h5py.h5s.create_simple((4,))
    dcpl.set_virtual(
        virtual_a,
        source_a_path.name.encode("utf-8"),
        b"/source",
        source_a,
    )

    virtual_b = h5py.h5s.create_simple((4,))
    virtual_b.select_hyperslab((1,), (1,), block=(2,))
    source_b = h5py.h5s.create_simple((2,))
    dcpl.set_virtual(
        virtual_b,
        source_b_path.name.encode("utf-8"),
        b"/source",
        source_b,
    )

    with h5py.File(vds_path, "w", libver="latest") as f:
        h5py.h5d.create(f.id, b"vds_overlap", h5py.h5t.STD_I32LE, dataset_space, dcpl=dcpl)


def write_virtual_irregular_hyperslab(vds_path: Path, source_path: Path) -> None:
    data = np.arange(16, dtype=np.int32).reshape(4, 4)
    with h5py.File(source_path, "w", libver="latest") as f:
        f.create_dataset("source", data=data)

    dataset_space = h5py.h5s.create_simple((4, 4))
    virtual_space = h5py.h5s.create_simple((4, 4))
    source_space = h5py.h5s.create_simple((4, 4))
    for space in (virtual_space, source_space):
        space.select_hyperslab((0, 1), (1, 1), block=(1, 2), op=h5py.h5s.SELECT_SET)
        space.select_hyperslab((2, 0), (1, 1), block=(1, 2), op=h5py.h5s.SELECT_OR)

    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dcpl.set_fill_value(np.array(-2, dtype=np.int32))
    dcpl.set_virtual(
        virtual_space,
        source_path.name.encode("utf-8"),
        b"/source",
        source_space,
    )

    with h5py.File(vds_path, "w", libver="latest") as f:
        h5py.h5d.create(
            f.id,
            b"vds_irregular_hyperslab",
            h5py.h5t.STD_I32LE,
            dataset_space,
            dcpl=dcpl,
        )


def point_selection_bytes(rank: int, points: list[tuple[int, ...]]) -> bytes:
    if rank <= 0:
        raise ValueError("point selection rank must be positive")
    if not points:
        raise ValueError("point selection fixture requires at least one point")
    if any(len(point) != rank for point in points):
        raise ValueError("point coordinates must match selection rank")

    max_value = max(max(point) for point in points)
    max_value = max(max_value, len(points))
    if max_value <= 0xFFFF:
        enc_size = 2
    elif max_value <= 0xFFFFFFFF:
        enc_size = 4
    else:
        enc_size = 8

    out = bytearray()
    out.extend((1).to_bytes(4, "little"))  # H5S_SEL_POINTS
    out.extend((2).to_bytes(4, "little"))  # point-selection version 2
    out.append(enc_size)
    out.extend(rank.to_bytes(4, "little"))
    out.extend(len(points).to_bytes(enc_size, "little"))
    for point in points:
        for coord in point:
            out.extend(coord.to_bytes(enc_size, "little"))
    return bytes(out)


def rewrite_single_global_heap_object(path: Path, object_data: bytes) -> None:
    raw = bytearray(path.read_bytes())
    heap_addr = raw.find(GLOBAL_HEAP_MAGIC)
    if heap_addr < 0:
        raise ValueError(f"global heap collection not found in {path}")

    object_header = heap_addr + 16
    object_index = int.from_bytes(raw[object_header:object_header + 2], "little")
    if object_index != 1:
        raise ValueError(f"expected global heap object index 1, got {object_index}")

    original_size = int.from_bytes(raw[object_header + 8:object_header + 16], "little")
    padded_size = (original_size + 7) & ~7
    if len(object_data) > padded_size:
        raise ValueError(
            f"patched global heap object ({len(object_data)} bytes) exceeds padded slot {padded_size}"
        )

    raw[object_header + 8:object_header + 16] = len(object_data).to_bytes(8, "little")
    data_start = object_header + 16
    data_end = data_start + padded_size
    raw[data_start:data_start + len(object_data)] = object_data
    raw[data_start + len(object_data):data_end] = b"\x00" * (data_end - (data_start + len(object_data)))
    path.write_bytes(raw)


def write_virtual_point_selection(vds_path: Path, source_path: Path) -> None:
    data = np.arange(16, dtype=np.int32).reshape(4, 4)
    with h5py.File(source_path, "w", libver="latest") as f:
        f.create_dataset("source", data=data)

    dataset_space = h5py.h5s.create_simple((4, 4))
    virtual_space = h5py.h5s.create_simple((4, 4))
    source_space = h5py.h5s.create_simple((4, 4))
    for space in (virtual_space, source_space):
        space.select_hyperslab((0, 1), (1, 1), block=(1, 2), op=h5py.h5s.SELECT_SET)
        space.select_hyperslab((2, 0), (1, 1), block=(1, 2), op=h5py.h5s.SELECT_OR)

    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dcpl.set_fill_value(np.array(-2, dtype=np.int32))
    dcpl.set_virtual(
        virtual_space,
        source_path.name.encode("utf-8"),
        b"/source",
        source_space,
    )

    with h5py.File(vds_path, "w", libver="latest") as f:
        h5py.h5d.create(
            f.id,
            b"vds_point_selection",
            h5py.h5t.STD_I32LE,
            dataset_space,
            dcpl=dcpl,
        )

    source_points = point_selection_bytes(2, [(2, 1)])
    virtual_points = point_selection_bytes(2, [(0, 3)])
    heap_object = (
        bytes([0]) +
        (1).to_bytes(8, "little") +
        source_path.name.encode("utf-8") + b"\x00" +
        b"/source\x00" +
        source_points +
        virtual_points
    )
    rewrite_single_global_heap_object(vds_path, heap_object)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"h5py version: {h5py.__version__}")
    print(f"HDF5 C library version: {h5py.version.hdf5_version}")

    generated: list[Path] = []

    def record(path: Path) -> Path:
        generated.append(path)
        return path

    write_fixed_array(record(OUT / "v4_fixed_array_chunks.h5"))
    write_fixed_array_deflate_parallel_threshold_tail(
        record(OUT / "v4_fixed_array_deflate_parallel_threshold_tail.h5")
    )
    write_fixed_array_deflate_mask_parallel_fallback(
        record(OUT / "v4_fixed_array_deflate_mask_parallel_fallback.h5")
    )
    write_fixed_array_3d_edges(record(OUT / "v4_fixed_array_3d_edges.h5"))
    write_paged_fixed_array(record(OUT / "v4_fixed_array_paged_chunks.h5"))
    write_paged_fixed_array_sparse(record(OUT / "v4_fixed_array_paged_sparse.h5"))
    write_extensible_array(record(OUT / "v4_extensible_array_chunks.h5"))
    write_extensible_array_deflate_parallel_threshold_tail(
        record(OUT / "v4_extensible_array_deflate_parallel_threshold_tail.h5")
    )
    write_extensible_array_deflate_mask_parallel_fallback(
        record(OUT / "v4_extensible_array_deflate_mask_parallel_fallback.h5")
    )
    write_extensible_array_2d_unlimited_edges(
        record(OUT / "v4_extensible_array_2d_unlimited_edges.h5")
    )
    write_extensible_array_spillover(record(OUT / "v4_extensible_array_spillover.h5"))
    write_extensible_array_sparse_transitions(
        record(OUT / "v4_extensible_array_sparse_transitions.h5")
    )
    write_v2_btree(record(OUT / "v4_btree2_chunks.h5"))
    write_v2_btree_internal(record(OUT / "v4_btree2_internal_chunks.h5"))
    write_v2_btree_deep_internal(record(OUT / "v4_btree2_deep_internal_chunks.h5"))
    write_v2_btree_filtered_mask(record(OUT / "v4_btree2_filtered_mask.h5"))
    write_filtered_implicit(record(OUT / "v4_filtered_chunked.h5"))
    write_implicit_2d_edge_chunks(record(OUT / "v4_implicit_2d_edge_chunks.h5"))
    write_sparse_chunked_fill_value(record(OUT / "sparse_chunked_fill_value.h5"))
    write_filtered_chunk_filter_mask(record(OUT / "filtered_chunk_filter_mask.h5"))
    write_filtered_single_chunk_filter_mask(record(OUT / "filtered_single_chunk_filter_mask.h5"))
    write_filtered_middle_filter_mask(record(OUT / "filtered_middle_filter_mask.h5"))
    write_multi_filter_orders(record(OUT / "multi_filter_orders.h5"))
    write_fletcher32_corrupt(record(OUT / "fletcher32_corrupt.h5"))
    write_nbit_filter(record(OUT / "nbit_filter_i32.h5"))
    write_nbit_filter_be(record(OUT / "nbit_filter_be_i32.h5"))
    write_nbit_parity_vectors(record(OUT / "nbit_parity_vectors.h5"))
    write_scaleoffset_filter(record(OUT / "scaleoffset_filter_i32.h5"))
    write_scaleoffset_filter_be(record(OUT / "scaleoffset_filter_be_i32.h5"))
    write_scaleoffset_parity_vectors(record(OUT / "scaleoffset_parity_vectors.h5"))
    write_integer_conversion_vectors(record(OUT / "integer_conversion_vectors.h5"))
    write_float_conversion_vectors(record(OUT / "float_conversion_vectors.h5"))
    write_fixed_string_cases(record(OUT / "fixed_string_cases.h5"))
    write_vlen_string_cases(record(OUT / "vlen_string_cases.h5"))
    write_opaque_cases(record(OUT / "opaque_cases.h5"))
    write_reference_cases(record(OUT / "reference_cases.h5"))
    write_time_cases(record(OUT / "time_cases.h5"))
    write_enum_conversion_cases(record(OUT / "enum_conversion_cases.h5"))
    write_fractal_heap_cases(record(OUT / "fractal_heap_modern.h5"))
    write_dense_group_cases(record(OUT / "dense_group_cases.h5"))
    write_attribute_cases(record(OUT / "attribute_cases.h5"))
    write_link_edge_cases(record(OUT / "link_edge_cases.h5"))
    write_complex_compound(record(OUT / "compound_complex.h5"))
    write_compound_layout_cases(record(OUT / "compound_layout_cases.h5"))
    write_array_datatype_cases(record(OUT / "array_datatype_cases.h5"))
    write_compact_read_cases(record(OUT / "compact_read_cases.h5"))
    write_external_raw_storage(
        record(OUT / "external_raw_storage.h5"),
        record(OUT / "external_raw_storage.bin"),
    )
    write_undefined_storage_address(record(OUT / "undefined_storage_address.h5"))
    write_late_fill_time_never(record(OUT / "late_fill_time_never.h5"))
    write_v1_btree_3d_chunks(record(OUT / "v1_btree_3d_chunks.h5"))
    write_v1_btree_deflate_parallel_threshold_tail(
        record(OUT / "v1_btree_deflate_parallel_threshold_tail.h5")
    )
    write_v1_btree_sparse_nonmonotonic(record(OUT / "v1_btree_sparse_nonmonotonic.h5"))
    write_v1_btree_full_leaf_gap(record(OUT / "v1_btree_full_leaf_gap.h5"))
    write_virtual_all(
        record(OUT / "vds_all.h5"),
        record(OUT / "vds_all_source.h5"),
    )
    write_virtual_same_file(record(OUT / "vds_same_file.h5"))
    write_virtual_mixed_all_regular(
        record(OUT / "vds_mixed_all_regular.h5"),
        record(OUT / "vds_mixed_all_regular_source.h5"),
    )
    write_virtual_fill_value(
        record(OUT / "vds_fill_value.h5"),
        record(OUT / "vds_fill_value_source.h5"),
    )
    write_virtual_f64(
        record(OUT / "vds_f64.h5"),
        record(OUT / "vds_f64_source.h5"),
    )
    write_virtual_scalar(
        record(OUT / "vds_scalar.h5"),
        record(OUT / "vds_scalar_source.h5"),
    )
    write_virtual_zero_sized(
        record(OUT / "vds_zero_sized.h5"),
        record(OUT / "vds_zero_sized_source.h5"),
    )
    write_virtual_null(
        record(OUT / "vds_null.h5"),
        record(OUT / "vds_null_source.h5"),
    )
    write_virtual_rank_mismatch(
        record(OUT / "vds_rank_mismatch.h5"),
        record(OUT / "vds_rank_mismatch_source.h5"),
    )
    write_virtual_overlap(
        record(OUT / "vds_overlap.h5"),
        record(OUT / "vds_overlap_source_a.h5"),
        record(OUT / "vds_overlap_source_b.h5"),
    )
    write_virtual_irregular_hyperslab(
        record(OUT / "vds_irregular_hyperslab.h5"),
        record(OUT / "vds_irregular_hyperslab_source.h5"),
    )
    write_virtual_point_selection(
        record(OUT / "vds_point_selection.h5"),
        record(OUT / "vds_point_selection_source.h5"),
    )

    print("Regenerated local fixtures:")
    for path in generated:
        print(f"  {path}")


if __name__ == "__main__":
    main()
