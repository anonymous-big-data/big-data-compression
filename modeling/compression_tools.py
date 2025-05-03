import numpy as np
import zstandard as zstd
import zlib
import bz2

import lzma as lz2
import lz4.frame as fastlz
from dahuffman import HuffmanCodec
import numpy as np
import blosc

import numpy as np
import blosc2
from blosc2 import Codec, Filter

def get_codec(codec):
    """
    Converts a codec string (e.g., 'zstd') to its corresponding blosc2.Codec enum.
    """
    if isinstance(codec, str):
        codec_upper = codec.upper()
        if codec_upper == 'ZSTD':
            return Codec.ZSTD
        elif codec_upper == 'LZ4':
            return Codec.LZ4
        elif codec_upper == 'LZ4HC':
            return Codec.LZ4HC
        elif codec_upper == 'BLOSCLZ':
            return Codec.BLOSCLZ
        elif codec_upper == 'ZLIB':
            return Codec.ZLIB
        else:
            raise ValueError("Unsupported codec: " + codec)
    return codec

def blosc_comp_No(data, clevel=3, codec='zstd'):
    """
    Compresses data using blosc2 without applying any filter.
    Uses blosc2.Filter.NOFILTER to disable filtering.
    """
    if isinstance(data, np.ndarray):
        typesize = data.dtype.itemsize  # e.g., 4 for float32.
        data_bytes = data.tobytes()
    else:
        typesize = 1
        data_bytes = data

    codec_enum = get_codec(codec)
    # Use Filter.NOFILTER for no filtering.
    return blosc2.compress(data_bytes, typesize, clevel, Filter.NOFILTER, codec_enum)

def blosc_comp(data, clevel=3, codec='zstd'):
    """
    Compresses data using blosc2 with the default shuffle filter.
    Uses blosc2.Filter.SHUFFLE.
    """
    if isinstance(data, np.ndarray):
        typesize = data.dtype.itemsize
        data_bytes = data.tobytes()
    else:
        typesize = 1
        data_bytes = data

    codec_enum = get_codec(codec)
    # Use Filter.SHUFFLE for the default shuffle.
    return blosc2.compress(data_bytes, typesize, clevel, Filter.SHUFFLE, codec_enum)

def blosc_comp_bit(data, clevel=3, codec='zstd'):
    """
    Compresses data using blosc2 with the bitshuffle filter.
    Uses blosc2.Filter.BITSHUFFLE.
    """
    if isinstance(data, np.ndarray):
        typesize = data.dtype.itemsize
        data_bytes = data.tobytes()
    else:
        typesize = 1
        data_bytes = data

    codec_enum = get_codec(codec)
    # Use Filter.BITSHUFFLE for bitshuffle filtering.
    return blosc2.compress(data_bytes, typesize, clevel, Filter.BITSHUFFLE, codec_enum)
def zstd_comp(data_set):
    return zstd.compress(data_set, 3)


def zlib_comp(data, level=9):
    return zlib.compress(data, level)

def bz2_comp(data, level=9):
    return bz2.compress(data, level)


# def lzma_comp(data, level=9):
#     return lz2.compress(data, level)

def fastlz_compress(data):
    return  fastlz.compress(data)


def rle_compress(data):

    if isinstance(data, np.ndarray):
        data = data.tobytes()
    if not data:
        return b'', 0
    compressed_data = []
    count = 1
    last = data[0]
    for current in data[1:]:
        if current == last and count < 255:
            count += 1
        else:
            compressed_data.extend([count, last])
            last = current
            count = 1
    compressed_data.extend([count, last])
    compressed_bytes = bytes(compressed_data)
    #compression_ratio = len(data) / len(compressed_bytes) if compressed_bytes else 1
    return compressed_bytes

def huffman_compress(data):
    # Assume data is a bytes
    codec = HuffmanCodec.from_data(data)
    compressed_data = codec.encode(data)
    return compressed_data