#!/usr/bin/env python3
"""Round-trip test for pngquant's metadata, alpha and palette behavior.

Quantizes the repository's metadata PNG (and generated edge-case images)
and compares ancillary chunks, per-pixel alpha, file headers and exit codes
between input and output.

Usage: python3 test/roundtrip.py <pngquant-binary> <path-to-metadata.png>

Exits 0 if all checks pass, 1 otherwise. Stdlib only, no image libraries.
"""

import os
import struct
import subprocess
import sys
import tempfile
import zlib

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# Ancillary chunks that carry metadata (PLTE/tRNS are image data, not metadata)
METADATA_CHUNKS = ("tEXt", "zTXt", "iTXt", "eXIf", "iCCP", "cHRM", "gAMA", "sRGB", "pHYs")

failures = []


def check(condition, message):
    print(("PASS" if condition else "FAIL"), "-", message)
    if not condition:
        failures.append(message)


# ---------------------------------------------------------------- PNG I/O

def make_chunk(ctype, payload):
    return (struct.pack(">I", len(payload)) + ctype + payload +
            struct.pack(">I", zlib.crc32(ctype + payload) & 0xFFFFFFFF))


def write_png(path, width, height, rows, extra_chunks=()):
    """rows: list of rows of (r, g, b, a) tuples; writes 8-bit RGBA."""
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    raw = b"".join(b"\x00" + b"".join(struct.pack("4B", *px) for px in row) for row in rows)
    data = PNG_SIGNATURE + make_chunk(b"IHDR", ihdr)
    for ctype, payload in extra_chunks:
        data += make_chunk(ctype, payload)
    data += make_chunk(b"IDAT", zlib.compress(raw)) + make_chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(data)


def parse_png(path):
    """Returns (header, chunks) where header is the decoded IHDR tuple and
    chunks is an ordered list of (type, payload)."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != PNG_SIGNATURE:
        raise ValueError("%s: bad PNG signature" % path)
    chunks = []
    pos = 8
    while pos < len(data):
        length, ctype = struct.unpack(">I4s", data[pos:pos + 8])
        payload = data[pos + 8:pos + 8 + length]
        crc_stored, = struct.unpack(">I", data[pos + 8 + length:pos + 12 + length])
        if zlib.crc32(ctype + payload) & 0xFFFFFFFF != crc_stored:
            raise ValueError("%s: CRC mismatch in %s" % (path, ctype))
        chunks.append((ctype.decode("latin1"), payload))
        pos += 12 + length
    if chunks[0][0] != "IHDR" or chunks[-1][0] != "IEND":
        raise ValueError("%s: missing IHDR/IEND" % path)
    header = struct.unpack(">IIBBBBB", chunks[0][1])
    return header, chunks


def _unfilter(raw, pos, stride, bpp, prev):
    filt = raw[pos]
    pos += 1
    line = bytearray(raw[pos:pos + stride])
    pos += stride
    for x in range(stride):
        a = line[x - bpp] if x >= bpp else 0
        b = prev[x]
        c = prev[x - bpp] if x >= bpp else 0
        if filt == 1:
            line[x] = (line[x] + a) & 255
        elif filt == 2:
            line[x] = (line[x] + b) & 255
        elif filt == 3:
            line[x] = (line[x] + (a + b) // 2) & 255
        elif filt == 4:
            p = a + b - c
            pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
            pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
            line[x] = (line[x] + pr) & 255
        elif filt != 0:
            raise ValueError("unknown filter %d" % filt)
    return line, pos


def read_pixels(path):
    """Decodes 8-bit RGBA (color type 6) and palette (color type 3) PNGs.
    Returns (width, height, rows of (r, g, b, a))."""
    (width, height, bit_depth, color_type, _, _, _), chunks = parse_png(path)
    idat = b"".join(payload for ctype, payload in chunks if ctype == "IDAT")
    raw = zlib.decompress(idat)
    rows = []
    if color_type == 6 and bit_depth == 8:
        stride = width * 4
        prev = bytearray(stride)
        pos = 0
        for _y in range(height):
            line, pos = _unfilter(raw, pos, stride, 4, prev)
            prev = line
            rows.append([tuple(line[x * 4:x * 4 + 4]) for x in range(width)])
    elif color_type == 3:
        plte = next(p for c, p in chunks if c == "PLTE")
        trns = next((p for c, p in chunks if c == "tRNS"), None)
        stride = (width * bit_depth + 7) // 8
        prev = bytearray(stride)
        pos = 0
        for _y in range(height):
            line, pos = _unfilter(raw, pos, stride, 1, prev)
            prev = line
            row = []
            for x in range(width):
                if bit_depth == 8:
                    idx = line[x]
                elif bit_depth == 4:
                    idx = (line[x // 2] >> (4 * (1 - x % 2))) & 15
                elif bit_depth == 2:
                    idx = (line[x // 4] >> (2 * (3 - x % 4))) & 3
                elif bit_depth == 1:
                    idx = (line[x // 8] >> (7 - x % 8)) & 1
                else:
                    raise ValueError("unsupported palette bit depth %d" % bit_depth)
                r, g, b = plte[idx * 3:idx * 3 + 3]
                a = trns[idx] if trns and idx < len(trns) else 255
                row.append((r, g, b, a))
            rows.append(row)
    else:
        raise ValueError("unsupported color type %d bit depth %d" % (color_type, bit_depth))
    return width, height, rows


def chunk_map(chunks):
    return {ctype: payload for ctype, payload in chunks}


# ---------------------------------------------------------------- test driver

def run(binary, args, expect_exit=0, label=""):
    proc = subprocess.run([binary] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    check(proc.returncode == expect_exit,
          "%s: exit code %d (expected %d)" % (label or " ".join(args), proc.returncode, expect_exit))
    return proc


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    binary, metadata_png = sys.argv[1], sys.argv[2]

    src_header, src_chunks = parse_png(metadata_png)
    src_w, src_h, src_rows = read_pixels(metadata_png)
    src_meta = chunk_map(src_chunks)
    print("source: %s %dx%d, chunks: %s" %
          (metadata_png, src_w, src_h, [c for c, _ in src_chunks]))

    with tempfile.TemporaryDirectory(prefix="pngquant-roundtrip") as tmp:
        def quantize(name, extra_args, expect_exit=0):
            out = os.path.join(tmp, name)
            run(binary, ["--force", "-o", out] + extra_args + [metadata_png], expect_exit, name)
            return out

        # -- file header + exit code, default mode ------------------------------
        out = quantize("default.png", [])
        (w, h, depth, ctype, comp, filt, inter), chunks = parse_png(out)
        check((w, h) == (src_header[0], src_header[1]), "default: width/height preserved")
        check(ctype == 3, "default: output is palette (color type 3)")
        check(depth in (1, 2, 4, 8), "default: palette bit depth %d is valid" % depth)
        check(comp == 0 and filt == 0 and inter == 0, "default: compression/filter/interlace are 0")

        # -- ancillary chunks, default mode -------------------------------------
        out_meta = chunk_map(chunks)
        check(out_meta.get("iTXt") == src_meta.get("iTXt"), "default: iTXt copied byte-identical")
        check("sRGB" in out_meta, "default: color-managed output carries sRGB chunk")
        check("iCCP" not in out_meta, "default: iCCP consumed by color management (not copied)")

        # -- per-pixel alpha, default mode --------------------------------------
        _, _, out_rows = read_pixels(out)
        alpha_mismatch = sum(1 for y in range(src_h) for x in range(src_w)
                             if src_rows[y][x][3] != out_rows[y][x][3])
        check(alpha_mismatch == 0, "default: per-pixel alpha unchanged (%d mismatches)" % alpha_mismatch)

        # -- ancillary chunks, --keep-metadata ----------------------------------
        out = quantize("keep.png", ["--keep-metadata"])
        keep_meta = chunk_map(parse_png(out)[1])
        for ctype in ("gAMA", "iCCP", "cHRM", "iTXt"):
            if ctype in src_meta:
                check(keep_meta.get(ctype) == src_meta[ctype],
                      "keep-metadata: %s copied byte-identical" % ctype)
        # pixels were not color-transformed: with <=256 colors the round trip is exact
        _, _, keep_rows = read_pixels(out)
        pixel_mismatch = sum(1 for y in range(src_h) for x in range(src_w)
                             if src_rows[y][x] != keep_rows[y][x])
        check(pixel_mismatch == 0, "keep-metadata: per-pixel RGBA round trip exact (%d mismatches)" % pixel_mismatch)

        # -- ancillary chunks, --strip ------------------------------------------
        out = quantize("strip.png", ["--strip"])
        strip_meta = chunk_map(parse_png(out)[1])
        leftover = [c for c in METADATA_CHUNKS if c in strip_meta and c in src_meta]
        check(not leftover, "strip: no source metadata chunks left (%s)" % leftover)

        # -- palette results are deterministic ----------------------------------
        out1 = quantize("det1.png", [])
        out2 = quantize("det2.png", [])
        with open(out1, "rb") as f1, open(out2, "rb") as f2:
            check(f1.read() == f2.read(), "palette: identical input gives byte-identical output")

        # -- conflicting options rejected ---------------------------------------
        run(binary, ["--strip", "--keep-metadata", "-o", os.path.join(tmp, "x.png"), metadata_png],
            expect_exit=4, label="strip+keep-metadata conflict")

        # -- generated sRGB image with a transparent border ---------------------
        border_png = os.path.join(tmp, "border.png")
        size = 16
        rows = []
        for y in range(size):
            row = []
            for x in range(size):
                if x in (0, size - 1) or y in (0, size - 1):
                    row.append((200, 30, 90, 0))  # fully transparent border
                else:
                    row.append(((x * 37) % 256, (y * 57) % 256, 128, 255))
            rows.append(row)
        write_png(border_png, size, size, rows, extra_chunks=[(b"sRGB", b"\x00")])

        out = os.path.join(tmp, "border-out.png")
        run(binary, ["--force", "--preserve-alpha", "-o", out, border_png], 0, "preserve-alpha")
        hdr, chunks = parse_png(out)
        check(hdr[0] == size and hdr[1] == size, "preserve-alpha: dimensions preserved")
        _, _, out_rows = read_pixels(out)
        leaked = [(x, y) for y in range(size) for x in range(size)
                  if rows[y][x][3] == 0 and out_rows[y][x][3] != 0]
        check(not leaked, "preserve-alpha: transparent border stays transparent (%d leaks)" % len(leaked))
        vanished = [(x, y) for y in range(size) for x in range(size)
                    if rows[y][x][3] == 255 and out_rows[y][x][3] == 0]
        check(not vanished, "preserve-alpha: opaque pixels stay opaque (%d vanished)" % len(vanished))

        # -- sRGB chunk survives --keep-metadata --------------------------------
        out = os.path.join(tmp, "border-keep.png")
        run(binary, ["--force", "--keep-metadata", "--preserve-alpha", "-o", out, border_png],
            0, "keep-metadata sRGB")
        keep_meta = chunk_map(parse_png(out)[1])
        check(keep_meta.get("sRGB") == b"\x00", "keep-metadata: sRGB chunk copied byte-identical")
        _, _, out_rows = read_pixels(out)
        leaked = sum(1 for y in range(size) for x in range(size)
                     if rows[y][x][3] == 0 and out_rows[y][x][3] != 0)
        check(leaked == 0, "keep-metadata+preserve-alpha: transparent border stays transparent")

        # -- eXIf and tEXt survive the default policy ----------------------------
        exif_png = os.path.join(tmp, "exif.png")
        exif_payload = b"II*\x00\x08\x00\x00\x00\x00\x00\x00\x00"
        text_payload = b"Comment\x00round-trip test"
        write_png(exif_png, 4, 4, [[(10, 200, 30, 255)] * 4 for _ in range(4)],
                  extra_chunks=[(b"eXIf", exif_payload), (b"tEXt", text_payload)])
        out = os.path.join(tmp, "exif-out.png")
        run(binary, ["--force", "-o", out, exif_png], 0, "default eXIf/tEXt")
        out_meta = chunk_map(parse_png(out)[1])
        check(out_meta.get("eXIf") == exif_payload, "default: eXIf copied byte-identical")
        check(out_meta.get("tEXt") == text_payload, "default: tEXt copied byte-identical")

        # -- exit codes ----------------------------------------------------------
        run(binary, [], expect_exit=1, label="no arguments")
        run(binary, ["--keep-metadata", "-o", os.path.join(tmp, "never-written.png"),
                     os.path.join(tmp, "does-not-exist.png")],
            expect_exit=2, label="missing input file")

    print()
    if failures:
        print("%d check(s) failed" % len(failures))
        return 1
    print("all round-trip checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
