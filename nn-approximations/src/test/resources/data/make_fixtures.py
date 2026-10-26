"""Regenerates the Parquet fixtures used by ParquetReaderTest and the image/table dataset tests.

Requirements: pip install pyarrow pillow
Run from this directory: python make_fixtures.py
"""
import io
import random

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


def png(kind, position, size=8):
    image = Image.new("L", (size, size), 0)
    for i in range(size):
        x, y = {0: (i, position), 1: (position, i), 2: (i, i)}[kind]
        image.putpixel((x, y), 255)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def images(count, seed):
    rng = random.Random(seed)
    rows = []
    for i in range(count):
        kind = i % 3
        rows.append({"image": {"bytes": png(kind, rng.randrange(1, 7)), "path": None}, "label": kind})
    image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    return pa.table({
        "image": pa.array([r["image"] for r in rows], type=image_type),
        "label": pa.array([r["label"] for r in rows], type=pa.int64()),
    })


def types(count):
    def maybe(i, value):
        return None if i % 5 == 0 else value
    return pa.table({
        "id": pa.array(range(count), type=pa.int32()),
        "big": pa.array([maybe(i, i * 1_000_000_007) for i in range(count)], type=pa.int64()),
        "ratio": pa.array([maybe(i, i / 4) for i in range(count)], type=pa.float64()),
        "small": pa.array([i * 0.5 for i in range(count)], type=pa.float32()),
        "name": pa.array([maybe(i, "row" + str(i % 7)) for i in range(count)], type=pa.string()),
        "flag": pa.array([i % 2 == 0 for i in range(count)], type=pa.bool_()),
    })


def flowers():
    """Iris-like table sorted by class, like scikit-learn/iris; row 7 has a missing feature."""
    rng = random.Random(5)
    centres = {"setosa": (5.0, 3.4, 1.5, 0.2), "versicolor": (5.9, 2.8, 4.3, 1.3), "virginica": (6.6, 3.0, 5.6, 2.0)}
    rows = []
    for species, centre in centres.items():
        for _ in range(20):
            rows.append([round(c + rng.gauss(0, 0.15), 2) for c in centre] + [species])
    rows[7][2] = None
    columns = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    table = {"Id": pa.array(range(1, len(rows) + 1), type=pa.int64())}
    for c, name in enumerate(columns):
        table[name] = pa.array([r[c] for r in rows], type=pa.float64())
    table["Species"] = pa.array([r[4] for r in rows], type=pa.string())
    return pa.table(table)


def houses(count, seed):
    """price = 50 * rooms + 2 * age + 100 + noise, with columns on very different scales."""
    rng = random.Random(seed)
    rooms = [round(rng.uniform(2, 9), 3) for _ in range(count)]
    age = [rng.randrange(1, 60) for _ in range(count)]
    price = [round(50 * r + 2 * a + 100 + rng.gauss(0, 1), 3) for r, a in zip(rooms, age)]
    return pa.table({"rooms": pa.array(rooms, type=pa.float64()), "age": pa.array(age, type=pa.int64()),
                     "price": pa.array(price, type=pa.float64())})


pq.write_table(flowers(), "flowers-train.parquet", compression="snappy", row_group_size=25)
houses_train = houses(80, 6)
pq.write_table(houses_train.slice(0, 50), "houses-train-0.parquet", compression="snappy")
pq.write_table(houses_train.slice(50), "houses-train-1.parquet", compression="gzip")
pq.write_table(houses(20, 7), "houses-test.parquet", compression="snappy")
pq.write_table(images(90, 1), "tiny-train.parquet", compression="snappy", row_group_size=32)
pq.write_table(images(30, 2), "tiny-test.parquet", compression="snappy", row_group_size=32)
pq.write_table(types(50), "types-snappy-v1.parquet", compression="snappy", data_page_version="1.0",
               row_group_size=20)
pq.write_table(types(50), "types-gzip-v2.parquet", compression="gzip", data_page_version="2.0",
               use_dictionary=False)
pq.write_table(types(50), "types-plain-v2.parquet", compression="none", data_page_version="2.0",
               use_dictionary=["name"])
