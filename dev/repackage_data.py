"""
将Fineweb-Edu-Chinese-V2.1数据集中 4_5 目录数据重新分片打包：

- 每个分片大约2.5亿个字符，由于是中文字符，所以每个分片大约200MB（经过zstd压缩后）
- parquet文件按1000行的行组大小写入
- 对数据集进行随机打乱

这些分片将上传至HuggingFace平台托管。
关键优势在于我们的数据加载器能够流式读取数据，
并在磁盘上实时缓存，从而显著降低训练延迟。

载入 Datasets 时注意huggingface 的限额
（https://huggingface.co/docs/hub/en/rate-limits#rate-limit-tiers）

NOTE: This file is meant only as reference/documentation of the
dataset preparation and it is not used during the project runtime.
"""
import os
import time

from datasets import load_dataset
import pyarrow.parquet as pq
import pyarrow as pa

import os
os.environ["HF_TOKEN"] = "hf_xxx"

# Source dataset
dataset_kwargs = {
    "path": "opencsg/Fineweb-Edu-Chinese-V2.1",
    "split": "train",
    "data_dir": "4_5",
    "cache_dir": "./.cache"
}
ds = load_dataset(**dataset_kwargs)

# Shuffle to scramble the order
ds = ds.shuffle(seed=42)
ndocs = len(ds) # total number of documents to process
print(f"Total number of documents: {ndocs}")

# Repackage into parquet files
output_dir = "./.cache/nanochat/base_data"
os.makedirs(output_dir, exist_ok=True)

# Write to parquet files
chars_per_shard = 250_000_000
row_group_size = 1024 # HF uses 1000 but we use multiple of 2, nicer for distributed data loader later
shard_docs = []
shard_index = 0
shard_characters = 0
total_docs_processed = 0
total_time_spent = 0
t0 = time.time()
for doc in ds:
    text = doc['text']
    shard_docs.append(text)
    shard_characters += len(text)
    collected_enough_chars = shard_characters >= chars_per_shard
    docs_multiple_of_row_group_size = len(shard_docs) % row_group_size == 0
    if collected_enough_chars and docs_multiple_of_row_group_size: # leads to ~100MB of text (compressed)
        shard_path = os.path.join(output_dir, f"shard_{shard_index:05d}.parquet")
        shard_table = pa.Table.from_pydict({"text": shard_docs})
        pq.write_table(
            shard_table,
            shard_path,
            row_group_size=row_group_size,
            use_dictionary=False, # this is usually used for categorical data
            compression="zstd", # Valid values: {‘NONE’, ‘SNAPPY’, ‘GZIP’, ‘BROTLI’, ‘LZ4’, ‘ZSTD’}
            compression_level=3,
            write_statistics=False, # not needed for text
        )
        t1 = time.time()
        dt = t1 - t0 # for this shard alone
        t0 = t1
        total_docs_processed += len(shard_docs)
        total_time_spent += dt
        remaining_docs = ndocs - total_docs_processed
        avg_time_per_doc = total_time_spent / total_docs_processed
        remaining_time = remaining_docs * avg_time_per_doc
        remaining_time_hours = remaining_time / 3600
        print(f"Wrote {shard_path}. #documents: {len(shard_docs)} | #characters: {shard_characters} | time: {dt:.2f}s | remaining time: {remaining_time_hours:.2f}h")
        shard_docs = []
        shard_characters = 0
        shard_index += 1

# Demonstration of how the data was later uploaded to HuggingFace
def upload():
    import os
    from huggingface_hub import HfApi
    token = os.getenv("HF_TOKEN")
    api = HfApi(token=token)
    api.upload_large_folder(
        folder_path=output_dir,
        repo_id="mrhuangtalk/fineweb-edu-chinese-shuffle",
        repo_type="dataset",
    )
# upload()
