import polars as pl
import tensorflow as tf

def dapt_data_from_parquet(path_prefix, db_folder = "ldc_corpus"): 
    ldc_pq = pl.scan_parquet(f"{path_prefix}/{db_folder}/**/*.parquet", hive_partitioning = True)
    ldc_data = ldc_pq.select(pl.col(["id", "headline", "lead_paragraph"]))
    ldc_data = ldc_data.collect()
    # Replacing missing headlines and leads with empty strings
    ldc_data = ldc_data.with_columns(
        pl.when(pl.col.headline == "NA").then(pl.lit("")).otherwise(pl.col.headline).alias("headline"), 
        pl.when(pl.col.lead_paragraph == "NA").then(pl.lit("")).otherwise(pl.col.lead_paragraph).alias("lead_paragraph"))
    # joinging the headlines and leads, including the RoBERTa separator token (which I'm fairly certain does not
    # get spaces on either side; at least not added ones)
    # note that, if we create "headline_with_lead" ourselves, we don't need to built in MaskedLM preprocessor, 
    # since we'll have a single series of strings, each of which needs to be packed, padded, and masked separately.
    headline_lead = [x + "</s>" + y for x,y in zip(ldc_data.get_column("headline"), ldc_data.get_column("lead_paragraph"))]
    ldc_data = ldc_data.with_columns(pl.Series(name = "headline_with_lead", values = headline_lead))
    print(ldc_data.shape) # LOG
    return(ldc_data)

def dataset_create(shuffle_buffer, batch_size, preprocessor, path = None, data = None, parallelism = "default"):
    if path is not None: 
        dataset = tf.data.Dataset.load(path)
    elif data is not None: 
        print("Note: creating a dataset from tensor slices can take several minutes.")
        dataset = tf.data.Dataset.from_tensor_slices(data) 
    else: 
        raise ValueError("One of path or data must be provided.") # not sure this is actually a value error; maybe a TypeError?
    dataset = dataset.shuffle(buffer_size = shuffle_buffer).batch(batch_size = batch_size)
    if parallelism == "default": 
        dataset = dataset.map(preprocessor, num_parallel_calls = tf.data.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    else: 
        dataset = dataset.map(preprocessor, num_parallel_calls = parallelism)
        dataset = dataset.prefetch(parallelism)
    return(dataset)


