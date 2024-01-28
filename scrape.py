import requests
import argparse
import time
import polars as pl


def call_api(param):
    url = "https://api.pullpush.io/reddit/search/submission/"
    response = requests.get(url, params=param)
    json_data = response.json()["data"]
    create_utc = []
    media_id = []
    media_type_ls = []
    post_ids = []
    post_titles = []
    cur_utc = 0
    for submission in json_data:
        cur_flair = submission["link_flair_text"]
        cur_utc = submission["created_utc"]
        media_ls = submission["media_metadata"] if "media_metadata" in submission.keys() else None
        if param["flair"] is not None and cur_flair != param["flair"]:
            continue
        if media_ls is None:
            continue
        for id in media_ls.keys():
            if media_ls[id]["status"] != "valid":
                continue
            try:
                media_type = media_ls[id]["m"]
            except:
                # video will error out
                continue
            if media_type == "image/png":
                media_type_ls.append("png")
            elif media_type == "image/jpg":
                media_type_ls.append("jpg")
            else:
                continue
            create_utc.append(int(cur_utc))
            post_ids.append(submission["id"])
            post_titles.append(submission["title"])
            media_id.append(id)
    
    df = pl.DataFrame(
        {
            "create_utc": create_utc,
            "media_id": media_id,
            "media_type": media_type_ls,
            "post_id": post_ids,
            "post_title": post_titles
        },
        schema={
            "create_utc": pl.Int64,
            "media_id": pl.Utf8,
            "media_type": pl.Utf8,
            "post_id": pl.Utf8,
            "post_title": pl.Utf8
        }
    )
    return df, int(cur_utc)


def scraping_loop(subreddit, flair, max_num=30000, output_name=None, before=None):
    collected_all = []
    collected_len = 0
    last_timestamp = int(time.time()) if before is None else before
    param = {
        "subreddit": subreddit,
        "flair": flair,
        "before": last_timestamp
    }
    while collected_len < max_num:
        collected_df, last_timestamp = call_api(param)
        if collected_df.shape[0] == 0:
            print("No more data, saving current data and exiting...")
            break
        collected_all.append(collected_df)
        collected_len += collected_df.shape[0]
        print(f"collected_len: {collected_len}, last_timestamp: {last_timestamp}")
        param["before"] = last_timestamp
    
    df = pl.concat(collected_all)
    df = df.with_columns(
        pl.col("media_id").str.replace(r"^", "https://i.redd.it/").alias("url1"),
        pl.col("create_utc").cast(pl.Int64).cast(pl.Utf8).str.to_datetime("%s").alias("time")
    ).with_columns(
        pl.col("media_type").str.replace(r"^", ".").alias("url2")
    ).with_columns(
        pl.concat_str(
            [pl.col("url1"),pl.col("url2")],
            separator=""
        ).alias("url")
    ).select("time", "url", "post_id", "post_title")
    if output_name is None:
        output_name = subreddit
    df.write_parquet(f"urls/{output_name}.parquet")
    df.select("url").write_csv(f"urls/{output_name}.csv", has_header=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subreddit", help="subreddit name")
    parser.add_argument("--flair", help="flair filter", default=None, type=str)
    parser.add_argument("--max_num", help="max number of posts to scrape", default=30000, type=int)
    parser.add_argument("--output_name", help="custom output name", default=None)
    parser.add_argument("--before", help="before timestamp", default=None, type=int)

    args = parser.parse_args()

    scraping_loop(**args.__dict__)