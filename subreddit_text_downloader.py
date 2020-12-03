import praw
import requests
import json
from datetime import datetime
import time
import argparse
from os.path import join
from os import getcwd
from tqdm import tqdm

class Downloader:
    def __init__(self, client_id = None, client_secret = None, user_agent = None):
        if client_id is not None:
            self.reddit = praw.Reddit(client_id=client_id,
                                client_secret=client_secret,
                                user_agent=user_agent)


    def scrape_posts(self, subreddit, start_time = None, end_time = None, interval = 3600*6):
        '''With default args, downloads all posts from a subreddit
        For optimal speed, interval should be big enough that there are more post
        than the pushshift_max (currently 100) during that period. 
        '''
        wait_time = 0.5
        retries = 10
        pushshift_max = 100
        if start_time is None:
            if not hasattr(self, "reddit"):
                raise Exception("Since no API keys were provided, start_time and end_time must be specified")
            start_time = int(self.reddit.subreddit(subreddit).created_utc)
        if end_time is None:
            end_time = int(datetime.now().timestamp())

        output_file = f'fetched_r_{subreddit}_{start_time}_{end_time}.json'
        with open(output_file, 'w') as f:
            print(f"Fetching data and saving it in {join(getcwd(),output_file)}...")
            f.write("[")
            tqdmbar = tqdm(total = end_time - start_time)
            while start_time < end_time:
                end_interval = min(start_time + interval, end_time)
                url = f"https://api.pushshift.io/reddit/search/submission/?after={start_time}&before={end_interval}&subreddit={subreddit}&limit=1000&score=%3E0"
                r = requests.get(url)
                for i in range(retries):
                    if r.status_code == 200:
                        break
                    else:
                        print("ERROR", str(r.status_code))
                        time.sleep(1*(2**i))
                        r = requests.get(url)
                posts = json.loads(r.text)["data"]
                if len(posts) < pushshift_max:
                    tqdmbar.update(interval)
                    start_time += interval
                else:
                    tqdmbar.update(int(posts[-1]["created_utc"]) - start_time)
                    start_time = int(posts[-1]["created_utc"])
                for post in posts:
                    json.dump(post, f)
                    f.write(",\n")
                time.sleep(wait_time)
            tqdmbar.close()
            # correcting end of file:
            f.seek(f.tell()-2)
            f.truncate()
            f.write("]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--subreddit', type=str, required = True)
    parser.add_argument('--client_id', type=str, default = None)
    parser.add_argument('--client_secret', type=str, default = None)
    parser.add_argument('--user_agent', type=str, default = None)
    parser.add_argument('--start_time', type=int, default = None)
    parser.add_argument('--end_time', type=int, default = None)
    parser.add_argument('--interval', type=int, default = 3600*6)

    args = parser.parse_args()

    downloader = Downloader(args.client_id, args.client_secret, args.user_agent)
    downloader.scrape_posts(args.subreddit, args.start_time, args.end_time, args.interval)
