"""
Twitter APIについてはhttps://developer.twitter.com/en/docs/api-reference-index を見ればわかる
Tweepyについては
https://docs.tweepy.org/en/v3.5.0/api.html
https://akitoshiblogsite.com/tweepy-basic-functions/

を参照
"""


import json

import tweepy
from wtfml.auth.config import TWITTER_API_KEY

CONSUMER_KEY = TWITTER_API_KEY["CONSUMER_KEY"]
CONSUMER_SECRET = TWITTER_API_KEY["CONSUMER_SECRET"]  # ? 名前が違うのは大丈夫なのかな？
ACCESS_TOKEN = TWITTER_API_KEY["ACCESS_TOKEN"]
ACCESS_SECRET = TWITTER_API_KEY["ACCESS_SECRET"]

auth = tweepy.AppAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
if ACCESS_TOKEN and ACCESS_TOKEN:
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN)
api = tweepy.API(
    auth,
    wait_on_rate_limit=True,
    wait_on_rate_limit_notify=True,
)


# userの取得
user = api.get_user(screen_name="NewsDigestWeb")

users = api.search_users(
    "JX通信社",
    per_page=2,
    page=1,
)

# 特定のユーザーのツイートを取得
for status in api.user_timeline(id=""):
    ...

search_results = api.search("JX通信社")


# jsearch_results_json = json.dumps(public_tweets[0]._json, indent=4, ensure_ascii=False)
