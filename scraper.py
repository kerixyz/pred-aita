import praw
import pandas as pd


posts = []
# change subreddit to specific page
sr = reddit.subreddit('AmITheAsshole')

#limit = how many pages
for post in sr.new(limit=1000):
    posts.append([post.created,post.url,post.id,post.score,post.num_comments,post.author,post.link_flair_text,post.title,post.selftext])

posts = pd.DataFrame(posts, columns=['created','url','id','score','num_comments','author','flair','title','body'])

posts['flair'] = posts['id'].apply(lambda x: reddit.submission(id = x).link_flair_text)
posts['created'] = pd.to_datetime(posts['created'], unit='s')
#can use this one to filter flairs if the pages do have flairs
#questions = posts[posts.flair=='Question']

posts.to_csv(r'aita.csv', index=False, encoding='utf-8-sig')
#questions.to_csv(r'questions.csv', index=False, encoding='utf-8-sig')