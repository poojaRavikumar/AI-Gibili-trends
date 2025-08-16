
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from bokeh.plotting import figure, output_file, save
from bokeh.layouts import column
from bokeh.models import ColumnDataSource
from bokeh.transform import dodge
from bs4 import BeautifulSoup

# Load data
df = pd.read_csv("ai_ghibli_trend_dataset_v2.csv")
df['creation_date'] = pd.to_datetime(df['creation_date'])
df['prompt_length'] = df['prompt'].apply(lambda x: len(str(x).split()))

def extract_category(prompt):
    prompt = prompt.lower()
    if "forest" in prompt or "nature" in prompt or "ocean" in prompt:
        return "Nature"
    elif "city" in prompt or "village" in prompt or "urban" in prompt or "market" in prompt:
        return "Urban"
    elif "magic" in prompt or "enchanted" in prompt or "fantasy" in prompt or "floating" in prompt:
        return "Fantasy"
    else:
        return "Other"
df['category'] = df['prompt'].apply(extract_category)

plots = []

# Q1
q1 = df.groupby('category')[['likes', 'shares']].mean().reset_index()
src = ColumnDataSource(q1)
p1 = figure(x_range=q1['category'], title="Q1: Likes & Shares by Category", height=300)
p1.vbar(x=dodge("category", -0.15, range=p1.x_range), top="likes", width=0.3, source=src, color="blue", legend_label="Likes")
p1.vbar(x=dodge("category", 0.15, range=p1.x_range), top="shares", width=0.3, source=src, color="green", legend_label="Shares")
plots.append(p1)

# Q2
q2 = df.groupby('platform')[['likes', 'shares']].mean().reset_index()
src = ColumnDataSource(q2)
p2 = figure(y_range=q2['platform'], title="Q2: Likes & Shares by Platform", height=300)
p2.hbar(y=dodge("platform", -0.15, range=p2.y_range), right="likes", height=0.3, source=src, color="orange", legend_label="Likes")
p2.hbar(y=dodge("platform", 0.15, range=p2.y_range), right="shares", height=0.3, source=src, color="teal", legend_label="Shares")
plots.append(p2)

# Q3
df['month'] = df['creation_date'].dt.to_period('M').astype(str)
q3 = df.groupby('month')[['likes', 'shares']].mean().reset_index()
src = ColumnDataSource(q3)
p3 = figure(x_range=q3['month'], title="Q3: Monthly Engagement Trends", height=300)
p3.vbar_stack(['likes', 'shares'], x='month', width=0.9, color=["purple", "olive"], legend_label=["Likes", "Shares"], source=src)
plots.append(p3)

# Q4
top200 = df.sort_values('likes', ascending=False).head(200).copy()
top200['prompt_length_jitter'] = top200['prompt_length'] + np.random.uniform(-0.3, 0.3, size=200)
src = ColumnDataSource(top200)
p4 = figure(title="Q4: Prompt Length vs Likes", height=300)
p4.circle(x="prompt_length_jitter", y="likes", size=7, source=src, color="firebrick", alpha=0.6)
plots.append(p4)

# Q5
df['hour'] = df['creation_date'].dt.hour
df['time_of_day'] = df['hour'].apply(lambda h: "Morning" if 5 <= h < 12 else "Afternoon" if 12 <= h < 17 else "Evening" if 17 <= h < 21 else "Night")
q5 = df.groupby('time_of_day')['likes'].mean().reset_index()
src = ColumnDataSource(q5)
p5 = figure(x_range=q5['time_of_day'], title="Q5: Likes by Time of Day", height=300)
p5.vbar(x="time_of_day", top="likes", width=0.5, source=src, color="teal")
plots.append(p5)

# Q6 Word Cloud
from PIL import Image
top_prompts = df.sort_values('likes', ascending=False).head(100)['prompt']
text = " ".join(top_prompts.astype(str))
wc = WordCloud(width=800, height=300, background_color="white").generate(text)
plt.figure(figsize=(10, 4))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.tight_layout()
plt.savefig("q6_wordcloud.png")
plt.close()

# Q7
q7 = df.groupby('resolution')[['likes', 'shares']].mean().reset_index()
src = ColumnDataSource(q7)
p7 = figure(x_range=q7['resolution'], title="Q7: Engagement by Resolution", height=300)
p7.vbar(x=dodge("resolution", -0.15, range=p7.x_range), top="likes", width=0.3, source=src, color="navy", legend_label="Likes")
p7.vbar(x=dodge("resolution", 0.15, range=p7.x_range), top="shares", width=0.3, source=src, color="orange", legend_label="Shares")
plots.append(p7)

# Q8
df['week'] = df['creation_date'].dt.to_period('W').astype(str)
weekly = df.groupby(['user_id', 'week']).size().reset_index(name='posts_per_week')
avg_post_freq = weekly.groupby('user_id')['posts_per_week'].mean().reset_index(name='avg_posts_per_week')
df = df.merge(avg_post_freq, on='user_id', how='left')
bin_edges = [0, 1, 2, 3, 5, 10, 20, 30]
bin_labels = ['<1', '1-2', '2-3', '3-5', '5-10', '10-20', '20+']
df['posting_bin'] = pd.cut(df['avg_posts_per_week'], bins=bin_edges, labels=bin_labels)
q8 = df.groupby('posting_bin')[['likes', 'shares']].mean().reset_index()
src = ColumnDataSource(q8)
p8 = figure(x_range=bin_labels, title="Q8: Engagement by Posting Frequency", height=300)
p8.vbar(x=dodge("posting_bin", -0.15, range=p8.x_range), top="likes", width=0.3, source=src, color="red", legend_label="Likes")
p8.vbar(x=dodge("posting_bin", 0.15, range=p8.x_range), top="shares", width=0.3, source=src, color="green", legend_label="Shares")
plots.append(p8)

# Output all Bokeh plots
output_file("ai_ghibli_dashboard_corrected.html")
save(column(*plots))
