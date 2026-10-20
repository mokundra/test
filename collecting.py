import requests
import pandas as pd
import json

count = 100
limit = 5000

all_posts = []

for offset in range(0, limit, count):
    params = {
        "access_token": "fea61851fea61851fea618513bfde2a1ebffea6fea61851947091288a1bb666da419206",
        "v": "5.199",
        "domain": "wylsacom",
        "count": count,
        "offset": offset,
        "filter": "owner"
    }

    data = requests.get("https://api.vk.com/method/wall.get", params=params).json()

    posts = data["response"]["items"]

    all_posts.extend(posts)

    print(f"Собрано постов: {len(all_posts)}")

with open("wylsacom.json", "w", encoding="utf-8") as file:
    json.dump(all_posts, file, ensure_ascii=False, indent=2)

rows = []

for post in all_posts:
    attachments = post.get("attachments", [])
    attachment_types = [attachment.get("type") for attachment in attachments]
    attachment_texts = []

    for attachment in attachments:
        attachment_type = attachment.get("type")

        if attachment_type == "video":
            video = attachment.get("video", {})
            attachment_texts.append(video.get("title", ""))
            attachment_texts.append(video.get("description", ""))

        elif attachment_type == "photo":
            photo = attachment.get("photo", {})
            attachment_texts.append(photo.get("text", ""))

    rows.append({
        "post_id": post["id"],

        "date": pd.to_datetime(post["date"], unit="s", utc=True).tz_convert("Europe/Moscow"),

        "text": post.get("text", ""),

        "likes": post.get("likes", {}).get("count", 0),

        "comments": post.get("comments", {}).get("count", 0),

        "reposts": post.get("reposts", {}).get("count", 0),

        "views": post.get("views", {}).get("count"),

        "attachment_types": ",".join(attachment_types),

        "attachments_count": len(attachments),

        "attachment_text": " ".join(text for text in attachment_texts if text),

        "is_pinned": post.get("is_pinned",0),

        "url": (f"https://vk.com/wall"f"{post['owner_id']}_"f"{post['id']}")
    })

df = pd.DataFrame(rows)

df = df.drop_duplicates(subset="post_id")

df.to_csv("wylsacom.csv",index=False, encoding="utf-8-sig")

print()
print(df.head())
print()
print("Итоговое количество:", len(df))
print()
print(df.info())
