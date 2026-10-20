import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("wylsacom.csv", parse_dates=["date"])

df = df.sort_values("date").reset_index(drop=True)

df = df[df["is_pinned"] == 0].copy()

df["text"] = df["text"].fillna("")
df["attachment_types"] = df["attachment_types"].fillna("")
df["attachment_text"] = df["attachment_text"].fillna("")

df["full_text"] = (df["text"] + " " + df["attachment_text"])

df["hour"] = df["date"].dt.hour
df["weekday"] = df["date"].dt.day_name()
df["is_weekend"] = df["date"].dt.weekday >= 5
df["time_of_day"] = pd.cut(
    df["hour"],
    bins=[-1, 5, 11, 17, 23],
    labels=["night", "morning", "afternoon", "evening"]
)

df["text_length"] = df["full_text"].str.len()
df["word_count"] = df["full_text"].str.split().str.len()
df["has_question"] = df["full_text"].str.contains("?", regex=False, na=False)
df["hashtag_count"] = df["full_text"].str.count("#")

df["has_photo"] = df["attachment_types"].str.contains("photo", na=False)
df["has_video"] = df["attachment_types"].str.contains("video", na=False)
df["has_link"] = df["attachment_types"].str.contains("link", na=False)

df["hours_since_prev_post"] = df["date"].diff().dt.total_seconds() / 3600

df["likes_per_1000_views"] = (df["likes"] / df["views"] * 1000)
df["comments_per_1000_views"] = (df["comments"] / df["views"] * 1000)
df["reposts_per_1000_views"] = (df["reposts"] / df["views"] * 1000)
df["engagement_per_1000_views"] = (df["likes"] + df["comments"] + df["reposts"]) / df["views"] * 1000

keywords = {
    "apple": [
        "apple",
        "iphone",
        "ipad",
        "macbook",
        "ios",
        "airpods"
    ],

    "samsung": [
        "samsung",
        "galaxy"
    ],

    "google": [
        "google",
        "pixel",
        "android"
    ],

    "xiaomi": [
        "xiaomi",
        "redmi",
        "poco"
    ],

    "gaming": [
        "playstation",
        "ps5",
        "xbox",
        "nintendo",
        "steam"
    ],

    "cars": [
        "tesla",
        "bmw",
        "mercedes",
        "автомоб",
        "машин"
    ],
    "giveaway": [
        "розыгрыш",
        "разыг",
        "конкурс",
        "подар",
        "выигр",
        "победит",
        "участв"
    ]
}

for group, words in keywords.items():
    df[group] = df["full_text"].str.contains(
        "|".join(words), regex=True, na=False, case=False
    )

metrics = [
    "views",
    "likes",
    "comments",
    "reposts",
    "likes_per_1000_views",
    "comments_per_1000_views",
    "reposts_per_1000_views",
    "engagement_per_1000_views"
]

print("\nОБЩАЯ СТАТИСТИКА")
print(df[metrics].describe().round(2))

threshold = df["engagement_per_1000_views"].quantile(0.90)

df_90 = df[df["engagement_per_1000_views"] <= threshold].copy()

df_10 = df[df["engagement_per_1000_views"] > threshold].copy()

print("\nГраница верхних 10%:", round(threshold, 2))
print("Обычные 90%:", len(df_90))
print("Верхние 10%:", len(df_10))

reaction_metrics = [
    "likes_per_1000_views",
    "comments_per_1000_views",
    "reposts_per_1000_views",
    "engagement_per_1000_views"
]

weekday_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

time_order = [
    "night",
    "morning",
    "afternoon",
    "evening"
]

topic_columns = [
    "apple",
    "samsung",
    "google",
    "xiaomi",
    "gaming",
    "cars",
    "giveaway"
]

def print_stats(data, column, title):
    print(f"\n{title}")
    result = data.groupby(column, observed=True)[reaction_metrics].median().round(2)
    print(result)

def print_plot(data, x, title, order=None, rotation=0):
    plt.figure(figsize=(10, 5))

    sns.barplot(
        data=data,
        x=x,
        y="engagement_per_1000_views",
        estimator="median",
        errorbar=None,
        order=order
    )

    plt.title(title)
    plt.xlabel(x)
    plt.ylabel("Медианная вовлечённость на 1000 просмотров")

    if rotation:
        plt.xticks(rotation=rotation)

    plt.tight_layout()
    plt.show()

print("АНАЛИЗ ОБЫЧНЫХ 90% ПОСТОВ")


print_stats(df_90, "time_of_day", "Время суток")
print_plot(
    df_90,
    "time_of_day",
    "Вовлечённость по времени суток, обычные 90%",
    order=time_order
)

weekday_stats_90 = (
    df_90.groupby("weekday")[reaction_metrics]
    .median()
    .reindex(weekday_order)
    .round(2)
)

print("\nДни недели")
print(weekday_stats_90)

print_plot(
    df_90,
    "weekday",
    "Вовлечённость по дням недели, обычные 90%",
    order=weekday_order,
    rotation=45
)

print_stats(df_90, "is_weekend", "Будни и выходные")
print_plot(
    df_90,
    "is_weekend",
    "Будни и выходные, обычные 90%"
)

for column in ["has_photo", "has_video", "has_link"]:
    print_stats(df_90, column, f"Формат: {column}")
    print_plot(
        df_90,
        column,
        f"{column}: обычные 90%"
    )

print_stats(df_90, "has_question", "Наличие вопроса в тексте")
print_plot(
    df_90,
    "has_question",
    "Вопрос в тексте, обычные 90%"
)

for topic in topic_columns:
    print_stats(df_90, topic, f"Тема: {topic}")

numeric_features = [
    "hour",
    "text_length",
    "word_count",
    "hashtag_count",
    "attachments_count",
    "hours_since_prev_post",
    "engagement_per_1000_views"
]

correlation_90 = df_90[numeric_features].corr(method="spearman").round(2)


print("\nКорреляции, обычные 90%")
print(correlation_90["engagement_per_1000_views"].sort_values(ascending=False))

plt.figure(figsize=(10, 6))

sns.heatmap(correlation_90,annot=True,fmt=".2f")

plt.title("Корреляция Спирмена, обычные 90%")
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df_90,
    x="text_length",
    y="engagement_per_1000_views",
    alpha=0.5
)

plt.title("Длина текста и вовлечённость, обычные 90%")
plt.xlabel("Длина текста")
plt.ylabel("Вовлечённость на 1000 просмотров")

plt.tight_layout()
plt.show()



plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df_90,
    x="hours_since_prev_post",
    y="engagement_per_1000_views",
    alpha=0.5
)

plt.title("Интервал между постами и вовлечённость, обычные 90%")
plt.xlabel("Часов с предыдущего поста")
plt.ylabel("Вовлечённость на 1000 просмотров")

plt.tight_layout()
plt.show()



print("АНАЛИЗ ВЕРХНИХ 10% ПОСТОВ")

print("\nСтатистика верхних 10%")
print(df_10[metrics].describe().round(2))

print_stats(df_10, "time_of_day", "Время суток")
print_plot(
    df_10,
    "time_of_day",
    "Вовлечённость по времени суток, верхние 10%",
    order=time_order
)

weekday_stats_10 = (
    df_10.groupby("weekday")[reaction_metrics]
    .median()
    .reindex(weekday_order)
    .round(2)
)

print("\nДни недели")
print(weekday_stats_10)

print_plot(
    df_10,
    "weekday",
    "Вовлечённость по дням недели, верхние 10%",
    order=weekday_order,
    rotation=45
)

has_columns = [
    "has_photo",
    "has_video",
    "has_link",
    "has_question"
]

print("\nДоли форматов внутри верхних 10%")

for column in has_columns:
    print(column, round(df_10[column].mean() * 100, 1), "%")

print("\nДоли тем внутри верхних 10%")

for topic in topic_columns:
    print(topic, round(df_10[topic].mean() * 100, 1), "%")

top_20 = (
    df_10[
        [
            "date",
            "views",
            "likes",
            "comments",
            "reposts",
            "engagement_per_1000_views",
            "attachment_types",
            "giveaway",
            "full_text",
            "url"
        ]
    ]
    .sort_values("engagement_per_1000_views",ascending=False).head(20)
)

print("\nТОП-20 ПОСТОВ")
print(top_20.to_string(index=False))



comparison = pd.DataFrame({
    "regular_90": {
        "median_engagement": df_90["engagement_per_1000_views"].median(),
        "median_likes": df_90["likes_per_1000_views"].median(),
        "median_comments": df_90["comments_per_1000_views"].median(),
        "median_reposts": df_90["reposts_per_1000_views"].median(),
        "video_mean": df_90["has_video"].mean(),
        "photo_mean": df_90["has_photo"].mean(),
        "giveaway_mean": df_90["giveaway"].mean()
    },

    "top_10": {
        "median_engagement": df_10["engagement_per_1000_views"].median(),
        "median_likes": df_10["likes_per_1000_views"].median(),
        "median_comments": df_10["comments_per_1000_views"].median(),
        "median_reposts": df_10["reposts_per_1000_views"].median(),
        "video_mean": df_10["has_video"].mean(),
        "photo_mean": df_10["has_photo"].mean(),
        "giveaway_mean": df_10["giveaway"].mean()
    }
})

print("\nСРАВНЕНИЕ 90% И 10%")
print(comparison.round(3))

comparison_plot = pd.DataFrame({
    "group": ["Обычные 90%", "Верхние 10%"],
    "engagement": [
        df_90["engagement_per_1000_views"].median(),
        df_10["engagement_per_1000_views"].median()
    ]
})

plt.figure(figsize=(10, 5))

sns.barplot(
    data=comparison_plot,
    x="group",
    y="engagement",
    errorbar=None
)

plt.title("Медианная вовлечённость: 90% против верхних 10%")
plt.xlabel("")
plt.ylabel("Engagement на 1000 просмотров")

plt.tight_layout()
plt.show()


comparison_features = pd.DataFrame({
    "feature": [
        "Видео",
        "Фото",
        "Ссылки",
        "Вопрос",
        "Розыгрыш",
        "Apple",
        "Samsung",
        "Gaming",
        "Cars"
    ],

    "regular_90": [
        df_90["has_video"].mean() * 100,
        df_90["has_photo"].mean() * 100,
        df_90["has_link"].mean() * 100,
        df_90["has_question"].mean() * 100,
        df_90["giveaway"].mean() * 100,
        df_90["apple"].mean() * 100,
        df_90["samsung"].mean() * 100,
        df_90["gaming"].mean() * 100,
        df_90["cars"].mean() * 100
    ],

    "top_10": [
        df_10["has_video"].mean() * 100,
        df_10["has_photo"].mean() * 100,
        df_10["has_link"].mean() * 100,
        df_10["has_question"].mean() * 100,
        df_10["giveaway"].mean() * 100,
        df_10["apple"].mean() * 100,
        df_10["samsung"].mean() * 100,
        df_10["gaming"].mean() * 100,
        df_10["cars"].mean() * 100
    ]
})

comparison_long = comparison_features.melt(
    id_vars="feature",
    var_name="group",
    value_name="share"
)

sns.barplot(
    data=comparison_long,
    x="feature",
    y="share",
    hue="group"
)

plt.title("Чем отличаются верхние 10% постов")
plt.xlabel("")
plt.ylabel("Доля постов, %")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

df_excel = df.copy()
df_90_excel = df_90.copy()
df_10_excel = df_10.copy()

for data in [df_excel, df_90_excel, df_10_excel]:
    data["date"] = data["date"].dt.tz_localize(None)

with pd.ExcelWriter("wylsacom.xlsx", engine="xlsxwriter") as writer:
    df_excel.to_excel(writer,sheet_name="all_posts",index=False)

    df_90_excel.to_excel(writer,sheet_name="regular_90",index=False)

    df_10_excel.to_excel(writer,sheet_name="top_10",index=False)