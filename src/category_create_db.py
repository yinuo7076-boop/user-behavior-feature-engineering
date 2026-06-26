import sqlite3
import pandas as pd
import random
from datetime import datetime, timedelta
sqlite3.connect('middle_category.db')
conn = sqlite3.connect('middle_category.db')
cursor = conn.cursor()
# Create the feature table
cursor.execute('''
CREATE TABLE IF NOT EXISTS category_feature(
    item_category TEXT PRIMARY KEY,
    total_pv REAL,
    cart_count REAL,
    favorite_count REAL,
    buy_count REAL,
    users_count REAL,
    buy_rate REAL,
    peak_view_hour TEXT,
    peak_buy_hour TEXT
)
               ''')
# Load data from CSV and insert into the database
df = pd.read_csv("data/data_min.csv")
df['time'] = pd.to_datetime(df['time'],format='%Y-%m-%d %H')
df['hour'] = df['time'].dt.hour
df['behavior_type'] = df['behavior_type'].astype(int)
cutoff_date = pd.Timestamp('2014-12-18')
df_feature = (df[df['time'] < cutoff_date].copy())
df.to_sql('user_behavior', conn, if_exists='replace', index=False)
print(df.head())
print("Data inserted into the database successfully.")
# Create indexes to speed up queries
cursor.execute('''CREATE INDEX IF NOT EXISTS idx_category_id ON user_behavior(item_category)''')
cursor.execute('''CREATE INDEX IF NOT EXISTS idx_behavior_type ON user_behavior(item_category, behavior_type)''')
cursor.execute('''CREATE INDEX IF NOT EXISTS idx_category_type_time ON user_behavior(item_category, behavior_type, time)''')
cursor.execute('''CREATE INDEX IF NOT EXISTS idx_category_user ON user_behavior(item_category, user_id)''')
conn.commit()
print("Index created successfully.")

# 统一计算 max_time
max_time = df_feature['time'].max()

# Create a new table with item features
cursor.execute('''SELECT * FROM user_behavior LIMIT 10''')

# 构建完整特征查询
# 用 Pandas 高效计算所有特征
print("Starting feature engineering with Pandas...")
# 初始化特征字典
features = {}
# basic counts
for bt, col in [(1, 'total_pv'), (3, 'cart_count'), (2, 'favorite_count'), (4, 'buy_count')]:
    features[col] = df_feature[df_feature['behavior_type'] == bt].groupby('item_category').size()
pv_all = features['total_pv']
# users_count
features['users_count'] = df_feature.groupby('item_category')['user_id'].nunique()
# buy_rate
features['buy_rate'] = features['buy_count'] / pv_all.replace(0, pd.NA)
# peak hours
def get_peak_hour(bt):
    hourly = df_feature[df_feature['behavior_type'] == bt].copy()
    peak = hourly.groupby(['item_category', 'hour']).size().reset_index(name='cnt')
    if peak.empty: return pd.Series(dtype='object')
    return peak.loc[peak.groupby('item_category')['cnt'].idxmax()].set_index('item_category')['hour']
features['peak_view_hour'] = get_peak_hour(1)
features['peak_buy_hour'] = get_peak_hour(4)
# 合并所有特征
result_df = pd.DataFrame(features).fillna(0)
result_df.index.name = 'item_category'
result_df = result_df.reset_index()
# 保存到数据库
cursor.execute("DELETE FROM category_feature")
result_df.to_sql('category_feature', conn, if_exists='append', index=False)
conn.commit()
print(f"Successfully computed features for {len(result_df)} categorys!")

# Total views (pv) for each category
cursor.execute('''
SELECT
    item_category,
    COUNT(*) AS total_pv
FROM user_behavior
WHERE behavior_type = '1'
GROUP BY item_category;
    ''')
# Check the result for total_pv
# Step 1: 随机抽取3个category_item及其total_pv
cursor.execute("SELECT item_category, total_pv FROM category_feature ORDER BY RANDOM() LIMIT 3;")
sample_items = cursor.fetchall()
for item_category, total_pv in sample_items:
    print(f"Item ID: {item_category}, Total PV (from category_feature): {total_pv}")
    # Step 2: 手动统计 user_behavior 中 behavior_type='1' 的数量
    cursor.execute("""
        SELECT COUNT(*) 
        FROM user_behavior 
        WHERE item_category = ? AND behavior_type = '1'
        AND time < '2014-12-18'
    """, (item_category,))
    actual_pv = cursor.fetchone()[0]
    # Step 3: 对比验证
    if actual_pv == total_pv:
        print(f"Passed: actual PV = {actual_pv}")
    else:
        print(f"Failed: actual PV = {actual_pv}, recorded value = {total_pv}")

# Total cart
cursor.execute('''
SELECT
    item_category,
    COUNT(*) AS total_cart
FROM user_behavior
WHERE behavior_type = '3'
GROUP BY item_category;
                ''')
# Check the result for total cart
print("\nCheck for cart_count（random three categories）:")
cursor.execute("SELECT item_category, cart_count FROM category_feature WHERE cart_count IS NOT NULL ORDER BY RANDOM() LIMIT 3;")
for item_category, recorded in cursor.fetchall():
    cursor.execute("SELECT COUNT(*) FROM user_behavior WHERE item_category = ? AND behavior_type = '3' AND time < '2014-12-18'", (item_category,))
    actual = cursor.fetchone()[0]
    print(f"Category: {item_category} | Recorded: {recorded} | Actual: {actual} | {'passed' if recorded == actual else 'failed'}")

# Total fav
cursor.execute('''
SELECT
    item_category,
    COUNT(*) AS total_fav
FROM user_behavior
WHERE behavior_type = '2'
GROUP BY item_category;
                ''')
# Check the result for total fav
print("\nCheck for favorite_count（random three categories）:")
cursor.execute("SELECT item_category, favorite_count FROM category_feature WHERE favorite_count IS NOT NULL ORDER BY RANDOM() LIMIT 3;")
for item_category, recorded in cursor.fetchall():
    cursor.execute("SELECT COUNT(*) FROM user_behavior WHERE item_category = ? AND behavior_type = '2' AND time < '2014-12-18'", (item_category,))
    actual = cursor.fetchone()[0]
    print(f"Category: {item_category} | Recorded: {recorded} | Actual: {actual} | {'passed' if recorded == actual else 'failed'}")

# Total buy
cursor.execute('''
SELECT
    item_category,
    COUNT(*) AS total_buy
FROM user_behavior
WHERE behavior_type = '4'
GROUP BY item_category;
                ''')
# Check the result for total buy
print("\nCheck for buy_count（random three categories）:")
cursor.execute("SELECT item_category, buy_count FROM category_feature WHERE buy_count IS NOT NULL ORDER BY RANDOM() LIMIT 3;")
for item_category, recorded in cursor.fetchall():
    cursor.execute("SELECT COUNT(*) FROM user_behavior WHERE item_category = ? AND behavior_type = '4' AND time < '2014-12-18'", (item_category,))
    actual = cursor.fetchone()[0]
    print(f"Category: {item_category} | Recorded: {recorded} | Actual: {actual} | {'passed' if recorded == actual else 'failed'}")

# User counts
cursor.execute('''
SELECT
    item_category,
    COUNT(DISTINCT user_id) AS category_user_cnt
FROM user_behavior
GROUP BY item_category;
               ''')
# Check the result for user counts
print("\nCheck for users_count（random 3 categories）:")
cursor.execute("""
SELECT item_category, users_count
FROM category_feature
ORDER BY RANDOM()
LIMIT 3
""")
for item_category, recorded in cursor.fetchall():
    cursor.execute("""
        SELECT COUNT(DISTINCT user_id)
        FROM user_behavior
        WHERE item_category = ?
        AND time < '2014-12-18'
    """, (item_category,))
    actual = cursor.fetchone()[0]
    status = (
        "Passed"
        if recorded == actual
        else "Failed"
    )
    print(
        f"Category: {item_category} | "
        f"Recorded: {recorded} | "
        f"Actual: {actual} | "
        f"{status}"
    )

# Buy rate
cursor.execute('''
SELECT
    item_category,
    SUM(CASE WHEN behavior_type = '4' THEN 1 ELSE 0 END) * 1.0 /
    SUM(CASE WHEN behavior_type = '1' THEN 1 ELSE 0 END) AS buy_rate
FROM user_behavior
GROUP BY item_category;
                ''')
# Check the result for buy rate
print("\nCheck for buy_rate（random 3 categories）:")
cursor.execute("SELECT item_category, buy_rate FROM category_feature WHERE buy_rate IS NOT NULL ORDER BY RANDOM() LIMIT 3;")
for item_category, recorded in cursor.fetchall():
    cursor.execute("""
        SELECT 
            CAST(SUM(CASE WHEN behavior_type='4' THEN 1 ELSE 0 END) AS REAL) * 1.0 /
            NULLIF(SUM(CASE WHEN behavior_type='1' THEN 1 ELSE 0 END), 0)
        FROM user_behavior WHERE item_category = ?
        AND time < '2014-12-18'
    """, (item_category,))
    actual = cursor.fetchone()[0] or 0.0
    match = abs((recorded or 0.0) - actual) < 1e-6
    print(f"Category: {item_category} | Recorded: {recorded:.4f} | Actual: {actual:.4f} | {'passed' if match else 'failed'}")

# Peak time for item buys
cursor.execute('''
WITH t AS (
    SELECT
        item_category,
        strftime('%H', time) AS hour,
        COUNT(*) AS cnt
    FROM user_behavior
    WHERE behavior_type='4'
    GROUP BY item_category, hour
),
ranked AS (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY item_category
               ORDER BY cnt DESC
           ) AS rn
    FROM t
)
SELECT
    item_category,
    hour AS peak_hour,
    cnt AS peak_cnt
FROM ranked
WHERE rn = 1;
                ''')
# Check peak time for item buys
print("\nCheck for peak_hour for category views（random 3 categories）:")
cursor.execute("""
SELECT item_category, peak_view_hour
FROM category_feature
WHERE peak_view_hour IS NOT NULL
ORDER BY RANDOM()
LIMIT 3
""")
for item_category, recorded_hour in cursor.fetchall():
    cursor.execute("""
        SELECT hour
        FROM user_behavior
        WHERE item_category = ?
            AND behavior_type = 1
            AND time < '2014-12-18'
        GROUP BY hour
        ORDER BY COUNT(*) DESC, hour ASC
        LIMIT 1
    """, (item_category,))
    result = cursor.fetchone()
    if result:
        actual_hour = result[0]
        status = ("Passed" if float(recorded_hour) == float(actual_hour) else "Failed")
        print(
            f"Category: {item_category} | "
            f"Recorded: {recorded_hour} | "
            f"Actual: {actual_hour} | "
            f"{status}"
        )

# Peak time for item buys
cursor.execute('''
WITH t AS (
    SELECT
        item_category,
        strftime('%H', time) AS hour,
        COUNT(*) AS cnt
    FROM user_behavior
    WHERE behavior_type='4'
    GROUP BY item_category, hour
),
ranked AS (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY item_category
               ORDER BY cnt DESC
           ) AS rn
    FROM t
)
SELECT
    item_category,
    hour AS peak_hour,
    cnt AS peak_cnt
FROM ranked
WHERE rn = 1;
                ''')
# Check peak time for item buys
print("\nCheck for peak_hour for category buy（random 3 categories）:")
cursor.execute("""
SELECT item_category, peak_buy_hour
FROM category_feature
WHERE peak_buy_hour IS NOT NULL
ORDER BY RANDOM()
LIMIT 3
""")
for item_category, recorded_hour in cursor.fetchall():
    cursor.execute("""
        SELECT hour
        FROM user_behavior
        WHERE item_category = ?
            AND time < '2014-12-18'
            AND behavior_type = 4
        GROUP BY hour
        ORDER BY COUNT(*) DESC, hour ASC
        LIMIT 1
    """, (item_category,))
    result = cursor.fetchone()
    if result:
        actual_hour = result[0]
        status = ("Passed" if float(recorded_hour) == float(actual_hour) else "Failed")
        print(
            f"Category: {item_category} | "
            f"Recorded: {recorded_hour} | "
            f"Actual: {actual_hour} | "
            f"{status}"
        )
    else:
        print(
            f"Category {item_category} "
            f"has no buy records before 2014-12-18"
    )

conn.commit()
conn.close()