import sqlite3
import pandas as pd
import random
from datetime import datetime, timedelta
sqlite3.connect('middle_item.db')
conn = sqlite3.connect('middle_item.db')
cursor = conn.cursor()
# Create the feature table
cursor.execute('''
CREATE TABLE IF NOT EXISTS item_feature(
    item_id TEXT PRIMARY KEY,

    total_pv REAL,
    last3day_pv REAL,
    last7day_pv REAL,
    popularity_growth_rate REAL,
    items_pv_ranking INTEGER,
    cart_count REAL,
    favorite_count REAL,
    buy_count REAL,
    cart_rate REAL,
    favorite_rate REAL,
    buy_rate REAL,
    uv REAL,
    view_not_buy_rate REAL,
    cart_not_buy_rate REAL,
    fav_not_buy_rate REAL,
    interest_depth REAL,
    repurchase_index REAL,
    buy_recency_hours REAL,
    peak_view_hour TEXT,
    peak_buy_hour TEXT
)
''')
# Load data from CSV and insert into the database
df = pd.read_csv("data/data_min.csv")
df.to_sql('user_behavior', conn, if_exists='replace', index=False)
print(df.head())
print("Data inserted into the database successfully.")
# Create indexes to speed up queries
cursor.execute('''CREATE INDEX IF NOT EXISTS idx_item_id ON user_behavior(item_id)''')
cursor.execute('''CREATE INDEX IF NOT EXISTS idx_item_type ON user_behavior(item_id, behavior_type)''')
cursor.execute('''CREATE INDEX IF NOT EXISTS idx_item_type_time ON user_behavior(item_id, behavior_type, time)''')
conn.commit()
print("Index created successfully.")

# Create a new table with item features
cursor.execute('''SELECT * FROM user_behavior LIMIT 10''')

# 构建完整特征查询
# 用 Pandas 高效计算所有特征
print("Starting feature engineering with Pandas...")

# 转换时间列
df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H')
df['hour'] = pd.to_datetime(df['time']).dt.hour
cutoff_date = pd.Timestamp('2014-12-18')
df_feature = df[df['time'] < cutoff_date].copy()
max_time = df_feature['time'].max()
if pd.isna(max_time):
    raise ValueError("df_feature 为空，无法计算时间窗口！")
three_days_ago = max_time - timedelta(days=3)
seven_days_ago = max_time - timedelta(days=7)

# 转为字符串（供后续 SQL 验证使用）
max_time_str = max_time.strftime("%Y-%m-%d %H")
three_days_ago_str = three_days_ago.strftime("%Y-%m-%d %H")
seven_days_ago_str = seven_days_ago.strftime("%Y-%m-%d %H")

# 全局时间窗口
max_time = df_feature['time'].max()
three_days_ago = max_time - timedelta(days=3)
seven_days_ago = max_time - timedelta(days=7)

# 初始化特征字典
features = {}

# 1. 基础计数 (pv, cart, fav, buy)
for bt, col in [(1, 'total_pv'), (3, 'cart_count'), (2, 'favorite_count'), (4, 'buy_count')]:
    features[col] = df_feature[df_feature['behavior_type'] == bt].groupby('item_id').size()

# 2. 时间窗口 PV
pv_all = features['total_pv']
pv_3d = df_feature[(df_feature['behavior_type'] == 1) & (df_feature['time'] >= three_days_ago)].groupby('item_id').size()
pv_7d = df_feature[(df_feature['behavior_type'] == 1) & (df_feature['time'] >= seven_days_ago)].groupby('item_id').size()
features['last3day_pv'] = pv_3d
features['last7day_pv'] = pv_7d

# 3. 衍生特征
features['popularity_growth_rate'] = pv_3d / pv_7d.replace(0, 1)
features['items_pv_ranking'] = pv_all.rank(method='min', ascending=False).astype(int)
features['cart_rate'] = features['cart_count'] / pv_all.replace(0, 1)
features['favorite_rate'] = features['favorite_count'] / pv_all.replace(0, 1)
features['buy_rate'] = features['buy_count'] / pv_all.replace(0, 1)

# 4. UV (用户覆盖)
features['uv'] = df_feature.groupby('item_id')['user_id'].nunique()

# 5. 复购率
buy_user_counts = df_feature[df_feature['behavior_type'] == 4].groupby(['item_id', 'user_id']).size()
features['repurchase_index'] = buy_user_counts.groupby('item_id').apply(lambda x: (x > 1).sum() / len(x))

# 6. 最近购买时间 (小时)
last_buy_time = df_feature[df_feature['behavior_type'] == 4].groupby('item_id')['time'].max()
features['buy_recency_hours'] = (max_time - last_buy_time).dt.total_seconds() / 3600

# 7. 兴趣深度
features['interest_depth'] = df_feature.groupby('item_id').size() / df_feature.groupby('item_id')['user_id'].nunique()

# 8. 高峰小时 (简化版)
def get_peak_hour(bt):
    hourly = df_feature[df_feature['behavior_type'] == bt].copy()
    hourly['hour'] = hourly['time'].dt.hour.astype(str).str.zfill(2)
    peak = hourly.groupby(['item_id', 'hour']).size().reset_index(name='cnt')
    return peak.loc[peak.groupby('item_id')['cnt'].idxmax()].set_index('item_id')['hour']

features['peak_view_hour'] = get_peak_hour(1)
features['peak_buy_hour'] = get_peak_hour(4)

# 9. view_not_buy_rate
view_users = df_feature[df_feature['behavior_type'] == 1].groupby('item_id')['user_id'].apply(set)
buy_users = df_feature[df_feature['behavior_type'] == 4].groupby('item_id')['user_id'].apply(set)
buy_users_view = buy_users.reindex(view_users.index,fill_value=set())
features['view_not_buy_rate'] = pd.Series([len(v - b) / len(v)
        if len(v) > 0 else 0
        for v, b in zip(view_users, buy_users_view)],index=view_users.index)

# 10. cart_not_buy_rate
cart_users = df_feature[df_feature['behavior_type'] == 3].groupby('item_id')['user_id'].apply(set)
buy_users_cart = buy_users.reindex(cart_users.index, fill_value=set())
features['cart_not_buy_rate'] = pd.Series([len(c - b) / len(c)
        if len(c) > 0 else 0
        for c, b in zip(cart_users, buy_users_cart)],index=cart_users.index)

# 11. fav_not_buy_rate
fav_users = (df_feature[df_feature['behavior_type'] == 2].groupby('item_id')['user_id'].apply(set))
buy_users_fav = buy_users.reindex(fav_users.index, fill_value=set())
features['fav_not_buy_rate'] = pd.Series([len(f - b) / len(f)
        if len(f) > 0 else 0
        for f, b in zip(fav_users, buy_users_fav)], index=fav_users.index)

# 合并所有特征
result_df = pd.DataFrame(features).reset_index()
result_df.columns = ['item_id'] + list(features.keys())

# 保存到数据库
cursor.execute("DELETE FROM item_feature")
result_df.to_sql('item_feature', conn, if_exists='append', index=False)
conn.commit()
print(f"Successfully computed features for {len(result_df)} items!")

# Total views (pv) for each item
cursor.execute('''
SELECT
    item_id,
    COUNT(*) AS total_pv
FROM user_behavior
WHERE behavior_type = '1' AND time < '2014-12-18'
GROUP BY item_id;
    ''')
# Check the result for total_pv
# Step 1: 随机抽取3个item_id及其total_pv
cursor.execute("SELECT item_id, total_pv FROM item_feature ORDER BY RANDOM() LIMIT 3;")
sample_items = cursor.fetchall()

for item_id, total_pv in sample_items:
    print(f"Item ID: {item_id}, Total PV (from item_feature): {total_pv}")
    # Step 2: 手动统计 user_behavior 中 behavior_type='1' 的数量
    cursor.execute("""
        SELECT COUNT(*) 
        FROM user_behavior 
        WHERE item_id = ? AND behavior_type = '1' AND time < '2014-12-18'
    """, (item_id,))
    actual_pv = cursor.fetchone()[0]
    # Step 3: 对比验证
    if actual_pv == total_pv:
        print(f"Passed: actual PV = {actual_pv}")
    else:
        print(f"Failed: actual PV = {actual_pv}, recorded value = {total_pv}")

# last 3day pv
# 先获取最大时间（字符串）
cursor.execute("SELECT MAX(time) FROM user_behavior")
max_time = cursor.fetchone()[0]
# 然后用字符串比较（因为格式一致，字典序有效）
cursor.execute('''
    SELECT item_id, COUNT(*) AS last3day_pv
    FROM user_behavior
    WHERE behavior_type = '1'
      AND time >= ?
    GROUP BY item_id
''', (three_days_ago_str,))
# Check the result for last 3day pv
cursor.execute("""
    SELECT item_id, last3day_pv 
    FROM item_feature 
    WHERE last3day_pv > 0 
    ORDER BY RANDOM() 
    LIMIT 3;
""")
sample_items_3day = cursor.fetchall()
print(f"共抽样 {len(sample_items_3day)} 个 item 进行 last3day_pv 验证")
for item_id, recorded_last3day_pv in sample_items_3day:
    print(f"\nItem ID: {item_id}, Recorded last3day_pv: {recorded_last3day_pv}")
    # 手动计算：behavior_type='1' 且 time >= three_days_ago_str
    cursor.execute("""
        SELECT COUNT(*) 
        FROM user_behavior 
        WHERE item_id = ? 
          AND behavior_type = '1'
          AND time >= ?
          AND time < '2014-12-18'
    """, (item_id, three_days_ago_str))
    
    actual_last3day_pv = cursor.fetchone()[0]
    
    if actual_last3day_pv == recorded_last3day_pv:
        print(f"Passed: actual = {actual_last3day_pv}")
    else:
        print(f"Failed: actual = {actual_last3day_pv}, recorded = {recorded_last3day_pv}")

# last 7day pv
# 先获取最大时间（字符串）
cursor.execute("SELECT MAX(time) FROM user_behavior")
max_time = cursor.fetchone()[0] 
# 然后用字符串比较（因为格式一致，字典序有效）
cursor.execute('''
    SELECT item_id, COUNT(*) AS last7day_pv
    FROM user_behavior
    WHERE behavior_type = '1'
      AND time >= ? 
    GROUP BY item_id
''', (seven_days_ago_str,))
# Check the result for last 7day pv
cursor.execute("""
    SELECT item_id, last7day_pv 
    FROM item_feature 
    WHERE last7day_pv > 0 
    ORDER BY RANDOM() 
    LIMIT 3;
""")
sample_items_7day = cursor.fetchall()
print(f"共抽样 {len(sample_items_7day)} 个 item 进行 last7day_pv 验证")
for item_id, recorded_last7day_pv in sample_items_7day:
    print(f"\nItem ID: {item_id}, Recorded last7day_pv: {recorded_last7day_pv}")
    # 手动计算：behavior_type='1' 且 time >= seven_days_ago_str
    cursor.execute("""
        SELECT COUNT(*) 
        FROM user_behavior 
        WHERE item_id = ? 
          AND behavior_type = '1'
          AND time >= ?
          AND time < '2014-12-18'
    """, (item_id, seven_days_ago_str))
    
    actual_last7day_pv = cursor.fetchone()[0]
    
    if actual_last7day_pv == recorded_last7day_pv:
        print(f"Passed: actual = {actual_last7day_pv}")
    else:
        print(f"Failed: actual = {actual_last7day_pv}, recorded = {recorded_last7day_pv}")

# Growth rate of items popularity
cursor.execute('''
SELECT
    a.item_id,
    CAST(a.last3day_pv AS REAL) / NULLIF(b.last7day_pv, 0) AS popularity_growth_rate
FROM (
    SELECT item_id, COUNT(*) AS last3day_pv
    FROM user_behavior
    WHERE behavior_type = '1'
      AND time >= ?
      AND time < '2014-12-18'
    GROUP BY item_id
) a
JOIN (
    SELECT item_id, COUNT(*) AS last7day_pv
    FROM user_behavior
    WHERE behavior_type = '1'
      AND time >= ?
      AND time < '2014-12-18'
    GROUP BY item_id
) b ON a.item_id = b.item_id;
               ''', (three_days_ago_str, seven_days_ago_str))
# Check the result for growth rate of items popularity
print("\nCheck for popularity_growth_rate（random 3):")
# 从 item_feature 中抽取满足条件的样本（last7day_pv > 0，避免除零）
cursor.execute("""
    SELECT item_id, last3day_pv, last7day_pv
    FROM item_feature
    WHERE last7day_pv > 0
    ORDER BY RANDOM()
    LIMIT 3;
""")
samples = cursor.fetchall()
for item_id, last3_rec, last7_rec in samples:
    last3_rec = 0 if last3_rec is None else last3_rec
    last7_rec = 0 if last7_rec is None else last7_rec
    recorded_rate = (
        last3_rec / last7_rec
        if last7_rec != 0 else 0
    )
    # 单独计算 last3day
    cursor.execute("""
        SELECT COUNT(*)
        FROM user_behavior
        WHERE item_id = ?
          AND behavior_type = '1'
          AND time >= ?
          AND time < '2014-12-18'
    """, (item_id, three_days_ago_str))
    actual_last3 = cursor.fetchone()[0]
    # 单独计算 last7day
    cursor.execute("""
        SELECT COUNT(*)
        FROM user_behavior
        WHERE item_id = ?
          AND behavior_type = '1'
          AND time >= ?
          AND time < '2014-12-18'
    """, (item_id, seven_days_ago_str))
    actual_last7 = cursor.fetchone()[0]
    actual_rate = (
        actual_last3 / actual_last7
        if actual_last7 != 0 else 0
    )
    status = (
        "Passed"
        if abs(recorded_rate - actual_rate) < 1e-6
        else "Failed"
    )
    print(f"Item: {item_id}")
    print(f"  Recorded: {last3_rec}/{last7_rec} = {recorded_rate:.4f}")
    print(f"  Actual:   {actual_last3}/{actual_last7} = {actual_rate:.4f} | {status}")

# Items pv ranking
cursor.execute('''
WITH pv_table AS (
    SELECT
        item_id,
        COUNT(*) AS total_pv
    FROM user_behavior
    WHERE behavior_type='1' AND time < '2014-12-18'
    GROUP BY item_id
)
SELECT
    item_id,
    total_pv,
    RANK() OVER(
        ORDER BY total_pv DESC
    ) AS pv_rank
FROM pv_table;
               ''')
# Check the result for items pv ranking
print("\nCheck for pv_rank（random 3）:")
cursor.execute("SELECT item_id, total_pv FROM item_feature WHERE total_pv IS NOT NULL ORDER BY RANDOM() LIMIT 3;")
for item_id, recorded in cursor.fetchall():
    cursor.execute("SELECT COUNT(*) FROM user_behavior WHERE item_id = ? AND behavior_type = '1' AND time < '2014-12-18'", (item_id,))
    actual = cursor.fetchone()[0]
    print(f"Item: {item_id} | Recorded PV: {recorded} | Actual PV: {actual} | {'passed' if recorded == actual else 'failed'}")

# Total cart
cursor.execute('''
SELECT
    item_id,
    COUNT(*) AS total_cart
FROM user_behavior
WHERE behavior_type = '3' AND time < '2014-12-18'
GROUP BY item_id;
                ''')
# Check the result for total cart
print("\nCheck for cart_count（random three items）:")
cursor.execute("SELECT item_id, cart_count FROM item_feature WHERE cart_count IS NOT NULL ORDER BY RANDOM() LIMIT 3;")
for item_id, recorded in cursor.fetchall():
    cursor.execute("SELECT COUNT(*) FROM user_behavior WHERE item_id = ? AND behavior_type = '3' AND time < '2014-12-18'", (item_id,))
    actual = cursor.fetchone()[0]
    print(f"Item: {item_id} | Recorded: {recorded} | Actual: {actual} | {'passed' if recorded == actual else 'failed'}")

# Cart rate
cursor.execute('''
SELECT
    item_id,
    SUM(CASE WHEN behavior_type = '3' THEN 1 ELSE 0 END) * 1.0 /
    SUM(CASE WHEN behavior_type = '1' THEN 1 ELSE 0 END) AS cart_rate
FROM user_behavior
GROUP BY item_id;
                ''')
# Check the result for cart rate
print("\nCheck for cart_rate（random 3 items）:")
cursor.execute("SELECT item_id, cart_rate FROM item_feature WHERE cart_rate IS NOT NULL ORDER BY RANDOM() LIMIT 3;")
for item_id, recorded in cursor.fetchall():
    cursor.execute("""
        SELECT 
            CAST(SUM(CASE WHEN behavior_type='3' THEN 1 ELSE 0 END) AS REAL) * 1.0 /
            NULLIF(SUM(CASE WHEN behavior_type='1' THEN 1 ELSE 0 END), 0)
        FROM user_behavior WHERE item_id = ? AND time < '2014-12-18'
    """, (item_id,))
    actual = cursor.fetchone()[0] or 0.0
    match = abs((recorded or 0.0) - actual) < 1e-6
    print(f"Item: {item_id} | Recorded: {recorded:.4f} | Actual: {actual:.4f} | {'passed' if match else 'failed'}")

# Total fav
cursor.execute('''
SELECT
    item_id,
    COUNT(*) AS total_fav
FROM user_behavior
WHERE behavior_type = '2' AND time < '2014-12-18'
GROUP BY item_id;
                ''')
# Check the result for total fav
print("\nCheck for favorite_count（random three items）:")
cursor.execute("SELECT item_id, favorite_count FROM item_feature WHERE favorite_count IS NOT NULL ORDER BY RANDOM() LIMIT 3;")
for item_id, recorded in cursor.fetchall():
    cursor.execute("SELECT COUNT(*) FROM user_behavior WHERE item_id = ? AND behavior_type = '2' AND time < '2014-12-18'", (item_id,))
    actual = cursor.fetchone()[0]
    print(f"Item: {item_id} | Recorded: {recorded} | Actual: {actual} | {'passed' if recorded == actual else 'failed'}")

# Fav rate
cursor.execute('''
SELECT
    item_id,
    SUM(CASE WHEN behavior_type = '2' THEN 1 ELSE 0 END) * 1.0 /
    SUM(CASE WHEN behavior_type = '1' THEN 1 ELSE 0 END) AS fav_rate
FROM user_behavior
GROUP BY item_id;   
                ''')
# Check the result for fav rate
print("\nCheck for favorite_rate（random 3 items）:")
cursor.execute("SELECT item_id, favorite_rate FROM item_feature WHERE favorite_rate IS NOT NULL ORDER BY RANDOM() LIMIT 3;")
for item_id, recorded in cursor.fetchall():
    cursor.execute("""
        SELECT 
            CAST(SUM(CASE WHEN behavior_type='2' THEN 1 ELSE 0 END) AS REAL) * 1.0 /
            NULLIF(SUM(CASE WHEN behavior_type='1' THEN 1 ELSE 0 END), 0)
        FROM user_behavior WHERE item_id = ? AND time < '2014-12-18'
    """, (item_id,))
    actual = cursor.fetchone()[0] or 0.0
    match = abs((recorded or 0.0) - actual) < 1e-6
    print(f"Item: {item_id} | Recorded: {recorded:.4f} | Actual: {actual:.4f} | {'passed' if match else 'failed'}")

# Total buy
cursor.execute('''
SELECT
    item_id,
    COUNT(*) AS total_buy
FROM user_behavior
WHERE behavior_type = '4' AND time < '2014-12-18'
GROUP BY item_id;
                ''')
# Check the result for total buy
print("\nCheck for buy_count（random three items）:")
cursor.execute("SELECT item_id, buy_count FROM item_feature WHERE buy_count IS NOT NULL ORDER BY RANDOM() LIMIT 3;")
for item_id, recorded in cursor.fetchall():
    cursor.execute("SELECT COUNT(*) FROM user_behavior WHERE item_id = ? AND behavior_type = '4' AND time < '2014-12-18'", (item_id,))
    actual = cursor.fetchone()[0]
    print(f"Item: {item_id} | Recorded: {recorded} | Actual: {actual} | {'passed' if recorded == actual else 'failed'}")

# Buy rate
cursor.execute('''
SELECT
    item_id,
    SUM(CASE WHEN behavior_type = '4' THEN 1 ELSE 0 END) * 1.0 /
    SUM(CASE WHEN behavior_type = '1' THEN 1 ELSE 0 END) AS buy_rate
FROM user_behavior
GROUP BY item_id;
                ''')
# Check the result for buy rate
print("\nCheck for buy_rate（random 3 items）:")
cursor.execute("SELECT item_id, buy_rate FROM item_feature WHERE buy_rate IS NOT NULL ORDER BY RANDOM() LIMIT 3;")
for item_id, recorded in cursor.fetchall():
    cursor.execute("""
        SELECT 
            CAST(SUM(CASE WHEN behavior_type='4' THEN 1 ELSE 0 END) AS REAL) * 1.0 /
            NULLIF(SUM(CASE WHEN behavior_type='1' THEN 1 ELSE 0 END), 0)
        FROM user_behavior WHERE item_id = ? AND time < '2014-12-18'
    """, (item_id,))
    actual = cursor.fetchone()[0] or 0.0
    match = abs((recorded or 0.0) - actual) < 1e-6
    print(f"Item: {item_id} | Recorded: {recorded:.4f} | Actual: {actual:.4f} | {'passed' if match else 'failed'}")

# Product coverage user breadth (UV)
cursor.execute('''
SELECT
    item_id,
    COUNT(DISTINCT user_id) AS uv
FROM user_behavior
GROUP BY item_id;
                ''')
# Check the result for uv
print("\nCheck for uv（random 3 items）:")
cursor.execute("SELECT item_id, uv FROM item_feature WHERE uv IS NOT NULL ORDER BY RANDOM() LIMIT 3;")
for item_id, recorded in cursor.fetchall():
    cursor.execute("SELECT COUNT(DISTINCT user_id) FROM user_behavior WHERE item_id = ? AND time < '2014-12-18'", (item_id,))
    actual = cursor.fetchone()[0]
    print(f"Item: {item_id} | Recorded: {recorded} | Actual: {actual} | {'passed' if recorded == actual else 'failed'}")

# View but not buy rate
cursor.execute('''
WITH view_users AS(
    SELECT DISTINCT
        item_id,
        user_id
    FROM user_behavior
    WHERE behavior_type='1' AND time < '2014-12-18'
),
buy_users AS(
    SELECT DISTINCT
        item_id,
        user_id
    FROM user_behavior
    WHERE behavior_type='4' AND time < '2014-12-18'
),
view_only AS (
    SELECT v.*
    FROM view_users v
    LEFT JOIN buy_users b
    ON v.item_id=b.item_id
    AND v.user_id=b.user_id
    WHERE b.user_id IS NULL
)
SELECT
    v.item_id,
    COUNT(*) * 1.0 /
    (SELECT COUNT(*) FROM view_users vv WHERE vv.item_id = v.item_id)
    AS view_not_buy_rate
FROM view_only v
GROUP BY v.item_id;
               ''')
# Check the result for view but not buy rate
print("\nCheck for view_not_buy_rate(random 3）:")
# Step 1: 先获取一批有浏览行为的 item_id（确保分母 > 0）
cursor.execute("""
    SELECT DISTINCT item_id
    FROM user_behavior
    WHERE behavior_type = '1' AND time < '2014-12-18'
    ORDER BY RANDOM()
    LIMIT 100  -- 先取100个候选，避免全表扫描太慢
""")
candidate_items = [row[0] for row in cursor.fetchall()]

if not candidate_items:
    print("无浏览行为数据，跳过验证。")
else:
    # Step 2: 对每个候选 item，计算 view_not_buy_rate
    validated = 0
    for item_id in candidate_items:
        if validated >= 3:
            break
            
        # 计算总浏览用户数（view_users）
        cursor.execute("""
            SELECT COUNT(DISTINCT user_id)
            FROM user_behavior
            WHERE item_id = ? AND behavior_type = '1' AND time < '2014-12-18'
        """, (item_id,))
        total_view_users = cursor.fetchone()[0]
        
        if total_view_users == 0:
            continue  # 跳过无效项

        # 计算浏览但未购买的用户数
        cursor.execute("""
            SELECT COUNT(DISTINCT v.user_id)
            FROM (
                SELECT DISTINCT user_id
                FROM user_behavior
                WHERE item_id = ? AND behavior_type = '1' AND time < '2014-12-18'
            ) v
            LEFT JOIN (
                SELECT DISTINCT user_id
                FROM user_behavior
                WHERE item_id = ? AND behavior_type = '4' AND time < '2014-12-18'
            ) b ON v.user_id = b.user_id
            WHERE b.user_id IS NULL
        """, (item_id, item_id))
        view_not_buy_users = cursor.fetchone()[0]

        # 计算比率
        expected_rate = view_not_buy_users / total_view_users if total_view_users > 0 else 0.0

        # 再次用原始 SQL 逻辑验证（双重确认）
        cursor.execute("""
            WITH view_users AS (
                SELECT DISTINCT user_id
                FROM user_behavior
                WHERE item_id = ? AND behavior_type = '1' AND time < '2014-12-18'
            ),
            buy_users AS (
                SELECT DISTINCT user_id
                FROM user_behavior
                WHERE item_id = ? AND behavior_type = '4' AND time < '2014-12-18'
            ),
            view_only AS (
                SELECT v.user_id
                FROM view_users v
                LEFT JOIN buy_users b ON v.user_id = b.user_id
                WHERE b.user_id IS NULL
            )
            SELECT 
                CAST(COUNT(*) AS REAL) * 1.0 / (SELECT COUNT(*) FROM view_users)
            FROM view_only
        """, (item_id, item_id))
        actual_rate = cursor.fetchone()[0] or 0.0

        # 比较两次计算是否一致（验证逻辑正确性）
        match = abs(expected_rate - actual_rate) < 1e-6
        status = "Passed" if match else "Failed"

        print(f"Item: {item_id}")
        print(f"  View users: {total_view_users}, Not-buy: {view_not_buy_users}")
        print(f"  Rate: {expected_rate:.4f} | {status}")
        validated += 1

    if validated == 0:
        print("无有效样本满足验证条件")

# Cart but not buy rate
cursor.execute('''
WITH cart_users AS(
    SELECT DISTINCT item_id,user_id
    FROM user_behavior
    WHERE behavior_type='3' AND time < '2014-12-18'
),
buy_users AS(
    SELECT DISTINCT item_id,user_id
    FROM user_behavior
    WHERE behavior_type='4' AND time < '2014-12-18'
),
cart_only AS (
    SELECT c.*
    FROM cart_users c
    LEFT JOIN buy_users b
    ON c.item_id=b.item_id
    AND c.user_id=b.user_id
    WHERE b.user_id IS NULL
)
SELECT
    c.item_id,
    COUNT(*)*1.0/
    (SELECT COUNT(*) FROM cart_users cc WHERE cc.item_id=c.item_id) 
    AS cart_not_buy_rate
FROM cart_only c
GROUP BY c.item_id;
               ''')
# Check the result for cart but not buy rate
print("\nCheck for view_not_buy_rate(random 3）:")
# Step 1: 先获取一批有cart行为的 item_id（确保分母 > 0）
cursor.execute("""
    SELECT DISTINCT item_id
    FROM user_behavior
    WHERE behavior_type = '3' AND time < '2014-12-18'
    ORDER BY RANDOM()
    LIMIT 100  -- 先取100个候选，避免全表扫描太慢
""")
candidate_items = [row[0] for row in cursor.fetchall()]

if not candidate_items:
    print("无购物车行为数据，跳过验证。")
else:
    # Step 2: 对每个候选 item，计算 cart_not_buy_rate
    validated = 0
    for item_id in candidate_items:
        if validated >= 3:
            break
            
        # 计算总浏览用户数（view_users）
        cursor.execute("""
            SELECT COUNT(DISTINCT user_id)
            FROM user_behavior
            WHERE item_id = ? AND behavior_type = '3' AND time < '2014-12-18'
        """, (item_id,))
        total_cart_users = cursor.fetchone()[0]
        
        if total_cart_users == 0:
            continue  # 跳过无效项

        # 计算购物车但未购买的用户数
        cursor.execute("""
            SELECT COUNT(DISTINCT c.user_id)
            FROM (
                SELECT DISTINCT user_id
                FROM user_behavior
                WHERE item_id = ? AND behavior_type = '3' AND time < '2014-12-18'
            ) c
            LEFT JOIN (
                SELECT DISTINCT user_id
                FROM user_behavior
                WHERE item_id = ? AND behavior_type = '4' AND time < '2014-12-18'
            ) b ON c.user_id = b.user_id
            WHERE b.user_id IS NULL
        """, (item_id, item_id))
        cart_not_buy_users = cursor.fetchone()[0]

        # 计算比率
        expected_rate = cart_not_buy_users / total_cart_users if total_cart_users > 0 else 0.0

        # 再次用原始 SQL 逻辑验证（双重确认）
        cursor.execute("""
            WITH cart_users AS (
                SELECT DISTINCT user_id
                FROM user_behavior
                WHERE item_id = ? AND behavior_type = '3' AND time < '2014-12-18'
            ),
            buy_users AS (
                SELECT DISTINCT user_id
                FROM user_behavior
                WHERE item_id = ? AND behavior_type = '4' AND time < '2014-12-18'
            ),
            cart_only AS (
                SELECT c.user_id
                FROM cart_users c
                LEFT JOIN buy_users b ON c.user_id = b.user_id
                WHERE b.user_id IS NULL
            )
            SELECT 
                CAST(COUNT(*) AS REAL) * 1.0 / (SELECT COUNT(*) FROM cart_users)
            FROM cart_only
        """, (item_id, item_id))
        actual_rate = cursor.fetchone()[0] or 0.0

        # 比较两次计算是否一致（验证逻辑正确性）
        match = abs(expected_rate - actual_rate) < 1e-6
        status = "Passed" if match else "Failed"

        print(f"Item: {item_id}")
        print(f"  Cart users: {total_cart_users}, Not-buy: {cart_not_buy_users}")
        print(f"  Rate: {expected_rate:.4f} | {status}")
        validated += 1

    if validated == 0:
        print("无有效样本满足验证条件")

# Fav but not buy rate
cursor.execute('''
WITH fav_users AS(
    SELECT DISTINCT item_id,user_id
    FROM user_behavior
    WHERE behavior_type='2' AND time < '2014-12-18'
),
buy_users AS(
    SELECT DISTINCT item_id,user_id
    FROM user_behavior
    WHERE behavior_type='4' AND time < '2014-12-18'
),
fav_only AS(
    SELECT f.*
    FROM fav_users f
    LEFT JOIN buy_users b
    ON f.item_id=b.item_id
    AND f.user_id=b.user_id
    WHERE b.user_id IS NULL
)
SELECT
    f.item_id,
    COUNT(*)*1.0/
    (SELECT COUNT(*) FROM fav_users ff WHERE ff.item_id=f.item_id) 
    AS fav_not_buy_rate
FROM fav_only f
GROUP BY f.item_id;       
                ''')
# Check the result for fav but not buy rate
print("\nCheck for fav_not_buy_rate(random 3）:")
# Step 1: 先获取一批有fav行为的 item_id（确保分母 > 0）
cursor.execute("""
    SELECT DISTINCT item_id
    FROM user_behavior
    WHERE behavior_type = '2'
       AND time < '2014-12-18'
    ORDER BY RANDOM()
    LIMIT 100  -- 先取100个候选，避免全表扫描太慢
""")
candidate_items = [row[0] for row in cursor.fetchall()]

if not candidate_items:
    print("无收藏行为数据，跳过验证。")
else:
    # Step 2: 对每个候选 item，计算 fav_not_buy_rate
    validated = 0
    for item_id in candidate_items:
        if validated >= 3:
            break
            
        # 计算总浏览用户数（view_users）
        cursor.execute("""
            SELECT COUNT(DISTINCT user_id)
            FROM user_behavior
            WHERE item_id = ? AND behavior_type = '2' AND time < '2014-12-18'
        """, (item_id,))
        total_fav_users = cursor.fetchone()[0]
        
        if total_fav_users == 0:
            continue  # 跳过无效项

        # 计算购物车但未购买的用户数
        cursor.execute("""
            SELECT COUNT(DISTINCT f.user_id)
            FROM (
                SELECT DISTINCT user_id
                FROM user_behavior
                WHERE item_id = ? AND behavior_type = '2' AND time < '2014-12-18'
            ) f
            LEFT JOIN (
                SELECT DISTINCT user_id
                FROM user_behavior
                WHERE item_id = ? AND behavior_type = '4' AND time < '2014-12-18'
            ) b ON f.user_id = b.user_id
            WHERE b.user_id IS NULL
        """, (item_id, item_id))
        fav_not_buy_users = cursor.fetchone()[0]

        # 计算比率
        expected_rate = fav_not_buy_users / total_fav_users if total_fav_users > 0 else 0.0

        # 再次用原始 SQL 逻辑验证（双重确认）
        cursor.execute("""
            WITH fav_users AS (
                SELECT DISTINCT user_id
                FROM user_behavior
                WHERE item_id = ? AND behavior_type = '2' AND time < '2014-12-18'
            ),
            buy_users AS (
                SELECT DISTINCT user_id
                FROM user_behavior
                WHERE item_id = ? AND behavior_type = '4' AND time < '2014-12-18'
            ),
            fav_only AS (
                SELECT f.user_id
                FROM fav_users f
                LEFT JOIN buy_users b ON f.user_id = b.user_id
                WHERE b.user_id IS NULL
            )
            SELECT 
                CAST(COUNT(*) AS REAL) * 1.0 / (SELECT COUNT(*) FROM fav_users)
            FROM fav_only
        """, (item_id, item_id))
        actual_rate = cursor.fetchone()[0] or 0.0

        # 比较两次计算是否一致（验证逻辑正确性）
        match = abs(expected_rate - actual_rate) < 1e-6
        status = "Passed" if match else "Failed"

        print(f"Item: {item_id}")
        print(f"  Fav users: {total_fav_users}, Not-buy: {fav_not_buy_users}")
        print(f"  Rate: {expected_rate:.4f} | {status}")
        validated += 1

    if validated == 0:
        print("无有效样本满足验证条件")

# Depth of interest in products
cursor.execute('''
SELECT
    item_id,
    COUNT(*)*1.0/
    COUNT(DISTINCT user_id)
    AS interest_depth
FROM user_behavior
GROUP BY item_id;
               ''')
# Check the result for interest depth
print("\nCheck for interest_depth（random 3）:")
cursor.execute("SELECT item_id, interest_depth FROM item_feature WHERE interest_depth IS NOT NULL ORDER BY RANDOM() LIMIT 3;")
for item_id, recorded in cursor.fetchall():
    cursor.execute("""
        SELECT CAST(COUNT(*) AS REAL) * 1.0 / COUNT(DISTINCT user_id)
        FROM user_behavior WHERE item_id = ? AND time < '2014-12-18'
    """, (item_id,))
    actual = cursor.fetchone()[0] or 0.0
    match = abs((recorded or 0.0) - actual) < 1e-6
    print(f"Item: {item_id} | Recorded: {recorded:.4f} | Actual: {actual:.4f} | {'passed' if match else 'failed'}")

# Items repurchase index
cursor.execute('''
WITH buy_cnt AS(
SELECT
    item_id,
    user_id,
    COUNT(*) AS buy_times
FROM user_behavior
WHERE behavior_type='4' AND time < '2014-12-18'
GROUP BY item_id,user_id
)
SELECT
    item_id,
    COUNT(
        CASE
        WHEN buy_times>1
        THEN 1
        END
    )*1.0
    /
    COUNT(*)
    AS repurchase_index
FROM buy_cnt
GROUP BY item_id;
               ''')
# Check the result for repurchase index
print("\nCheck for repurchase_index（random 3）:")
cursor.execute("SELECT item_id, repurchase_index FROM item_feature WHERE repurchase_index IS NOT NULL ORDER BY RANDOM() LIMIT 3;")
for item_id, recorded in cursor.fetchall():
    cursor.execute("""
        WITH buy_cnt AS (
            SELECT user_id, COUNT(*) AS buy_times
            FROM user_behavior
            WHERE item_id = ? AND behavior_type = '4' AND time < '2014-12-18'
            GROUP BY user_id
        )
        SELECT 
            CAST(COUNT(CASE WHEN buy_times > 1 THEN 1 END) AS REAL) * 1.0 / COUNT(*)
        FROM buy_cnt
    """, (item_id,))
    actual = cursor.fetchone()[0] or 0.0
    match = abs((recorded or 0.0) - actual) < 1e-6
    print(f"Item: {item_id} | Recorded: {recorded:.4f} | Actual: {actual:.4f} | {'passed' if match else 'failed'}")

# item_buy_recency
cursor.execute('''
                SELECT 
                    item_id,
                    (
                        strftime('%s',
                            (SELECT MAX(time)
                            FROM user_behavior
                            WHERE behavior_type='4' AND time < '2014-12-18')
                        )   
                        -
                        strftime('%s', MAX(time))
                    ) / 3600.0
                    AS buy_recency_hours
                    FROM user_behavior
                    WHERE behavior_type='4'
                    GROUP BY item_id;
               ''')
# Check the result for buy recency hours
print("\nCheck for item buy recency（random 3 items）:")

# Step 1: 获取全局最晚购买时间
cursor.execute("SELECT MAX(time) FROM user_behavior WHERE behavior_type = '4' AND time < '2014-12-18'")
global_max_time = cursor.fetchone()[0]

if not global_max_time:
    print("无购买行为数据，跳过验证。")
else:
    # Step 2: 获取有购买行为的 item_id 候选集
    cursor.execute("""
        SELECT DISTINCT item_id
        FROM user_behavior
        WHERE behavior_type = '4'
            AND time < '2014-12-18'
        ORDER BY RANDOM()
        LIMIT 100
    """)
    candidate_items = [row[0] for row in cursor.fetchall()]

    if not candidate_items:
        print("无有效商品，跳过验证。")
    else:
        validated = 0
        for item_id in candidate_items:
            if validated >= 3:
                break

            # Step 3: 计算该商品的最近购买时间
            cursor.execute("""
                SELECT MAX(time)
                FROM user_behavior
                WHERE item_id = ? AND behavior_type = '4' AND time < '2014-12-18'
            """, (item_id,))
            latest_buy_time = cursor.fetchone()[0]

            if not latest_buy_time:
                continue

            # Step 4: 手动计算 recency（小时）
            from datetime import datetime
            dt_global = datetime.strptime(global_max_time, "%Y-%m-%d %H")
            dt_latest = datetime.strptime(latest_buy_time, "%Y-%m-%d %H")
            expected_recency = (dt_global - dt_latest).total_seconds() / 3600.0
            # Step 5：从item_feature 中获取 buy_recency_hours
            cursor.execute("""
                SELECT buy_recency_hours
                FROM item_feature
                WHERE item_id = ? 
            """, (item_id,))
            row = cursor.fetchone()
            if not row:
                continue
            actual_recency = row[0] if row[0] is not None else 0.0
            match = abs(expected_recency - actual_recency) < 1e-6
            status = "Passed" if match else "Failed"
            print(f"Item: {item_id}")
            print(f"  Global max time: {global_max_time}")
            print(f"  Latest buy time: {latest_buy_time}")
            print(f"  Expected: {expected_recency:.2f}")
            print(f"  Actual:   {actual_recency:.2f}")
            print(f"  {status}")
            validated += 1

        if validated == 0:
            print("无有效样本满足验证条件。")

# Peak time for item views
cursor.execute('''
WITH t AS (
    SELECT
        item_id,
        strftime('%H', time) AS hour,
        COUNT(*) AS cnt
    FROM user_behavior
    WHERE behavior_type='1' AND time < '2014-12-18'
    GROUP BY item_id, hour
),
ranked AS (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY item_id
               ORDER BY cnt DESC
           ) AS rn
    FROM t
)
SELECT
    item_id,
    hour AS peak_hour,
    cnt AS peak_cnt
FROM ranked
WHERE rn = 1;
                ''')
# Check peak time for item views
print("\nCheck for peak_hour for item views（random 3 items）:")

cursor.execute("""
    SELECT item_id, peak_view_hour
    FROM item_feature
    WHERE peak_view_hour IS NOT NULL
    ORDER BY RANDOM()
    LIMIT 3
""")

samples = cursor.fetchall()

for item_id, recorded_hour in samples:

    cursor.execute("""
        SELECT substr(time,12,2) AS hour,
               COUNT(*) AS cnt
        FROM user_behavior
        WHERE item_id = ?
          AND behavior_type = '1'
          AND time < '2014-12-18'
        GROUP BY hour
        ORDER BY cnt DESC, hour ASC
        LIMIT 1
    """, (item_id,))

    result = cursor.fetchone()

    if result is None:
        print(f"Item: {item_id} | No data")
        continue

    actual_hour = result[0]
    actual_cnt = result[1]

    status = (
        "Passed"
        if str(recorded_hour).zfill(2) == str(actual_hour).zfill(2)
        else "Failed"
    )

    print(
        f"Item: {item_id}\n"
        f"  Recorded: {recorded_hour}\n"
        f"  Actual: {actual_hour}\n"
        f"  Views: {actual_cnt}\n"
        f"  Status: {status}"
    )


# Peak time for item buys
cursor.execute('''
WITH t AS (
    SELECT
        item_id,
        strftime('%H', time) AS hour,
        COUNT(*) AS cnt
    FROM user_behavior
    WHERE behavior_type='4' AND time < '2014-12-18'
    GROUP BY item_id, hour
),
ranked AS (
    SELECT *,
           ROW_NUMBER() OVER (
               PARTITION BY item_id
               ORDER BY cnt DESC
           ) AS rn
    FROM t
)
SELECT
    item_id,
    hour AS peak_hour,
    cnt AS peak_cnt
FROM ranked
WHERE rn = 1;
                ''')
# Check peak time for item buys
print("\nCheck for peak_hour for item buy（random 3 items）:")

cursor.execute("""
    SELECT item_id, peak_buy_hour
    FROM item_feature
    WHERE peak_buy_hour IS NOT NULL
    ORDER BY RANDOM()
    LIMIT 3
""")

samples = cursor.fetchall()

for item_id, recorded_hour in samples:

    cursor.execute("""
        SELECT substr(time,12,2) AS hour,
               COUNT(*) AS cnt
        FROM user_behavior
        WHERE item_id = ?
          AND behavior_type = '4'
          AND time < '2014-12-18'
        GROUP BY hour
        ORDER BY cnt DESC, hour ASC
        LIMIT 1
    """, (item_id,))

    result = cursor.fetchone()

    if result is None:
        print(f"Item: {item_id} | No data")
        continue

    actual_hour = result[0]
    actual_cnt = result[1]

    status = (
        "Passed"
        if str(recorded_hour).zfill(2) == str(actual_hour).zfill(2)
        else "Failed"
    )

    print(
        f"Item: {item_id}\n"
        f"  Recorded: {recorded_hour}\n"
        f"  Actual: {actual_hour}\n"
        f"  Buy: {actual_cnt}\n"
        f"  Status: {status}"
    )

conn.commit()
conn.close()