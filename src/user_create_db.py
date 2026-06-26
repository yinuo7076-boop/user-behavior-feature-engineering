import sqlite3
import pandas as pd
import random
from datetime import datetime, timedelta
sqlite3.connect('middle_user.db')
conn = sqlite3.connect('middle_user.db')
cursor = conn.cursor()
# Create the feature table
cursor.execute('''CREATE TABLE IF NOT EXISTS user_feature(
               user_id TEXT PRIMARY KEY,
               total_behavior REAL,
               total_views REAL,
               last_1day_pv REAL,
               last_3day_pv REAL,
               last_7day_pv REAL,
               activity_growth_rate REAL,
               total_cart REAL,
               cart_rate REAL,
               total_fav REAL,
               fav_rate REAL,
               total_buy REAL,
               buy_rate REAL,
               first_active_time REAL,
               last_active_time REAL,
               buy_recency REAL,
               tendency_index REAL,
               view_not_buy_rate REAL,
               cart_not_buy_rate REAL,
               fav_not_buy_rate REAL,
               impulsive_buying_rate REAL,
               hesitation_buying_rate REAL,
               interest_concentration REAL,
               active_hour TEXT,
               morning_active_rate REAL,
               afternoon_active_rate REAL,
               night_active_rate REAL
)
''')
# Load data from CSV and insert into the database
df = pd.read_csv("data/data_min.csv")
df.to_sql('user_behavior', conn, if_exists='replace', index=False)
print(df.head())
print("Data inserted into the database successfully.")
# Create indexes to speed up queries
cursor.execute('''CREATE INDEX IF NOT EXISTS idx_user_id ON user_behavior(user_id)''')
cursor.execute('''CREATE INDEX IF NOT EXISTS idx_behavior_type ON user_behavior(behavior_type)''')
cursor.execute('''CREATE INDEX IF NOT EXISTS idx_user_time ON user_behavior(time)''')
cursor.execute('''CREATE INDEX IF NOT EXISTS idx_user_type_time ON user_behavior(user_id, behavior_type, time)''')
conn.commit()
print("Index created successfully.")
# Create a new table with users features
cursor.execute('''SELECT * FROM user_behavior LIMIT 10''')

# 构建完整特征查询
# 用 Pandas 高效计算所有特征
print("Starting feature engineering with Pandas...")
# 转换时间列

df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H')
# 预测12月18，所以特征只能用12月17及以前的数据
cutoff_date = pd.Timestamp('2014-12-18')
df_feature = df[df['time'] < cutoff_date].copy()
df_feature['hour'] = pd.to_datetime(df['time']).dt.hour
# 全局时间窗口
max_time = df_feature['time'].max()
one_day_ago = max_time - timedelta(days=1)
three_days_ago = max_time - timedelta(days=3)
seven_days_ago = max_time - timedelta(days=7)
print(max_time)
# 初始化特征字典
features = {}
# 1. total_behavior
features['total_behavior'] = (df_feature.groupby('user_id').size())
# 2. total_view
features['total_views'] = (df_feature[df_feature['behavior_type'] == 1].groupby('user_id').size())
# 345. 时间窗口 PV
pv_all = features['total_views']
pv_1d = df_feature[(df_feature['behavior_type'] == 1) & (df['time'] >= one_day_ago)].groupby('user_id').size()
pv_3d = df_feature[(df_feature['behavior_type'] == 1) & (df['time'] >= three_days_ago)].groupby('user_id').size()
pv_7d = df_feature[(df_feature['behavior_type'] == 1) & (df['time'] >= seven_days_ago)].groupby('user_id').size()
features['last_1day_pv'] = pv_1d
features['last_3day_pv'] = pv_3d
features['last_7day_pv'] = pv_7d
# 6. activity_growth_rate
features['activity_growth_rate'] = pv_3d / pv_7d.replace(0, 1)
# 7. total_cart
features['total_cart'] = (df_feature[df_feature['behavior_type'] == 3].groupby('user_id').size())
# 8. cart_rate
features['cart_rate'] = (features['total_cart'] / features['total_views'].replace(0, pd.NA))
# 9. total_fav
features['total_fav'] = (df_feature[df_feature['behavior_type'] == 2].groupby('user_id').size())
# 10. fav_rate
features['fav_rate'] = (features['total_fav'] / features['total_views'].replace(0, pd.NA))
# 11. total_buy
features['total_buy'] = (df_feature[df_feature['behavior_type'] == 4].groupby('user_id').size())
# 12. buy_rate
features['buy_rate'] = (features['total_buy'] / features['total_views'].replace(0, pd.NA))
# 13. first active time
first_active = (pd.to_datetime(df_feature['time']).groupby(df_feature['user_id']).min())
features['first_active_time'] = (max_time - first_active).dt.days
# 14. last active time
last_active = (pd.to_datetime(df_feature['time']).groupby(df_feature['user_id']).max())
features['last_active_time'] = (max_time - last_active).dt.days
# 15. buy_recency
last_buy_time = df_feature[df_feature['behavior_type'] == 4].groupby('user_id')['time'].max()
features['buy_recency'] = (max_time - last_buy_time).dt.total_seconds() / 3600
# 16. tendency_index
features['tendency_index'] = ((features['total_buy'].fillna(0) + features['total_cart'].fillna(0)) / features['total_views'].replace(0, pd.NA))
# 17. view_not_buy_rate
view_users = df_feature[df_feature['behavior_type'] == 1].groupby('user_id')['item_id'].apply(set)
buy_users = df_feature[df_feature['behavior_type'] == 4].groupby('user_id')['item_id'].apply(set)
buy_users_view = buy_users.reindex(view_users.index,fill_value=set())
features['view_not_buy_rate'] = pd.Series([len(v - b) / len(v)
        if len(v) > 0 else 0
        for v, b in zip(view_users, buy_users_view)],index=view_users.index)
# 18. cart_not_buy_rate
cart_users = df_feature[df_feature['behavior_type'] == 3].groupby('user_id')['item_id'].apply(set)
buy_users_cart = buy_users.reindex(cart_users.index, fill_value=set())
features['cart_not_buy_rate'] = pd.Series([len(c - b) / len(c)
        if len(c) > 0 else 0
        for c, b in zip(cart_users, buy_users_cart)],index=cart_users.index)
# 19. fav_not_buy_rate
fav_users = (df_feature[df_feature['behavior_type'] == 2].groupby('user_id')['item_id'].apply(set))
buy_users_fav = buy_users.reindex(fav_users.index, fill_value=set())
features['fav_not_buy_rate'] = pd.Series([len(f - b) / len(f)
        if len(f) > 0 else 0
        for f, b in zip(fav_users, buy_users_fav)], index=fav_users.index)
# 20. impulsive_buying_rate
features['impulsive_buying_rate'] = (features['total_buy'] / features['total_views'].replace(0, pd.NA))
# 21. hesitation_buying_rate
features['hesitation_buying_rate'] = (features['total_views'] / features['total_buy'].replace(0, pd.NA))
# 22. interest_concentration
category_cnt = (df_feature.groupby(['user_id', 'item_category']).size())
max_category_cnt = (category_cnt.groupby('user_id').max())
features['interest_concentration'] = (max_category_cnt / features['total_behavior'])
# 23. active_hour
active_hour = (df_feature.groupby(['user_id','hour']).size().reset_index(name='cnt'))
active_hour = (active_hour.loc[active_hour.groupby('user_id')['cnt'].idxmax()].set_index('user_id')['hour'])
features['active_hour'] = active_hour
# 24. morning_buy_rate
morning = (df_feature[df_feature['hour'].between(6,13)].groupby('user_id').size())
features['morning_active_rate'] = (morning / features['total_behavior'])
# 25. afternoon_buy_rate
afternoon= (df_feature[df_feature['hour'].between(14,21)].groupby('user_id').size())
features['afternoon_active_rate'] = (afternoon / features['total_behavior'])
# 26. night_buy_rate
night = (df_feature[(df_feature['hour'] >= 22) | (df_feature['hour'] < 6) ].groupby('user_id').size())
features['night_active_rate'] = (night / features['total_behavior'])

# 合并所有特征
result_df = pd.DataFrame(features).reset_index()
result_df.columns = ['user_id'] + list(features.keys())

print(result_df.shape)
print(result_df['user_id'].nunique())
print(
    result_df['user_id']
    .value_counts()
    .head(20)
)

# 删除旧表
cursor.execute("DROP TABLE IF EXISTS user_feature")
conn.commit()

# 保存到数据库
result_df.to_sql('user_feature', conn, if_exists='append', index=False)
conn.commit()
print(f"Successfully computed features for {len(result_df)} users!")

# total behavior
cursor.execute('''
SELECT
    user_id,
    COUNT(*) AS total_behavior
FROM user_behavior
GROUP BY user_id;
               ''')
# total views
cursor.execute('''
SELECT
    user_id,
    COUNT(*) AS total_views
FROM user_behavior
WHERE behavior_type = '1'
GROUP BY user_id;
               ''')
rows = cursor.fetchall()
# Check if the result is empty
if not rows:
    print("No data found.")
else:    
    print("Feature rows: ")
    print(rows[:10])
    # Randomly check three users
    sample_users = random.sample(rows, 3)
    print("Randomly selected users for checking total views:")
    for user in sample_users:
        user_id = user[0]
        user_views = user[1]
        print(f"Checking user_id: {user_id}")
    # Check the original data for users
    cursor.execute(f'''
    SELECT *
    FROM user_behavior
    WHERE user_id = {user_id}
    AND behavior_type = '1';
                     ''')
    check_rows = cursor.fetchall()
    actual_views = len(check_rows)
    print(f"check rows: {user_views}")
    print(f"actual rows: {actual_views}")
    # Check if the counts match
    if user_views == actual_views:
        print("Passed")
    else:
        print("Failed")

# last 1day pv
cursor.execute('''
SELECT
    user_id,
    COUNT(*) AS last_1day_pv
FROM user_behavior
WHERE behavior_type = '1'
AND DATE(time) >= DATE((SELECT MAX(time) FROM user_behavior), '-1 day')
GROUP BY user_id;
                ''')
# Check the result for last 1day pv
rows = cursor.fetchall()
from datetime import datetime, timedelta
cursor.execute('SELECT MAX(time) FROM user_behavior')
max_time_str = cursor.fetchone()
max_time = max_time_str[0] if max_time_str else None
if max_time:
    max_time_dt = datetime.strptime(max_time, '%Y-%m-%d %H')
    new_time_dt = max_time_dt - timedelta(days=1)
    new_time_str = new_time_dt.strftime('%Y-%m-%d %H')
    print(f"Max time: {max_time}")
    print(f"New time: {new_time_str}")
    cursor.execute('''
        SELECT user_id, COUNT(*) AS last_1day_pv
        FROM user_behavior
        WHERE behavior_type = '1'
        AND time >= ?
        GROUP BY user_id;
    ''', (new_time_str,))
    rows = cursor.fetchall()
    if not rows:
        print("Still no data found after fix attempt. Please manually fix the issue.")      
    else:
        print(f"Found {len(rows)} users with activity in the last 1 days.")
        print("Last 1-day PV rows: ")
        print(rows[:10])
        # Sampling 3 users to check the original data
        sample_size = min(3, len(rows))
        sample_users = random.sample(rows, sample_size)
        if sample_size > 0:
            print("Randomly selected users for checking last 1-day PV:")
            for user in sample_users:
                user_id = user[0]
                calculated_pv = user[1]
                cursor.execute(f'''
                    SELECT COUNT(*)
                    FROM user_behavior
                    WHERE user_id = ?
                    AND behavior_type = '1'
                    AND time >= ?
                ''', (user_id, new_time_str))
                check_rows = cursor.fetchone()
                actual_pv = check_rows[0] if check_rows else 0
                print(f"calculated pv: {calculated_pv}")
                print(f"actual pv: {actual_pv}")
                if calculated_pv == actual_pv:
                    print("Passed")
                else:
                    print("Failed")

# last 3day pv
cursor.execute('''
SELECT
    user_id,
    COUNT(*) AS last_3day_pv
FROM user_behavior
WHERE behavior_type = '1'
AND DATE(time) >= DATE((SELECT MAX(time) FROM user_behavior), '-3 day')
GROUP BY user_id;
                ''')
# Check the result for last 3day pv
rows = cursor.fetchall()
from datetime import datetime, timedelta
cursor.execute('SELECT MAX(time) FROM user_behavior')
max_time_str = cursor.fetchone()
max_time = max_time_str[0] if max_time_str else None
if max_time:
    max_time_dt = datetime.strptime(max_time, '%Y-%m-%d %H')
    new_time_dt = max_time_dt - timedelta(days=3)
    new_time_str = new_time_dt.strftime('%Y-%m-%d %H')
    print(f"Max time: {max_time}")
    print(f"New time: {new_time_str}")
    cursor.execute('''
        SELECT user_id, COUNT(*) AS last_3day_pv
        FROM user_behavior
        WHERE behavior_type = '1'
        AND time >= ?
        GROUP BY user_id;
    ''', (new_time_str,))
    rows = cursor.fetchall()
    if not rows:
        print("Still no data found after fix attempt. Please manually fix the issue.")      
    else:
        print(f"Found {len(rows)} users with activity in the last 3 days.")
        print("Last 3-day PV rows: ")
        print(rows[:10])
        # Sampling 3 users to check the original data
        sample_size = min(3, len(rows))
        sample_users = random.sample(rows, sample_size)
        if sample_size > 0:
            print("Randomly selected users for checking last 3-day PV:")
            for user in sample_users:
                user_id = user[0]
                calculated_pv = user[1]
                cursor.execute(f'''
                    SELECT COUNT(*)
                    FROM user_behavior
                    WHERE user_id = ?
                    AND behavior_type = '1'
                    AND time >= ?
                ''', (user_id, new_time_str))
                check_rows = cursor.fetchone()
                actual_pv = check_rows[0] if check_rows else 0
                print(f"calculated pv: {calculated_pv}")
                print(f"actual pv: {actual_pv}")
                if calculated_pv == actual_pv:
                    print("Passed")
                else:
                    print("Failed")

# last 7day pv
cursor.execute('''
SELECT
    user_id,
    COUNT(*) AS last_7day_pv
FROM user_behavior
WHERE behavior_type = '1'
AND DATE(time) >= DATE((SELECT MAX(time) FROM user_behavior), '-7 day')
GROUP BY user_id;
                ''')
# Check the result for last 7day pv
rows = cursor.fetchall()
from datetime import datetime, timedelta
cursor.execute('SELECT MAX(time) FROM user_behavior')
max_time_str = cursor.fetchone()
max_time = max_time_str[0] if max_time_str else None
if max_time:
    max_time_dt = datetime.strptime(max_time, '%Y-%m-%d %H')
    new_time_dt = max_time_dt - timedelta(days=7)
    new_time_str = new_time_dt.strftime('%Y-%m-%d %H')
    print(f"Max time: {max_time}")
    print(f"New time: {new_time_str}")
    cursor.execute('''
        SELECT user_id, COUNT(*) AS last_7day_pv
        FROM user_behavior
        WHERE behavior_type = '1'
        AND time >= ?
        GROUP BY user_id;
    ''', (new_time_str,))
    rows = cursor.fetchall()
    if not rows:
        print("Still no data found after fix attempt. Please manually fix the issue.")      
    else:
        print(f"Found {len(rows)} users with activity in the last 7 days.")
        print("Last 7-day PV rows: ")
        print(rows[:10])
        # Sampling 3 users to check the original data
        sample_size = min(3, len(rows))
        sample_users = random.sample(rows, sample_size)
        if sample_size > 0:
            print("Randomly selected users for checking last 7-day PV:")
            for user in sample_users:
                user_id = user[0]
                calculated_pv = user[1]
                cursor.execute(f'''
                    SELECT COUNT(*)
                    FROM user_behavior
                    WHERE user_id = ?
                    AND behavior_type = '1'
                    AND time >= ?
                ''', (user_id, new_time_str))
                check_rows = cursor.fetchone()
                actual_pv = check_rows[0] if check_rows else 0
                print(f"calculated pv: {calculated_pv}")
                print(f"actual pv: {actual_pv}")
                if calculated_pv == actual_pv:
                    print("Passed")
                else:
                    print("Failed")

# Activity growth rate
cursor.execute('''
SELECT
    a.user_id,
    a.last_3day_pv * 1.0 / b.last_7day_pv AS activity_growth_rate
FROM
(
    SELECT
        user_id,
        COUNT(*) AS last_3day_pv
    FROM user_behavior
    WHERE behavior_type = '1'
    AND DATE(time) >= DATE((SELECT MAX(time) FROM user_behavior), '-3 day')
    GROUP BY user_id
) a
JOIN
(
    SELECT
        user_id,
        COUNT(*) AS last_7day_pv
    FROM user_behavior
    WHERE behavior_type = '1'
    AND DATE(time) >= DATE((SELECT MAX(time) FROM user_behavior), '-7 day')
    GROUP BY user_id
) b
ON a.user_id = b.user_id;
               ''')
# Check the result for activity growth rate
rows = cursor.fetchall()
from datetime import datetime, timedelta
cursor.execute('SELECT MAX(time) FROM user_behavior')
max_time_str = cursor.fetchone()
max_time = max_time_str[0] if max_time_str else None
if not max_time:
    print("No data found. Please check the data source.")
else:
    max_time_dt = datetime.strptime(max_time, '%Y-%m-%d %H')
    new_time_3day_str = (max_time_dt - timedelta(days=3)).strftime('%Y-%m-%d %H')
    new_time_7day_str = (max_time_dt - timedelta(days=7)).strftime('%Y-%m-%d %H')
    print(f"Max time: {max_time}")
    print(f"New time for 3 days: {new_time_3day_str}")
    print(f"New time for 7 days: {new_time_7day_str}")
    cursor.execute('''
        SELECT a.user_id, a.last_3day_pv * 1.0 / b.last_7day_pv AS activity_growth_rate
        FROM (
            SELECT user_id, COUNT(*) AS last_3day_pv
            FROM user_behavior
            WHERE behavior_type = '1' AND time >= ?
            GROUP BY user_id
                   ) a
        JOIN (
            SELECT user_id, COUNT(*) AS last_7day_pv
            FROM user_behavior
            WHERE behavior_type = '1' AND time >= ?
            GROUP BY user_id
                   ) b
        ON a.user_id = b.user_id;
    ''', (new_time_3day_str, new_time_7day_str))
    rows = cursor.fetchall()
    if not rows:
        print("Still no data found after fix attempt. Please manually fix the issue.")
    else:
        print(f"Found {len(rows)} users with activity growth rate.")
        print("Activity growth rate rows: ")
        print(rows[:10])
        # Sampling 3 users to check the original data
        sample_size = min(3, len(rows))
        sample_users = random.sample(rows, sample_size)
        if sample_size > 0:
            print("Randomly selected users for checking activity growth rate:")
            for user in sample_users:
                user_id = user[0]
                calculated_growth_rate = user[1]
                cursor.execute(f'''
                    SELECT
                        (SELECT COUNT(*) FROM user_behavior WHERE user_id = ? AND behavior_type = '1' AND time >= ?) * 1.0 /
                        (SELECT COUNT(*) FROM user_behavior WHERE user_id = ? AND behavior_type = '1' AND time >= ?)
                ''', (user_id, new_time_3day_str, user_id, new_time_7day_str))
                check_rows = cursor.fetchone()
                actual_growth_rate = check_rows[0] if check_rows else None
                print(f"calculated growth rate: {calculated_growth_rate}")
                print(f"actual growth rate: {actual_growth_rate}")
                if calculated_growth_rate == actual_growth_rate:
                    print("Passed")
                else:
                    print("Failed")

# Total cart
cursor.execute('''
SELECT
    user_id,
    COUNT(*) AS total_cart
FROM user_behavior
WHERE behavior_type = '3'
GROUP BY user_id;
                ''')
# Check if the result is empty
rows = cursor.fetchall()
if not rows:
    print("No data found.")
else:    
    print("Feature rows: ")
    print(rows[:10])
    # Randomly check three users
    sample_users = random.sample(rows, 3)
    print("Randomly selected users for checking cart rate:")
    for user in sample_users:
        user_id = user[0]
        user_views = user[1]
        print(f"Checking user_id: {user_id}")
    # Check the original data for users
        cursor.execute('''
            SELECT COUNT(*)
            FROM user_behavior
            WHERE user_id = ?
            AND behavior_type = '3';
         ''', (user_id,))
        check_rows = cursor.fetchone()
        actual_views = check_rows[0] if check_rows else 0
        print(f"check rows: {user_views}")
        print(f"actual rows: {actual_views}")
        # Check if the counts match
        if user_views == actual_views:
            print("Passed")
        else:
            print("Failed")

# Cart rate
cursor.execute('''
SELECT
    user_id,
    SUM(CASE WHEN behavior_type = '3' THEN 1 ELSE 0 END) * 1.0 /
    SUM(CASE WHEN behavior_type = '1' THEN 1 ELSE 0 END) AS cart_rate
FROM user_behavior
GROUP BY user_id;
                ''')
# Check the result for cart rate
rows = cursor.fetchall()
if not rows:
    print("No data found.")
else:    
    print("Feature rows: ")
    print(rows[:10])
    # Randomly check three users
    sample_users = random.sample(rows, 3)
    print("Randomly selected users for checking cart rate:")
    for user in sample_users:
        user_id = user[0]
        user_views = user[1]
        print(f"Checking user_id: {user_id}")
        cursor.execute('''
            SELECT
                SUM(CASE WHEN behavior_type = '3' THEN 1 ELSE 0 END) * 1.0 /
                SUM(CASE WHEN behavior_type = '1' THEN 1 ELSE 0 END)
            FROM user_behavior
            WHERE user_id = ?;
        ''', (user_id,))
        check_rows = cursor.fetchone()
        actual_rate = check_rows[0] if check_rows else None
        print(f"calculated cart rate: {user_views}")
        print(f"actual cart rate: {actual_rate}")
        if user_views == actual_rate:
            print("Passed")
        else:
            print("Failed")

# Total fav
cursor.execute('''
SELECT
    user_id,
    COUNT(*) AS total_fav
FROM user_behavior
WHERE behavior_type = '2'
GROUP BY user_id;
                ''')
# Check if the result is empty
rows = cursor.fetchall()
if not rows:
    print("No data found.")
else:    
    print("Feature rows: ")
    print(rows[:10])
    # Randomly check three users
    sample_users = random.sample(rows, 3)
    print("Randomly selected users for checking total fav:")
    for user in sample_users:
        user_id = user[0]
        user_views = user[1]
        print(f"Checking user_id: {user_id}")
    # Check the original data for users
        cursor.execute('''
            SELECT COUNT(*)
            FROM user_behavior
            WHERE user_id = ?
            AND behavior_type = '2';
         ''', (user_id,))
        check_rows = cursor.fetchone()
        actual_views = check_rows[0] if check_rows else 0
        print(f"check rows: {user_views}")
        print(f"actual rows: {actual_views}")
        # Check if the counts match
        if user_views == actual_views:
            print("Passed")
        else:
            print("Failed")

# Fav rate
cursor.execute('''
SELECT
    user_id,
    SUM(CASE WHEN behavior_type = '2' THEN 1 ELSE 0 END) * 1.0 /
    SUM(CASE WHEN behavior_type = '1' THEN 1 ELSE 0 END) AS fav_rate
FROM user_behavior
GROUP BY user_id;   
                ''')
# Check the result for fav rate
rows = cursor.fetchall()
if not rows:
    print("No data found.")
else:    
    print("Feature rows: ")
    print(rows[:10])
    # Randomly check three users
    sample_users = random.sample(rows, 3)
    print("Randomly selected users for checking fav rate:")
    for user in sample_users:
        user_id = user[0]
        user_views = user[1]
        print(f"Checking user_id: {user_id}")
        cursor.execute('''
            SELECT
                SUM(CASE WHEN behavior_type = '2' THEN 1 ELSE 0 END) * 1.0 /
                SUM(CASE WHEN behavior_type = '1' THEN 1 ELSE 0 END)
            FROM user_behavior
            WHERE user_id = ?;
        ''', (user_id,))
        check_rows = cursor.fetchone()
        actual_rate = check_rows[0] if check_rows else None
        print(f"calculated fav rate: {user_views}")
        print(f"actual fav rate: {actual_rate}")
        if user_views == actual_rate:
            print("Passed")
        else:
            print("Failed")

# Total buy
cursor.execute('''
SELECT
    user_id,
    COUNT(*) AS total_buy
FROM user_behavior
WHERE behavior_type = '4'
GROUP BY user_id;
                ''')
# Check if the result is empty
rows = cursor.fetchall()
if not rows:
    print("No data found.")
else:    
    print("Feature rows: ")
    print(rows[:10])
    # Randomly check three users
    sample_users = random.sample(rows, 3)
    print("Randomly selected users for checking total buy:")
    for user in sample_users:
        user_id = user[0]
        user_views = user[1]
        print(f"Checking user_id: {user_id}")
    # Check the original data for users
        cursor.execute('''
            SELECT COUNT(*)
            FROM user_behavior
            WHERE user_id = ?
            AND behavior_type = '4';
         ''', (user_id,))
        check_rows = cursor.fetchone()
        actual_views = check_rows[0] if check_rows else 0
        print(f"check rows: {user_views}")
        print(f"actual rows: {actual_views}")
        # Check if the counts match
        if user_views == actual_views:
            print("Passed")
        else:
            print("Failed")

# Buy rate
cursor.execute('''
SELECT
    user_id,
    SUM(CASE WHEN behavior_type = '4' THEN 1 ELSE 0 END) * 1.0 /
    SUM(CASE WHEN behavior_type = '1' THEN 1 ELSE 0 END) AS buy_rate
FROM user_behavior
GROUP BY user_id;
                ''')
# Check the result for buy rate
rows = cursor.fetchall()
if not rows:
    print("No data found.")
else:    
    print("Feature rows: ")
    print(rows[:10])
    # Randomly check three users
    sample_users = random.sample(rows, 3)
    print("Randomly selected users for checking buy rate:")
    for user in sample_users:
        user_id = user[0]
        user_views = user[1]
        print(f"Checking user_id: {user_id}")
        cursor.execute('''
            SELECT
                SUM(CASE WHEN behavior_type = '4' THEN 1 ELSE 0 END) * 1.0 /
                SUM(CASE WHEN behavior_type = '1' THEN 1 ELSE 0 END)
            FROM user_behavior
            WHERE user_id = ?;
        ''', (user_id,))
        check_rows = cursor.fetchone()
        actual_rate = check_rows[0] if check_rows else None
        print(f"calculated buy rate: {user_views}")
        print(f"actual buy rate: {actual_rate}")
        if user_views == actual_rate:
            print("Passed")
        else:
            print("Failed")

# First active time
cursor.execute('''
SELECT
    user_id,
    MIN(time) AS first_active_time
FROM user_behavior
GROUP BY user_id;
                ''')
# Check the result for earliest active time
rows = cursor.fetchall()
if not rows:
    print("No data found.")
else:
    print("Earliest active time rows:")
    print(rows[:10])
    # Random check three users
    sample_size = min(3, len(rows))
    sample_users = random.sample(rows, sample_size)
    print("Randomly selected users for checking earliest active time:")
    for user in sample_users:
        user_id = user[0]
        calculated_time = user[1]
        print(f"Checking user_id: {user_id}")
        cursor.execute('''
            SELECT MIN(time)
            FROM user_behavior
            WHERE user_id = ?
        ''', (user_id,))
        check_row = cursor.fetchone()
        actual_time = check_row[0] if check_row else None
        print(f"Calculated earliest time: {calculated_time}")
        print(f"Actual earliest time: {actual_time}")
        if calculated_time == actual_time:
            print("Passed")
        else:
            print("Failed")

# Last active time
cursor.execute('''
SELECT
    user_id,
    MAX(time) AS last_active_time
FROM user_behavior
GROUP BY user_id;
                ''')
# Check the result for last active time
rows = cursor.fetchall()
if not rows:
    print("No data found.")
else:
    print("Last active time rows:")
    print(rows[:10])
    # Random check three users
    sample_size = min(3, len(rows))
    sample_users = random.sample(rows, sample_size)
    print("Randomly selected users for checking last active time:")
    for user in sample_users:
        user_id = user[0]
        calculated_time = user[1]
        print(f"Checking user_id: {user_id}")
        cursor.execute('''
            SELECT MAX(time)
            FROM user_behavior
            WHERE user_id = ?
        ''', (user_id,))
        check_row = cursor.fetchone()
        actual_time = check_row[0] if check_row else None
        print(f"Calculated last active time: {calculated_time}")
        print(f"Actual last active time: {actual_time}")
        if calculated_time == actual_time:
            print("Passed")
        else:
            print("Failed")

# Buy recency
cursor.execute('''
SELECT
    user_id,
    JULIANDAY((SELECT MAX(time) FROM user_behavior)) -
    JULIANDAY(MAX(time)) AS buy_recency
FROM user_behavior
WHERE behavior_type = '4'
GROUP BY user_id;
                ''')
# Check the result for buy recency
from datetime import datetime
cursor.execute('SELECT MAX(time) FROM user_behavior')
max_time_str = cursor.fetchone()[0]
if not max_time_str:
    print("No data found.")
else:
    max_time_dt = datetime.strptime(max_time_str, '%Y-%m-%d %H')
    print(f"Max time: {max_time_dt}")
    cursor.execute('''
        SELECT user_id, MAX(time) AS last_buy_time_str
        FROM user_behavior
        WHERE behavior_type = '4'
        GROUP BY user_id;
                ''')
    rows = cursor.fetchall()
    if not rows:
        print("No data found.")
    else:
        print(f"Found {len(rows)} users with buy behavior.")
        results_with_recency = []
        for user in rows:
            user_id = user[0]
            last_buy_time_str = user[1]
            last_buy_time_dt = datetime.strptime(last_buy_time_str, '%Y-%m-%d %H')
            delta_seconds = (max_time_dt - last_buy_time_dt).total_seconds()
            buy_recency = delta_seconds / (24 * 60 * 60)
            results_with_recency.append((user_id, buy_recency))
            sample_size = min(3, len(results_with_recency))
            if sample_size > 0:
                sample_users = random.sample(results_with_recency, sample_size)
                print("Randomly selected users for checking buy recency:")
                has_error = False
                for user in sample_users:
                    user_id = user[0]
                    calculated_recency = user[1]
                    print(f"Checking user_id: {user_id}")
                    cursor.execute('''
                        SELECT MAX(time) 
                        FROM user_behavior
                        WHERE user_id = ? 
                        AND behavior_type = '4'
                    ''', (user_id,))
                    check_row = cursor.fetchone()
                    if check_row and check_row[0]:
                        verify_time_str = check_row[0]
                        verify_time_dt = datetime.strptime(verify_time_str, '%Y-%m-%d %H')
                        actual_delta_seconds = (max_time_dt - verify_time_dt).total_seconds()
                        actual_recency = actual_delta_seconds / (24 * 60 * 60)
                        if abs(calculated_recency - actual_recency) >= 1e-6:
                            has_error = True
                            print(f"[ERROR]User: Failed")
                            print(f"Calculated buy recency: {calculated_recency}")
                            print(f"Actual buy recency: {actual_recency}")
                    else:
                            has_error = True
                            print(f"[ERROR] User{user_id} NO record found")
                if not has_error:
                        print("All passed. No errors found.")
            else:
                print("No enough users for checking buy recency.")

# Tendency index
cursor.execute('''
SELECT
    user_id,
    (
        SUM(CASE WHEN behavior_type = '4' THEN 1 ELSE 0 END) +
        SUM(CASE WHEN behavior_type = '3' THEN 1 ELSE 0 END)
    ) * 1.0 /
    SUM(CASE WHEN behavior_type = '1' THEN 1 ELSE 0 END)
    AS tendency_index
FROM user_behavior
GROUP BY user_id;
                ''')
# Check the result for tendency index
rows = cursor.fetchall()

if not rows:
    print("No data found for Tendency Index.")
else:    
    print("Feature rows (Tendency Index): ")
    print(rows[:10])
    
    # Randomly check three users
    sample_size = min(3, len(rows))
    if sample_size > 0:
        sample_users = random.sample(rows, sample_size)
        
        print("Randomly selected users for checking:")
        
        for user in sample_users:
            user_id = user[0]
            calculated_index = user[1]  # SQL 算出来的指数
            
            print(f"Checking user_id: {user_id}")
            
            # 验证查询：分别查出 buy, cart, view 的数量
            cursor.execute('''
                SELECT 
                    SUM(CASE WHEN behavior_type = '4' THEN 1 ELSE 0 END) AS total_buy,
                    SUM(CASE WHEN behavior_type = '3' THEN 1 ELSE 0 END) AS total_cart,
                    SUM(CASE WHEN behavior_type = '1' THEN 1 ELSE 0 END) AS total_view
                FROM user_behavior
                WHERE user_id = ?
            ''', (user_id,))
            
            result_row = cursor.fetchone()
            
            if result_row:
                total_buy = result_row[0] or 0
                total_cart = result_row[1] or 0
                total_view = result_row[2] or 0
                
                # 在 Python 中手动计算指数
                if total_view > 0:
                    actual_index = (total_buy + total_cart) * 1.0 / total_view
                    # 【修复点】先格式化好字符串
                    actual_index_str = f"{actual_index:.4f}"
                else:
                    actual_index = None
                    actual_index_str = "N/A"  # 如果除数为0，显示 N/A
                
                # 同样处理 calculated_index 的显示
                calculated_index_str = f"{calculated_index:.4f}" if calculated_index is not None else "N/A"

                print(f"Buy: {total_buy}, Cart: {total_cart}, View: {total_view}")
                print(f"Calculated Index: {calculated_index_str}")
                print(f"Actual Index:     {actual_index_str}")
                
                # 检查是否匹配
                if actual_index is not None and calculated_index is not None:
                    if abs(calculated_index - actual_index) < 1e-6:
                        print("Passed")
                    else:
                        print("Failed")
                elif actual_index is None and calculated_index is None:
                    print("Passed (Both are None/Zero division)")
                else:
                    print("Failed (Mismatch in None/Value)")
            else:
                print("Failed: No data found for verification.")
    else:
        print("Not enough data to sample.")

# View not buy rate
cursor.execute('''
SELECT
    a.user_id,
    a.view_not_buy_count * 1.0 / b.total_views AS view_not_buy_rate
FROM
(
SELECT
    ub.user_id,
    COUNT(*) AS view_not_buy_count
FROM user_behavior ub
WHERE ub.behavior_type = '1'
AND NOT EXISTS (
    SELECT 1
    FROM user_behavior b
    WHERE ub.user_id = b.user_id
    AND ub.item_id = b.item_id
    AND b.behavior_type = '4'
               )
    GROUP BY ub.user_id
) a
JOIN
(
    SELECT
        user_id,
        COUNT(*) AS total_views
    FROM user_behavior
    WHERE behavior_type = 1
    GROUP BY user_id
) b
ON a.user_id = b.user_id;
            ''')
# Check the result for view_not_buy_rate
rows = cursor.fetchall()

if not rows:
    print("No data found for View Not Buy Rate.")
else:    
    print("Feature rows (View Not Buy Rate): ")
    print(rows[:10])
    
    # Randomly check three users
    sample_size = min(3, len(rows))
    if sample_size > 0:
        sample_users = random.sample(rows, sample_size)
        
        print("Randomly selected users for checking:")
        
        for user in sample_users:
            user_id = user[0]
            calculated_rate = user[1]  # SQL 聚合出来的比率
            
            print(f"Checking user_id: {user_id}")
            
            # 验证逻辑：
            # 1. 查出该用户所有的购买记录 (type 4) 对应的 item_id，存入集合
            cursor.execute('''
                SELECT DISTINCT item_id 
                FROM user_behavior 
                WHERE user_id = ? AND behavior_type = '4'
            ''', (user_id,))
            bought_items = set([row[0] for row in cursor.fetchall()])
            
            # 2. 查出该用户所有的浏览记录 (type 1)
            cursor.execute('''
                SELECT item_id 
                FROM user_behavior 
                WHERE user_id = ? AND behavior_type = '1'
            ''', (user_id,))
            view_rows = cursor.fetchall()
            
            total_views_count = len(view_rows)
            view_not_buy_count = 0
            
            # 3. 计算 view_not_buy_count
            for row in view_rows:
                item_id = row[0]
                if item_id not in bought_items:
                    view_not_buy_count += 1
            
            # 4. 在 Python 中手动计算比率
            if total_views_count > 0:
                actual_rate = view_not_buy_count * 1.0 / total_views_count
                # 【修复点】先格式化好字符串
                actual_rate_str = f"{actual_rate:.4f}"
            else:
                actual_rate = None
                actual_rate_str = "N/A"  # 如果除数为0，显示 N/A
            
            # 同样处理 calculated_rate 的显示
            calculated_rate_str = f"{calculated_rate:.4f}" if calculated_rate is not None else "N/A"

            print(f"Total Views: {total_views_count}")
            print(f"View Not Buy Count: {view_not_buy_count}")
            print(f"Calculated Rate: {calculated_rate_str}")
            print(f"Actual Rate:     {actual_rate_str}")
            
            # 检查是否匹配
            if actual_rate is not None and calculated_rate is not None:
                if abs(calculated_rate - actual_rate) < 1e-6:
                    print("Passed")
                else:
                    print("Failed")
            elif actual_rate is None and calculated_rate is None:
                print("Passed (Both are None/Zero division)")
            else:
                print("Failed (Mismatch in None/Value)")
    else:
        print("Not enough data to sample.")

# Cart not buy rate
cursor.execute('''
SELECT
    a.user_id,
    a.cart_not_buy_count * 1.0 / b.total_cart AS cart_not_buy_rate
FROM
(
SELECT
    ub.user_id,
    COUNT(*) AS cart_not_buy_count
FROM user_behavior ub
WHERE ub.behavior_type = '3'
AND NOT EXISTS (
    SELECT 1
    FROM user_behavior b
    WHERE ub.user_id = b.user_id
    AND ub.item_id = b.item_id
    AND b.behavior_type = '4'
               )
    GROUP BY ub.user_id
) a
JOIN
(
    SELECT
        user_id,
        COUNT(*) AS total_cart
    FROM user_behavior
    WHERE behavior_type = 3
    GROUP BY user_id
) b
ON a.user_id = b.user_id;
            ''')
# Check the result for cart_not_buy_rate
rows = cursor.fetchall()
if not rows:
    print("No data found for Cart Not Buy Rate.")
else:    
    print("Feature rows (Cart Not Buy Rate): ")
    print(rows[:10])
    # Randomly check three users
    sample_size = min(3, len(rows))
    if sample_size > 0:
        sample_users = random.sample(rows, sample_size)
        
        print("Randomly selected users for checking:")
        
        for user in sample_users:
            user_id = user[0]
            calculated_rate = user[1]  # SQL 算出来的比率
            
            print(f"Checking user_id: {user_id}")
            
            # 验证逻辑：
            # 1. 查出该用户所有购买过的 item_id (type 4)
            cursor.execute('''
                SELECT DISTINCT item_id 
                FROM user_behavior 
                WHERE user_id = ? AND behavior_type = '4'
            ''', (user_id,))
            bought_items = set([row[0] for row in cursor.fetchall()])
            
            # 2. 查出该用户所有的加购记录 (type 3)
            cursor.execute('''
                SELECT item_id 
                FROM user_behavior 
                WHERE user_id = ? AND behavior_type = '3'
            ''', (user_id,))
            cart_rows = cursor.fetchall()
            
            total_cart_count = len(cart_rows)
            cart_not_buy_count = 0
            
            # 3. 统计加购但未购买的次数
            for row in cart_rows:
                item_id = row[0]
                if item_id not in bought_items:
                    cart_not_buy_count += 1
            
            # 4. 在 Python 中手动计算比率
            if total_cart_count > 0:
                actual_rate = cart_not_buy_count * 1.0 / total_cart_count
                actual_rate_str = f"{actual_rate:.4f}"
            else:
                actual_rate = None
                actual_rate_str = "N/A"
            
            # 格式化 calculated_rate
            calculated_rate_str = f"{calculated_rate:.4f}" if calculated_rate is not None else "N/A"

            print(f"Total Cart Count: {total_cart_count}")
            print(f"Cart Not Buy Count: {cart_not_buy_count}")
            print(f"Calculated Rate: {calculated_rate_str}")
            print(f"Actual Rate:     {actual_rate_str}")
            
            # 检查是否匹配
            if actual_rate is not None and calculated_rate is not None:
                if abs(calculated_rate - actual_rate) < 1e-6:
                    print("Passed")
                else:
                    print("Failed")
            elif actual_rate is None and calculated_rate is None:
                print("Passed (Both are None)")
            else:
                print("Failed (Mismatch in None/Value)")
    else:
        print("Not enough data to sample.")

# Fav not buy rate
cursor.execute('''
SELECT
    a.user_id,
    a.fav_not_buy_count * 1.0 / b.total_fav AS fav_not_buy_rate
FROM
(   SELECT
    ub.user_id,
    COUNT(*) AS fav_not_buy_count
FROM user_behavior ub
WHERE ub.behavior_type = '2'
AND NOT EXISTS (
    SELECT 1
    FROM user_behavior b
    WHERE ub.user_id = b.user_id
    AND ub.item_id = b.item_id
    AND b.behavior_type = '4'
               )
    GROUP BY ub.user_id
) a
JOIN
(
    SELECT
        user_id,
        COUNT(*) AS total_fav
    FROM user_behavior
    WHERE behavior_type = '2'
    GROUP BY user_id
) b
ON a.user_id = b.user_id;
            ''')
# Check the result for fav_not_buy_rate
rows = cursor.fetchall()

if not rows:
    print("No data found for Fav Not Buy Rate.")
else:    
    print("Feature rows (Fav Not Buy Rate): ")
    print(rows[:10])
    
    # Randomly check three users
    sample_size = min(3, len(rows))
    if sample_size > 0:
        sample_users = random.sample(rows, sample_size)
        
        print("Randomly selected users for checking:")
        
        for user in sample_users:
            user_id = user[0]
            calculated_rate = user[1]  # SQL 算出来的比率
            
            print(f"Checking user_id: {user_id}")
            
            # 验证逻辑：
            # 1. 查出该用户所有购买过的 item_id (type 4)
            cursor.execute('''
                SELECT DISTINCT item_id 
                FROM user_behavior 
                WHERE user_id = ? AND behavior_type = '4'
            ''', (user_id,))
            bought_items = set([row[0] for row in cursor.fetchall()])
            
            # 2. 查出该用户所有的收藏记录 (type 2)
            cursor.execute('''
                SELECT item_id 
                FROM user_behavior 
                WHERE user_id = ? AND behavior_type = '2'
            ''', (user_id,))
            fav_rows = cursor.fetchall()
            
            total_fav_count = len(fav_rows)
            fav_not_buy_count = 0
            
            # 3. 统计收藏但未购买的次数
            for row in fav_rows:
                item_id = row[0]
                if item_id not in bought_items:
                    fav_not_buy_count += 1
            
            # 4. 在 Python 中手动计算比率
            if total_fav_count > 0:
                actual_rate = fav_not_buy_count * 1.0 / total_fav_count
                actual_rate_str = f"{actual_rate:.4f}"
            else:
                actual_rate = None
                actual_rate_str = "N/A"
            
            # 格式化 calculated_rate
            calculated_rate_str = f"{calculated_rate:.4f}" if calculated_rate is not None else "N/A"

            print(f"Total Fav Count: {total_fav_count}")
            print(f"Fav Not Buy Count: {fav_not_buy_count}")
            print(f"Calculated Rate: {calculated_rate_str}")
            print(f"Actual Rate:     {actual_rate_str}")
            
            # 检查是否匹配
            if actual_rate is not None and calculated_rate is not None:
                if abs(calculated_rate - actual_rate) < 1e-6:
                    print("Passed")
                else:
                    print("Failed")
            elif actual_rate is None and calculated_rate is None:
                print("Passed (Both are None)")
            else:
                print("Failed (Mismatch in None/Value)")
    else:
        print("Not enough data to sample.")

# Impulsive buying rate
cursor.execute('''
SELECT
    user_id,
    SUM(CASE WHEN behavior_type = '4' THEN 1 ELSE 0 END) * 1.0 /
    SUM(CASE WHEN behavior_type = '1' THEN 1 ELSE 0 END) AS impulsive_buying_rate
FROM user_behavior
GROUP BY user_id;
                ''')
# Check the result for impulsive_buying_rate
rows = cursor.fetchall()

if not rows:
    print("No data found for Impulsive Buying Rate.")
else:    
    print("Feature rows (Impulsive Buying Rate): ")
    print(rows[:10])
    
    # Randomly check three users
    sample_size = min(3, len(rows))
    if sample_size > 0:
        sample_users = random.sample(rows, sample_size)
        
        print("Randomly selected users for checking:")
        
        for user in sample_users:
            user_id = user[0]
            calculated_rate = user[1]  # SQL 算出来的比率
            
            print(f"Checking user_id: {user_id}")
            
            # 验证逻辑：分别查出该用户的 buy (4) 和 view (1) 数量
            cursor.execute('''
                SELECT 
                    SUM(CASE WHEN behavior_type = '4' THEN 1 ELSE 0 END) AS total_buy,
                    SUM(CASE WHEN behavior_type = '1' THEN 1 ELSE 0 END) AS total_view
                FROM user_behavior
                WHERE user_id = ?
            ''', (user_id,))
            
            result_row = cursor.fetchone()
            
            if result_row:
                total_buy = result_row[0] or 0
                total_view = result_row[1] or 0
                
                # 在 Python 中手动计算比率
                if total_view > 0:
                    actual_rate = total_buy * 1.0 / total_view
                    actual_rate_str = f"{actual_rate:.4f}"
                else:
                    actual_rate = None
                    actual_rate_str = "N/A"
                
                # 格式化 calculated_rate
                calculated_rate_str = f"{calculated_rate:.4f}" if calculated_rate is not None else "N/A"

                print(f"Total Buy: {total_buy}, Total View: {total_view}")
                print(f"Calculated Rate: {calculated_rate_str}")
                print(f"Actual Rate:     {actual_rate_str}")
                
                # 检查是否匹配
                if actual_rate is not None and calculated_rate is not None:
                    if abs(calculated_rate - actual_rate) < 1e-6:
                        print("Passed")
                    else:
                        print("Failed")
                elif actual_rate is None and calculated_rate is None:
                    print("Passed (Both are None)")
                else:
                    print("Failed (Mismatch in None/Value)")
            else:
                print("Failed: No data found for verification.")
    else:
        print("Not enough data to sample.")

# Hesitation buying rate
cursor.execute('''
SELECT
    user_id,
    SUM(CASE WHEN behavior_type = '1' THEN 1 ELSE 0 END) * 1.0 /
    SUM(CASE WHEN behavior_type = '4' THEN 1 ELSE 0 END) AS hesitation_buying_rate
FROM user_behavior
GROUP BY user_id;
                ''')
# Check the result for hesitation_buying_rate
rows = cursor.fetchall()

if not rows:
    print("No data found for Hesitation Buying Rate.")
else:    
    print("Feature rows (Hesitation Buying Rate): ")
    print(rows[:10])
    
    # 随机抽取最多3个用户进行校验
    sample_size = min(3, len(rows))
    sample_users = random.sample(rows, sample_size)
    
    print("Randomly selected users for checking Hesitation Buying Rate:")
    
    for user in sample_users:
        user_id = user[0]
        calculated_rate = user[1]  # SQL 计算出的犹豫购买率
        
        print(f"Checking user_id: {user_id}")
        
        # 手动验证：分别统计该用户的浏览数（behavior_type='1'）和购买数（behavior_type='4'）
        cursor.execute('''
            SELECT 
                SUM(CASE WHEN behavior_type = '1' THEN 1 ELSE 0 END) AS total_view,
                SUM(CASE WHEN behavior_type = '4' THEN 1 ELSE 0 END) AS total_buy
            FROM user_behavior
            WHERE user_id = ?
        ''', (user_id,))
        
        result_row = cursor.fetchone()
        
        if result_row:
            total_view = result_row[0] or 0
            total_buy = result_row[1] or 0
            
            # 注意：hesitation_buying_rate = 浏览数 / 购买数，若未购买则为 None 或无穷（此处设为 None）
            if total_buy > 0:
                actual_rate = total_view * 1.0 / total_buy
                actual_rate_str = f"{actual_rate:.4f}"
            else:
                actual_rate = None
                actual_rate_str = "N/A"
            
            calculated_rate_str = f"{calculated_rate:.4f}" if calculated_rate is not None else "N/A"

            print(f"Total View: {total_view}, Total Buy: {total_buy}")
            print(f"Calculated Rate: {calculated_rate_str}")
            print(f"Actual Rate:     {actual_rate_str}")
            
            # 比较结果
            if actual_rate is not None and calculated_rate is not None:
                if abs(calculated_rate - actual_rate) < 1e-6:
                    print("Passed")
                else:
                    print("Failed")
            elif actual_rate is None and calculated_rate is None:
                print("Passed (Both undefined due to no purchases)")
            else:
                print("Failed (Mismatch in defined/undefined status)")
        else:
            print("Failed: No behavior data found for this user.")

# Interest consentration
cursor.execute('''
SELECT
    a.user_id,
    CAST (a.max_category_count AS FLOAT) / b.total_category_count AS interest_concentration
FROM (
    SELECT
        user_id,
        MAX(category_count) AS max_category_count
    FROM (
        SELECT
            user_id,
            item_category,
            COUNT(*) AS category_count
        FROM user_behavior
        WHERE behavior_type = '1'
        GROUP BY user_id, item_category
    )
    Group By user_id
) a
JOIN (
    SELECT
        user_id,
        COUNT(*) AS total_category_count
    FROM user_behavior
    GROUP BY user_id
) b
ON a.user_id = b.user_id
GROUP BY a.user_id;
                ''')
# Check the result for interest concentration
rows = cursor.fetchall()

if not rows:
    print("No data found for Interest Concentration.")
else:
    print("Feature rows (Interest Concentration): ")
    print(rows[:10])
    
    # 随机抽取最多3个用户
    sample_size = min(3, len(rows))
    sample_users = random.sample(rows, sample_size)
    
    print("Randomly selected users for checking Interest Concentration:")
    
    for user in sample_users:
        user_id = user[0]
        calculated_concentration = user[1]  # SQL 计算出的兴趣集中度
        
        print(f"Checking user_id: {user_id}")
        
        # 步骤1: 获取该用户每个浏览类目（behavior_type='1'）的行为次数
        cursor.execute('''
            SELECT item_category, COUNT(*) AS category_count
            FROM user_behavior
            WHERE user_id = ? AND behavior_type = '1'
            GROUP BY item_category
            ORDER BY category_count DESC
            LIMIT 1
        ''', (user_id,))
        category_counts = cursor.fetchone()
        
        if not category_counts:
            print("No view behavior found for this user. Skipping.")
            continue
        
        # 步骤2: 手动计算 max_category_count 和 total_category_count
        max_category_count = category_counts[1]  # 最大类目行为数（已按降序）
       # 注意：这里改成与特征工程一致
        # 分母 = 所有行为数
        cursor.execute('''
            SELECT COUNT(*)
            FROM user_behavior
            WHERE user_id = ?
        ''', (user_id,))

        total_behavior_count = cursor.fetchone()[0]

        actual_concentration = (
            max_category_count / total_behavior_count
        )

        print(f"Max Category Count: {max_category_count}")
        print(f"Total Behavior Count: {total_behavior_count}")
        print(f"Calculated Concentration: {calculated_concentration:.4f}")
        print(f"Actual Concentration:     {actual_concentration:.4f}")

        if abs(calculated_concentration - actual_concentration) < 1e-6:
            print("Passed")
        else:
            print("Failed")

# Active hour
cursor.execute('''
SELECT
    a.user_id,
    a.active_hour
FROM(
    SELECT
        user_id,
        SUBSTR(time, 12, 2) AS active_hour,
        COUNT(*) AS behavior_count
    FROM user_behavior
    GROUP BY user_id, active_hour
) a
JOIN(
    SELECT
        user_id,
        MAX(behavior_count) AS max_count
    FROM(
        SELECT
            user_id,
            SUBSTR(time, 12, 2) AS active_hour,
            COUNT(*) AS behavior_count
        FROM user_behavior
        GROUP BY user_id, active_hour
    )
    GROUP BY user_id
) b
ON a.user_id = b.user_id
AND a.behavior_count = b.max_count;
            ''')
# Check the result for active_hour
rows = cursor.fetchall()

if not rows:
    print("No data found for Active Hour.")
else:    
    print("Feature rows (Active Hour): ")
    print(rows[:10])
    
    # Randomly check three users
    sample_size = min(3, len(rows))
    if sample_size > 0:
        sample_users = random.sample(rows, sample_size)
        
        print("Randomly selected users for checking:")
        
        for user in sample_users:
            user_id = user[0]
            calculated_hour = user[1]  # SQL 算出来的最活跃小时 (字符串 '00'-'23')
            
            print(f"Checking user_id: {user_id}")
            
            # 验证逻辑：
            # 1. 查出该用户每个小时的行为次数
            cursor.execute('''
                SELECT SUBSTR(time, 12, 2) AS hour, COUNT(*) as cnt
                FROM user_behavior
                WHERE user_id = ?
                GROUP BY hour
                ORDER BY cnt DESC
            ''', (user_id,))
            
            hour_rows = cursor.fetchall()
            
            if not hour_rows:
                print("Failed: No behavior records found.")
                continue
            
            # 2. 找到最大的行为次数
            max_count = hour_rows[0][1]
            
            # 3. 找出所有达到最大次数的“最活跃小时”集合
            # (因为可能有多个小时并列第一)
            most_active_hours_set = set()
            for row in hour_rows:
                if row[1] == max_count:
                    most_active_hours_set.add(row[0])
                else:
                    break # 因为已经按降序排列，后面的肯定更小
            
            print(f"Max Behavior Count: {max_count}")
            print(f"Most Active Hours (Tied): {most_active_hours_set}")
            print(f"SQL Returned Hour: {calculated_hour}")
            
            # 4. 检查 SQL 返回的小时是否在“最活跃小时集合”中
            if calculated_hour in most_active_hours_set:
                print("Passed")
            else:
                print("Failed")
                
                # 打印详细对比以便调试
                print("Hourly Breakdown:")
                for row in hour_rows[:5]: # 只打印前5个
                    print(f"  {row[0]}: {row[1]}")

# Moring active rate
cursor.execute('''
               SELECT
    user_id,
    SUM(CASE WHEN CAST(SUBSTR(time,12,2) AS INTEGER) BETWEEN 6 AND 13 THEN 1 ELSE 0 END) * 1.0 /
    COUNT(*) AS morning_active_rate
FROM user_behavior
GROUP BY user_id;
                ''')
# Check the result for morning active rate
# Check the result for morning_active_rate
rows = cursor.fetchall()

if not rows:
    print("No data found for Morning Active Rate.")
else:
    print("Feature rows (Morning Active Rate): ")
    print(rows[:10])
    
    # 随机抽取最多3个用户进行验证
    sample_size = min(3, len(rows))
    sample_users = random.sample(rows, sample_size)
    
    print("Randomly selected users for checking Morning Active Rate:")
    
    for user in sample_users:
        user_id = user[0]
        calculated_rate = user[1]  # SQL 计算出的早晨活跃率
        
        print(f"Checking user_id: {user_id}")
        
        # 手动验证：统计该用户总行为数和早晨（06-14点）行为数
        cursor.execute('''
            SELECT 
                COUNT(*) AS total_count,
                SUM(CASE WHEN CAST(SUBSTR(time,12,2) AS INTEGER) BETWEEN 6 AND 13 THEN 1 ELSE 0 END) AS morning_count
            FROM user_behavior
            WHERE user_id = ?
        ''', (user_id,))
        
        result_row = cursor.fetchone()
        
        if result_row:
            total_count = result_row[0] or 0
            morning_count = result_row[1] or 0
            
            # 手动计算早晨活跃率
            if total_count > 0:
                actual_rate = morning_count * 1.0 / total_count
                actual_rate_str = f"{actual_rate:.4f}"
            else:
                actual_rate = None
                actual_rate_str = "N/A"
            
            calculated_rate_str = f"{calculated_rate:.4f}" if calculated_rate is not None else "N/A"
            
            print(f"Total Behaviors: {total_count}")
            print(f"Morning Behaviors (06-14): {morning_count}")
            print(f"Calculated Rate: {calculated_rate_str}")
            print(f"Actual Rate:     {actual_rate_str}")
            
            # 比对结果
            if actual_rate is not None and calculated_rate is not None:
                if abs(calculated_rate - actual_rate) < 1e-6:
                    print("Passed")
                else:
                    print("Failed")
            elif actual_rate is None and calculated_rate is None:
                print("Passed (Both undefined)")
            else:
                print("Failed (Mismatch in defined/undefined status)")
        else:
            print("Failed: No behavior data found for this user.")

# Afternoon active rate
cursor.execute('''
SELECT
    user_id,
    SUM(CASE WHEN CAST(SUBSTR(time,12,2) AS INTEGER) BETWEEN 14 AND 21 THEN 1 ELSE 0 END) * 1.0 /
    COUNT(*) AS afternoon_active_rate
FROM user_behavior
GROUP BY user_id;
                ''')
# Check the result for afternoon active rate
rows = cursor.fetchall()

if not rows:
    print("No data found for Afternoon Active Rate.")
else:
    print("Feature rows (Afternoon Active Rate): ")
    print(rows[:10])
    
    # 随机抽取最多3个用户进行验证
    sample_size = min(3, len(rows))
    sample_users = random.sample(rows, sample_size)
    
    print("Randomly selected users for checking Afternoon Active Rate:")
    
    for user in sample_users:
        user_id = user[0]
        calculated_rate = user[1]  # SQL 计算出的中午活跃率
        
        print(f"Checking user_id: {user_id}")
        
        # 手动验证：统计该用户总行为数和中午（14-22点）行为数
        cursor.execute('''
            SELECT 
                COUNT(*) AS total_count,
                SUM(CASE WHEN CAST(SUBSTR(time,12,2) AS INTEGER) BETWEEN 14 AND 21 THEN 1 ELSE 0 END) AS afternoon_count
            FROM user_behavior
            WHERE user_id = ?
        ''', (user_id,))
        
        result_row = cursor.fetchone()
        
        if result_row:
            total_count = result_row[0] or 0
            afternoon_count = result_row[1] or 0
            
            # 手动计算中午活跃率
            if total_count > 0:
                actual_rate = afternoon_count * 1.0 / total_count
                actual_rate_str = f"{actual_rate:.4f}"
            else:
                actual_rate = None
                actual_rate_str = "N/A"
            
            calculated_rate_str = f"{calculated_rate:.4f}" if calculated_rate is not None else "N/A"
            
            print(f"Total Behaviors: {total_count}")
            print(f"Afternoon Behaviors (14-22): {afternoon_count}")
            print(f"Calculated Rate: {calculated_rate_str}")
            print(f"Actual Rate:     {actual_rate_str}")
            
            # 比对结果
            if actual_rate is not None and calculated_rate is not None:
                if abs(calculated_rate - actual_rate) < 1e-6:
                    print("Passed")
                else:
                    print("Failed")
            elif actual_rate is None and calculated_rate is None:
                print("Passed (Both undefined)")
            else:
                print("Failed (Mismatch in defined/undefined status)")
        else:
            print("Failed: No behavior data found for this user.")

# Night active rate
cursor.execute('''
SELECT
    user_id,
    SUM(CASE WHEN CAST(SUBSTR(time,12,2) AS INTEGER) >= 22
    OR CAST(SUBSTR(time,12,2) AS INTEGER) < 6 THEN 1 ELSE 0 END) * 1.0 /
    COUNT(*) AS night_active_rate
FROM user_behavior
GROUP BY user_id;
                ''')
# Check the result for night active rate
rows = cursor.fetchall()
if not rows:
    print("No data found for Night Active Rate.")
else:
    print("Feature rows (Night Active Rate): ")
    print(rows[:10])
    
    # 随机抽取最多3个用户进行验证
    sample_size = min(3, len(rows))
    sample_users = random.sample(rows, sample_size)
    
    print("Randomly selected users for checking Night Active Rate:")
    
    for user in sample_users:
        user_id = user[0]
        calculated_rate = user[1]  # SQL 计算出的夜间活跃率
        
        print(f"Checking user_id: {user_id}")
        
        # 手动验证：统计该用户总行为数和夜间（22-06点）行为数
        cursor.execute('''
            SELECT 
                COUNT(*) AS total_count,
                SUM(CASE WHEN CAST(SUBSTR(time,12,2) AS INTEGER) >= 22
                OR CAST(SUBSTR(time,12,2) AS INTEGER) < 6 THEN 1 ELSE 0 END) AS night_count
            FROM user_behavior
            WHERE user_id = ?
        ''', (user_id,))
        
        result_row = cursor.fetchone()
        
        if result_row:
            total_count = result_row[0] or 0
            night_count = result_row[1] or 0
            
            # 手动计算夜间活跃率
            if total_count > 0:
                actual_rate = night_count * 1.0 / total_count
                actual_rate_str = f"{actual_rate:.4f}"
            else:
                actual_rate = None
                actual_rate_str = "N/A"
            
            calculated_rate_str = f"{calculated_rate:.4f}" if calculated_rate is not None else "N/A"
            
            print(f"Total Behaviors: {total_count}")
            print(f"Night Behaviors (22-06): {night_count}")
            print(f"Calculated Rate: {calculated_rate_str}")
            print(f"Actual Rate:     {actual_rate_str}")
            
            # 比对结果
            if actual_rate is not None and calculated_rate is not None:
                if abs(calculated_rate - actual_rate) < 1e-6:
                    print("Passed")
                else:
                    print("Failed")
            elif actual_rate is None and calculated_rate is None:
                print("Passed (Both undefined)")
            else:
                print("Failed (Mismatch in defined/undefined status)")
        else:
            print("Failed: No behavior data found for this user.")

conn.commit()
conn.close()