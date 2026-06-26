# 中间特征表说明（Intermediate Feature Tables）
本项目在原始用户行为数据基础上，分别从user、item、category三个维度构建中间特征表，用于后续特征拆分与模型训练。

## 1. middle_user.db
用户维度统计特征表，用于描述用户整体行为模式、活跃程度、购买倾向以及时间偏好。

### 主要包含：

| 字段名 | 含义 |
|---|---|
| total_behavior | 用户总行为次数 |
| total_views | 用户总浏览次数 |
| last_1day_pv | 最近1天浏览次数 |
| last_3day_pv | 最近3天浏览次数 |
| last_7day_pv | 最近7天浏览次数 |
| activity_growth_rate | 用户活跃度增长率 |
| total_cart | 加购次数 |
| cart_rate | 加购率 |
| total_fav | 收藏次数 |
| fav_rate | 收藏率 |
| total_buy | 购买次数 |
| buy_rate | 购买率 |
| first_active_time | 首次活跃时间 |
| last_active_time | 最近活跃时间 |
| buy_recency | 最近购买距离当前时间间隔 |
| tendency_index | 用户购买倾向指数 |
| view_not_buy_rate | 浏览未购买率 |
| cart_not_buy_rate | 加购未购买率 |
| fav_not_buy_rate | 收藏未购买率 |
| impulsive_buying_rate | 冲动购买率 |
| hesitation_buying_rate | 犹豫购买率 |
| interest_concentration | 兴趣集中度 |
| active_hour | 用户最活跃时间段 |
| morning_active_rate | 上午活跃占比 |
| afternoon_active_rate | 下午活跃占比 |
| night_active_rate | 夜间活跃占比 |

### 特征构造逻辑：

- 对用户历史行为记录进行聚合统计，得到用户总行为次数、总浏览次数、加购次数、收藏次数、购买次数、最活跃时间段等特征
- 通过时间窗口统计用户近期行为活跃程度
- 使用行为转化率刻画用户购买意愿
- 使用时间特征描述用户行为习惯

## 2. middle_item.db
商品维度统计特征表，用于描述商品热度、转化能力以及用户兴趣深度。

### 主要包含：

| 字段名 | 含义 |
|---|---|
| total_pv | 商品总浏览量 |
| last3day_pv | 最近3天浏览量 |
| last7day_pv | 最近7天浏览量 |
| popularity_growth_rate | 商品热度增长率 |
| items_pv_ranking | 商品浏览量排名 |
| cart_count | 商品加购次数 |
| favorite_count | 商品收藏次数 |
| buy_count | 商品购买次数 |
| cart_rate | 商品加购率 |
| favorite_rate | 商品收藏率 |
| buy_rate | 商品购买率 |
| uv | 商品独立访客数 |
| view_not_buy_rate | 浏览未购买率 |
| cart_not_buy_rate | 加购未购买率 |
| fav_not_buy_rate | 收藏未购买率 |
| interest_depth | 用户兴趣深度 |
| repurchase_index | 商品复购指数 |
| buy_recency_hours | 最近购买时间间隔（小时） |
| peak_view_hour | 浏览高峰时段 |
| peak_buy_hour | 购买高峰时段 |

### 特征构造逻辑：

- 对商品行为数据进行聚合统计，得到商品总浏览量、加购次数、收藏次数、购买次数、浏览未购买率、加购未购买率、收藏未购买率、用户兴趣深度、商品复购指数、最近购买时间间隔等特征
- 使用近期行为反映商品短期热度变化
- 使用转化率衡量商品吸引力
- 使用时间分布特征描述商品购买规律

## 3. middle_category.db
商品类别维度统计特征表，用于描述类别热度、转化能力以及用户兴趣深度。

### 主要包含：

| 字段名 | 含义 |
|---|---|
| total_pv | 类别总浏览量 |
| cart_count | 类别总加购次数 |
| favorite_count | 类别总收藏次数 |
| buy_count | 类别总购买次数 |
| users_count | 类别涉及用户数量 |
| buy_rate | 类别购买率 |
| peak_view_hour | 类别浏览高峰时间 |
| peak_buy_hour | 类别购买高峰时间 |

### 特征构造逻辑：
- 对类别行为数据进行聚合统计，得到类别总浏览量、总加购次数、总收藏次数、总购买次数、类别涉及用户数量、类别购买率、类别浏览高峰时间、类别购买高峰时间等特征
- 使用行为转化率刻画类别购买趋势
- 使用时间高峰特征反映类别行为规律

## 4. 特征表拼接

最终通过以下主键进行特征拼接：

| 特征表 | 拼接键 |
|---|---|
| user_feature | user_id |
| item_feature | item_id |
| category_feature | item_category |

最终形成的feature_table，用于后续：

- 数据集划分
- 特征筛选
- 模型训练评估与预测