# HW1_數據預處理報告

<p class="text-right">
財金碩一 313707011 涂銥娗
</p>

## 數據集描述

### Bankruptcy數據集
- **目標**：預測公司是否會破產（分類問題）
- **規模**：6,819筆資料，97個特徵
- **缺失值**：共9,440個缺失值，分布在95個特徵中

### Diamonds數據集
- **目標**：預測鑽石價格（回歸問題）
- **規模**：53,940筆資料，12個特徵
- **缺失值**：共9,924個缺失值，分布在10個特徵中

## 預處理方法與實現

### 1. 缺失值處理

#### 採用的方法：
- **數值型特徵**：使用中位數(median)填充
- **分類型特徵**：使用眾數(mode)填充

#### 選擇理由：
1. **中位數vs平均數**：中位數對異常值更加穩健，能避免極端值的影響
2. **眾數**：對於分類數據，眾數代表最常見的類別，是合理的估計值
3. **方法簡單有效**：相比複雜的插補方法（如KNN插補、多重插補），這種方法計算效率高且易於實現

#### 實現代碼：
```python
# 處理bankruptcy數據集的缺失值
for column in df_bankruptcy.columns:
    if df_bankruptcy[column].isnull().sum() > 0:
        if df_bankruptcy[column].dtype == 'object':  # 分類變數
            mode_value = df_bankruptcy[column].mode()[0]
            df_bankruptcy_filled[column] = df_bankruptcy[column].fillna(mode_value)
        else:  # 數值型變數
            median_value = df_bankruptcy[column].median()
            df_bankruptcy_filled[column] = df_bankruptcy[column].fillna(median_value)
```

#### 效果：
- Bankruptcy數據集：成功處理了9,440個缺失值
- Diamonds數據集：成功處理了9,924個缺失值

### 2. 數據平衡（僅適用於Bankruptcy數據集）

類別不平衡問題會導致模型偏向多數類，降低對少數類的預測能力，特別是在如破產預測這樣的領域，我們更關注少數類（破產案例）。

#### 原始類別分布：
- 非破產樣本(0)：6,599個（96.8%）
- 破產樣本(1)：220個（3.2%）

這顯示了嚴重的類別不平衡。

#### 採用的方法：SMOTE (Synthetic Minority Over-sampling Technique)

#### 選擇理由：
1. **優於簡單複製**：SMOTE生成合成樣本而非簡單複製，減少過擬合風險
2. **保留特徵空間結構**：在特徵空間中生成合成樣本，保留了變量間的關係
3. **提高模型對少數類的敏感度**：使模型能更好地學習少數類的模式

#### 實現代碼：
```python
from imblearn.over_sampling import SMOTE

# 準備特徵和標籤
X = df_bankruptcy_filled.drop(["Bankrupt?"], axis=1)
y = df_bankruptcy_filled["Bankrupt?"]

# 應用SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 創建平衡後的數據框
df_bankruptcy_balanced = pd.DataFrame(X_resampled, columns=X.columns)
df_bankruptcy_balanced["Bankrupt?"] = y_resampled
```

#### 效果：
- 平衡後類別分布：
  - 非破產樣本(0)：6,599個（50%）
  - 破產樣本(1)：6,599個（50%）

### 3. 特徵選擇

特徵選擇有助於降低模型複雜度、提高計算效率，並可能提高模型性能。

#### 採用的方法：過濾法（Filter Method）
本研究採用的是過濾法（Filter Method）進行特徵選擇。過濾法是一種獨立於後續機器學習算法的特徵選擇方法，它基於統計指標評估每個特徵與目標變數的關係強度，然後選擇得分最高的特徵。
- **Bankruptcy數據集**：使用ANOVA F-value（F檢驗）選擇特徵
- **Diamonds數據集**：使用F回歸統計量選擇特徵

#### 過濾法的特點：
- **定義**：過濾法是一種獨立於後續機器學習算法的特徵選擇方法，它基於統計指標評估每個特徵與目標變數的關係強度，然後選擇得分最高的特徵
- **優點**：計算效率高、易於實現、不依賴於特定的機器學習算法、減少過擬合風險
- **限制**：不考慮特徵間的相互作用、不考慮後續使用的模型特性

#### 選擇理由：
1. **適合問題類型**：F檢驗適合分類問題，F回歸適合回歸問題
2. **統計顯著性**：基於統計顯著性評估特徵與目標的關係
3. **優於簡單相關係數**：考慮更複雜的統計關係，而非僅線性關係
4. **計算效率高**：與包裝法（Wrapper）（如遞歸特徵消除）相比，計算成本較低

#### 實現代碼：
```python
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

# Bankruptcy數據集（分類問題）
k_features_bankruptcy = int(original_features_bankruptcy * 0.5)
selector = SelectKBest(f_classif, k=k_features_bankruptcy)
X_bankruptcy_selected = selector.fit_transform(X_bankruptcy, y_bankruptcy)

# Diamonds數據集（回歸問題）
k_features_diamonds = int(encoded_features_count * 0.5)
selector_diamonds = SelectKBest(f_regression, k=k_features_diamonds)
X_diamonds_selected = selector_diamonds.fit_transform(X_diamonds_encoded, y_diamonds)
```

#### 選擇前50%特徵的理由：
1. **平衡計算複雜度與模型性能**：減少特徵數量可降低計算成本和過擬合風險
2. **實驗性選擇**：50%是實踐中常用的起點，可在不大幅降低性能的情況下減少特徵數
3. **適用於高維數據**：對於Bankruptcy數據集（95個特徵），特徵選擇尤為重要

#### 效果：
- Bankruptcy數據集：從95個特徵減少到47個（減少50%）
- Diamonds數據集：從23個編碼後特徵減少到11個（減少52%）

