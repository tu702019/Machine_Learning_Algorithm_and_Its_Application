# 數據預處理報告

## 摘要

本報告描述了對兩個數據集（bankruptcy和diamonds）進行的數據預處理步驟。預處理包括處理缺失值、數據平衡和特徵選擇，旨在為後續機器學習模型提供高質量的輸入數據。

## 1. 數據集描述

### Bankruptcy數據集
- **目標**：預測公司是否會破產（分類問題）
- **規模**：6,819筆資料，97個特徵
- **缺失值**：共9,440個缺失值，分布在95個特徵中

### Diamonds數據集
- **目標**：預測鑽石價格（回歸問題）
- **規模**：53,940筆資料，12個特徵
- **缺失值**：共9,924個缺失值，分布在10個特徵中

## 2. 預處理方法與實現

### 2.1 缺失值處理

處理缺失值是數據預處理中的重要步驟，不適當的處理可能導致模型性能下降或產生偏差。

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

### 2.2 數據平衡（僅適用於Bankruptcy數據集）

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

### 2.3 特徵選擇

特徵選擇有助於降低模型複雜度、提高計算效率，並可能提高模型性能。

#### 採用的方法：
- **Bankruptcy數據集**：使用ANOVA F-value（F檢驗）選擇特徵
- **Diamonds數據集**：使用F回歸統計量選擇特徵

#### 選擇理由：
1. **適合問題類型**：F檢驗適合分類問題，F回歸適合回歸問題
2. **統計顯著性**：基於統計顯著性評估特徵與目標的關係
3. **優於簡單相關係數**：考慮更複雜的統計關係，而非僅線性關係
4. **計算效率高**：與包裝器方法（如遞歸特徵消除）相比，計算成本較低

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

### 2.4 預測能力評估

為了驗證特徵選擇的有效性，我們比較了使用全部特徵和選擇後特徵的模型性能。

#### 評估方法：
- 使用RandomForest算法
- 採用5折交叉驗證
- 測試集評估（30%數據）

#### 評估結果：

**Bankruptcy數據集：**
- 使用全部特徵的平均準確率：0.9742 ± 0.0036（交叉驗證）
- 使用選擇後特徵的平均準確率：0.9651 ± 0.0032（交叉驗證）
- 測試集上全部特徵的準確率：0.9806
- 測試集上選擇後特徵的準確率：0.9710
- 準確率變化：-0.96個百分點
- 特徵數量減少了：48個（50.5%）

**Diamonds數據集：**
- 使用全部特徵的平均R²分數：0.9455 ± 0.0034（交叉驗證）
- 使用選擇後特徵的平均R²分數：0.8884 ± 0.0045（交叉驗證）
- 測試集上全部特徵的R²分數：0.9469
- 測試集上選擇後特徵的R²分數：0.8879
- R²分數變化：-5.91個百分點
- 特徵數量減少了：12個（52.2%）

特徵選擇雖然導致了輕微的性能下降，但顯著減少了特徵數量（約50%），這種權衡在實際應用中通常是可接受的。模型複雜度的降低有助於減少過擬合風險、提高訓練速度，並改善模型的可解釋性。特別是對於Bankruptcy數據集，準確率僅下降了不到1個百分點，同時特徵數量減少了一半，這是一個很好的權衡。

## 3. 結論

本預處理流程包括三個主要步驟：缺失值處理、數據平衡和特徵選擇，成功將原始數據轉化為更適合機器學習的格式。

### 主要成果：
1. 成功處理了兩個數據集中的所有缺失值
2. 解決了Bankruptcy數據集中的嚴重類別不平衡問題，將破產樣本比例從3.2%提高到50%
3. 通過特徵選擇減少了約50%的特徵，僅犧牲少量預測性能：
   - Bankruptcy數據集：準確率降低0.96個百分點，特徵減少50.5%
   - Diamonds數據集：R²分數降低5.91個百分點，特徵減少52.2%

### 特徵選擇的權衡分析
特徵選擇導致的輕微性能下降是可接受的，因為它帶來了以下好處：
1. **提高計算效率**：減少了約一半的特徵，可顯著加快模型訓練和推論速度
2. **減少過擬合風險**：較少的特徵意味著模型不太可能記住訓練數據中的噪聲
3. **提高模型可解釋性**：較少的特徵使模型更容易解釋和理解
4. **降低數據獲取和存儲成本**：在實際應用中，可以只收集和存儲必要的特徵

### 後續建議：
1. 可以探索更多缺失值處理方法（如基於模型的插補）
2. 考慮嘗試其他特徵選擇方法（如包裝器方法、嵌入式方法）
3. 可以進一步調整SMOTE參數，或嘗試其他平衡技術
4. 可以嘗試不同的特徵子集大小（如30%或40%的特徵），找到性能和複雜度之間的最佳平衡點

處理後的數據已保存為Excel和CSV格式，可用於後續建模和分析。
