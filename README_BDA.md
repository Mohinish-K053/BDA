# 📊 Data Science & Big Data Analytics — Lab Assignments

> All code runs on **Google Colab**. Each assignment is self-contained with its own install block, algorithm explanation, and step-by-step guide.

---

## 📋 Table of Contents

| # | Assignment | Topics |
|---|-----------|--------|
| [1](#assignment-1) | EDA — PDF, CDF, Univariate Analysis | NumPy, Pandas, Matplotlib, SciPy |
| [2](#assignment-2) | Distribution-Based Problems | NumPy, Pandas, Matplotlib |
| [3](#assignment-3) | Hypothesis Testing | SciPy — t-test, z-test, chi-square, ANOVA |
| [4](#assignment-4) | Data Wrangling + PySpark + SQL | IRIS Dataset, Book Dataset |
| [5](#assignment-5) | Linear Regression | Boston Housing, Scatter & KDE Plots |
| [6](#assignment-6) | Logistic Regression + K-Means | IRIS, PySpark, Confusion Matrix |
| [7](#assignment-7) | RDD Actions & Transformations | Apache Spark |
| [8](#assignment-8) | MapReduce | Hadoop & PySpark |

---

## 🚀 How to Use

1. Open [Google Colab](https://colab.research.google.com/)
2. Click **New Notebook**
3. Copy the code for the desired assignment (each section below)
4. Paste into a Colab cell and run (`Shift + Enter`)
5. The first cell in every assignment installs all required libraries

---

---

## Assignment 1

## 📈 EDA — PDF, CDF & Univariate Data Analysis

### 🧠 Algorithm
```
1. Generate or load dataset
2. Compute descriptive statistics (mean, median, std, skewness, kurtosis)
3. Plot Histogram with overlaid PDF using KDE
4. Plot CDF by sorting data and computing cumulative proportions
5. Visualize Boxplot and Violin plot for distribution shape
6. Interpret skewness and spread
```

### ▶️ How to Run
1. Copy the full code block below into a **Google Colab cell**
2. Run it — installs are at the top, everything is self-contained
3. All plots will render inline

---

```python
# ============================================================
# ASSIGNMENT 1: EDA — PDF, CDF, Univariate Analysis
# ============================================================

# ── INSTALL (run this cell first) ──────────────────────────
!pip install numpy pandas matplotlib scipy seaborn --quiet

# ── IMPORTS ────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

print("✅ Libraries loaded successfully")

# ── DATA GENERATION ────────────────────────────────────────
np.random.seed(42)
data = np.concatenate([
    np.random.normal(loc=50, scale=10, size=500),
    np.random.normal(loc=80, scale=5,  size=200)
])
df = pd.DataFrame({'value': data})

# ── DESCRIPTIVE STATISTICS ─────────────────────────────────
print("\n📊 Descriptive Statistics:")
print(df.describe().round(3))
print(f"  Skewness : {df['value'].skew():.4f}")
print(f"  Kurtosis : {df['value'].kurt():.4f}")

# ── FIGURE SETUP ───────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Univariate Analysis — PDF, CDF, Box & Violin', fontsize=16, fontweight='bold')

# 1. Histogram + PDF
ax = axes[0, 0]
ax.hist(data, bins=40, density=True, alpha=0.6, color='steelblue', label='Histogram')
kde = stats.gaussian_kde(data)
x_range = np.linspace(data.min(), data.max(), 300)
ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='PDF (KDE)')
ax.set_title('Histogram + Probability Density Function')
ax.set_xlabel('Value'); ax.set_ylabel('Density')
ax.legend()

# 2. CDF
ax = axes[0, 1]
sorted_data = np.sort(data)
cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
ax.plot(sorted_data, cdf, color='darkorange', linewidth=2)
ax.set_title('Cumulative Distribution Function (CDF)')
ax.set_xlabel('Value'); ax.set_ylabel('Cumulative Probability')
ax.grid(True, alpha=0.3)

# 3. Boxplot
ax = axes[1, 0]
ax.boxplot(data, vert=True, patch_artist=True,
           boxprops=dict(facecolor='lightcyan', color='navy'),
           medianprops=dict(color='red', linewidth=2))
ax.set_title('Box Plot')
ax.set_ylabel('Value')

# 4. Violin Plot
ax = axes[1, 1]
ax.violinplot(data, showmedians=True)
ax.set_title('Violin Plot')
ax.set_ylabel('Value')

plt.tight_layout()
plt.savefig('assignment1_eda.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

---

## Assignment 2

## 🎲 Distribution-Based Problem Statements

### 🧠 Algorithm
```
1. Define problem statements (e.g., modelling test scores, wait times)
2. Fit appropriate distributions — Normal, Binomial, Poisson, Exponential
3. Plot PMF/PDF for each distribution
4. Compute probabilities using distribution functions
5. Compare theoretical vs empirical distributions
```

### ▶️ How to Run
1. Copy the full code block below into a **Google Colab cell**
2. Run — outputs include printed probabilities and all plots

---

```python
# ============================================================
# ASSIGNMENT 2: Distribution-Based Problem Statements
# ============================================================

# ── INSTALL ────────────────────────────────────────────────
!pip install numpy pandas matplotlib scipy --quiet

# ── IMPORTS ────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

print("✅ Libraries loaded")

# ════════════════════════════════════════════════════════════
# PROBLEM 1: Student Exam Scores — Normal Distribution
# A class of 1000 students has mean score 65, std 12.
# What % scored above 80? Below 50?
# ════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("PROBLEM 1: Exam Scores — Normal Distribution")
print("="*55)

mu, sigma = 65, 12
dist_norm = stats.norm(mu, sigma)

p_above_80 = 1 - dist_norm.cdf(80)
p_below_50 = dist_norm.cdf(50)
p_between  = dist_norm.cdf(80) - dist_norm.cdf(50)

print(f"  Mean={mu}, Std={sigma}")
print(f"  P(score > 80)       = {p_above_80:.4f}  ({p_above_80*100:.2f}%)")
print(f"  P(score < 50)       = {p_below_50:.4f}  ({p_below_50*100:.2f}%)")
print(f"  P(50 < score < 80)  = {p_between:.4f}  ({p_between*100:.2f}%)")

# ════════════════════════════════════════════════════════════
# PROBLEM 2: Customer Arrivals — Poisson Distribution
# A shop gets avg 5 customers/hour. P(exactly 3)? P(>7)?
# ════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("PROBLEM 2: Customer Arrivals — Poisson Distribution")
print("="*55)

lam = 5
dist_poisson = stats.poisson(lam)

p_exactly_3 = dist_poisson.pmf(3)
p_more_than_7 = 1 - dist_poisson.cdf(7)

print(f"  Lambda (avg arrivals) = {lam}")
print(f"  P(exactly 3 arrivals) = {p_exactly_3:.4f}")
print(f"  P(more than 7)        = {p_more_than_7:.4f}")

# ── VISUALISATION ──────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Assignment 2: Distribution Analysis', fontsize=15, fontweight='bold')

# Normal PDF
ax = axes[0, 0]
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 300)
ax.plot(x, dist_norm.pdf(x), 'b-', linewidth=2, label='Normal PDF')
ax.fill_between(x, dist_norm.pdf(x), where=(x > 80), alpha=0.4, color='red', label='P(>80)')
ax.fill_between(x, dist_norm.pdf(x), where=(x < 50), alpha=0.4, color='green', label='P(<50)')
ax.axvline(mu, color='black', linestyle='--', label=f'Mean={mu}')
ax.set_title('Normal Distribution — Exam Scores')
ax.set_xlabel('Score'); ax.set_ylabel('Density')
ax.legend(fontsize=8)

# Normal CDF
ax = axes[0, 1]
ax.plot(x, dist_norm.cdf(x), 'b-', linewidth=2)
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
ax.set_title('Normal Distribution — CDF')
ax.set_xlabel('Score'); ax.set_ylabel('Cumulative Probability')
ax.grid(True, alpha=0.3)

# Poisson PMF
ax = axes[1, 0]
k_vals = np.arange(0, 16)
pmf_vals = dist_poisson.pmf(k_vals)
colors = ['red' if k == 3 else ('orange' if k > 7 else 'steelblue') for k in k_vals]
bars = ax.bar(k_vals, pmf_vals, color=colors, edgecolor='white')
ax.set_title('Poisson Distribution — Customer Arrivals\n(red=P(3), orange=P(>7))')
ax.set_xlabel('Number of Arrivals'); ax.set_ylabel('Probability')
ax.set_xticks(k_vals)

# Poisson CDF
ax = axes[1, 1]
cdf_vals = dist_poisson.cdf(k_vals)
ax.step(k_vals, cdf_vals, where='post', color='darkorange', linewidth=2)
ax.axhline(1 - p_more_than_7, color='red', linestyle='--', alpha=0.7, label='CDF at 7')
ax.set_title('Poisson Distribution — CDF')
ax.set_xlabel('Number of Arrivals'); ax.set_ylabel('Cumulative Probability')
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('assignment2_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

---

## Assignment 3

## 🧪 Hypothesis Testing

### 🧠 Algorithm
```
1. State H0 (null hypothesis) and H1 (alternative hypothesis)
2. Choose significance level α = 0.05
3. Select appropriate test:
   - One-sample / Two-sample t-test  → compare means (small samples)
   - Z-test                          → compare means (large samples, known σ)
   - Chi-Square test                 → test independence of categorical vars
   - ANOVA (F-test)                  → compare means across 3+ groups
4. Compute test statistic and p-value
5. Decision: if p < α → Reject H0, else → Fail to Reject H0
```

### ▶️ How to Run
1. Copy code below into a Colab cell
2. Run — prints hypothesis, test statistic, p-value, and decision for each test

---

```python
# ============================================================
# ASSIGNMENT 3: Hypothesis Testing
# ============================================================

# ── INSTALL ────────────────────────────────────────────────
!pip install numpy pandas scipy statsmodels matplotlib --quiet

# ── IMPORTS ────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.weightstats import ztest

print("✅ Libraries loaded")
np.random.seed(42)
ALPHA = 0.05

def print_result(test_name, stat, p_val, alpha=ALPHA):
    print(f"\n  Test      : {test_name}")
    print(f"  Statistic : {stat:.4f}")
    print(f"  p-value   : {p_val:.4f}")
    decision = "✅ Reject H0" if p_val < alpha else "❌ Fail to Reject H0"
    print(f"  Decision  : {decision}  (α = {alpha})")

# ════════════════════════════════════════════════════════════
# TEST 1: One-Sample t-test
# H0: Mean student height = 170 cm
# H1: Mean student height ≠ 170 cm
# ════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("TEST 1: One-Sample t-test")
print("  H0: μ = 170 cm  |  H1: μ ≠ 170 cm")
print("="*55)
heights = np.random.normal(loc=172, scale=8, size=30)
t_stat, p_val = stats.ttest_1samp(heights, popmean=170)
print(f"  Sample mean: {heights.mean():.2f} cm")
print_result("One-Sample t-test", t_stat, p_val)

# ════════════════════════════════════════════════════════════
# TEST 2: Two-Sample t-test
# H0: Mean scores of Group A = Group B
# H1: Mean scores differ
# ════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("TEST 2: Two-Sample t-test")
print("  H0: μ_A = μ_B  |  H1: μ_A ≠ μ_B")
print("="*55)
group_a = np.random.normal(loc=75, scale=10, size=40)
group_b = np.random.normal(loc=70, scale=10, size=40)
t_stat, p_val = stats.ttest_ind(group_a, group_b)
print(f"  Mean A: {group_a.mean():.2f}  |  Mean B: {group_b.mean():.2f}")
print_result("Independent t-test", t_stat, p_val)

# ════════════════════════════════════════════════════════════
# TEST 3: Z-test
# H0: μ = 50  |  H1: μ > 50 (one-tailed)
# ════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("TEST 3: Z-test (large sample)")
print("  H0: μ = 50  |  H1: μ > 50")
print("="*55)
large_sample = np.random.normal(loc=52, scale=15, size=200)
z_stat, p_val = ztest(large_sample, value=50, alternative='larger')
print(f"  Sample mean: {large_sample.mean():.2f}")
print_result("Z-test (one-tailed)", z_stat, p_val)

# ════════════════════════════════════════════════════════════
# TEST 4: Chi-Square Test of Independence
# H0: Gender and product preference are independent
# H1: They are associated
# ════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("TEST 4: Chi-Square Test of Independence")
print("  H0: Gender ⊥ Product Preference  |  H1: Associated")
print("="*55)
contingency = np.array([[30, 10, 20],
                         [15, 25, 10]])
chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
print(f"  Observed table:\n{contingency}")
print(f"  Degrees of Freedom: {dof}")
print_result("Chi-Square", chi2, p_val)

# ════════════════════════════════════════════════════════════
# TEST 5: One-Way ANOVA
# H0: Mean yield is same across 3 fertilizer groups
# H1: At least one group mean differs
# ════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("TEST 5: One-Way ANOVA")
print("  H0: μ1 = μ2 = μ3  |  H1: At least one differs")
print("="*55)
fert_A = np.random.normal(loc=50, scale=5, size=20)
fert_B = np.random.normal(loc=55, scale=5, size=20)
fert_C = np.random.normal(loc=52, scale=5, size=20)
f_stat, p_val = stats.f_oneway(fert_A, fert_B, fert_C)
print(f"  Means — A:{fert_A.mean():.2f}  B:{fert_B.mean():.2f}  C:{fert_C.mean():.2f}")
print_result("One-Way ANOVA (F-test)", f_stat, p_val)

# ── VISUALISATION ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Assignment 3 — Hypothesis Testing Visuals', fontsize=14, fontweight='bold')

# T-test groups
axes[0].boxplot([group_a, group_b], labels=['Group A', 'Group B'], patch_artist=True,
                boxprops=dict(facecolor='lightblue'))
axes[0].set_title('Two-Sample t-test\nGroup Comparison')
axes[0].set_ylabel('Score')

# Chi-square heatmap
import matplotlib.colors as mcolors
im = axes[1].imshow(contingency, cmap='Blues', aspect='auto')
axes[1].set_xticks([0,1,2]); axes[1].set_xticklabels(['Prod A','Prod B','Prod C'])
axes[1].set_yticks([0,1]); axes[1].set_yticklabels(['Male','Female'])
axes[1].set_title('Chi-Square\nContingency Table')
for i in range(2):
    for j in range(3):
        axes[1].text(j, i, contingency[i,j], ha='center', va='center', fontsize=12)

# ANOVA groups
axes[2].boxplot([fert_A, fert_B, fert_C], labels=['Fert A','Fert B','Fert C'],
                patch_artist=True, boxprops=dict(facecolor='lightyellow'))
axes[2].set_title('One-Way ANOVA\nFertilizer Groups')
axes[2].set_ylabel('Yield')

plt.tight_layout()
plt.savefig('assignment3_hypothesis.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

---

## Assignment 4

## 🔧 Data Wrangling + PySpark + SQL

### 🧠 Algorithm
```
IRIS Data Wrangling:
  1. Load IRIS dataset via sklearn / seaborn
  2. Handle missing values, rename columns, encode labels
  3. Compute group-wise statistics
  4. Filter, sort, and transform data

PySpark + SQL on Book Dataset:
  1. Install Java + PySpark on Colab
  2. Initialize SparkSession
  3. Create Book DataFrame
  4. Register as TempView → run SQL queries
  5. Perform DataFrame API transformations
```

### ▶️ How to Run
1. Copy code into a Colab cell
2. First install block sets up Java + PySpark (takes ~1 min)
3. Run remaining cells sequentially

---

```python
# ============================================================
# ASSIGNMENT 4: Data Wrangling (IRIS) + PySpark + SQL (Books)
# ============================================================

# ── INSTALL ────────────────────────────────────────────────
!apt-get install -y openjdk-11-jdk-headless -qq > /dev/null
!pip install pyspark pandas numpy matplotlib seaborn --quiet
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
print("✅ Java + PySpark installed")

# ── IMPORTS ────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, desc, upper, length

# ════════════════════════════════════════════════════════════
# PART A: IRIS Data Wrangling with Pandas
# ════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("PART A: IRIS Dataset — Data Wrangling")
print("="*55)

iris = load_iris()
df = pd.DataFrame(iris.data, columns=['sepal_length','sepal_width',
                                       'petal_length','petal_width'])
df['species_id']  = iris.target
df['species']     = df['species_id'].map({0:'setosa', 1:'versicolor', 2:'virginica'})

print("\n🔹 Shape:", df.shape)
print("\n🔹 First 5 rows:\n", df.head())
print("\n🔹 Missing values:\n", df.isnull().sum())
print("\n🔹 Descriptive stats:\n", df.describe().round(3))

# Group-wise statistics
print("\n🔹 Mean per species:")
print(df.groupby('species')[['sepal_length','petal_length']].mean().round(3))

# Filter: sepal_length > 6.0
filtered = df[df['sepal_length'] > 6.0]
print(f"\n🔹 Rows with sepal_length > 6.0: {len(filtered)}")

# New feature: petal_ratio
df['petal_ratio'] = (df['petal_length'] / df['petal_width']).round(3)
print("\n🔹 Petal ratio (first 5):\n", df[['species','petal_ratio']].head())

# Visualise
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(data=df, x='species', y='sepal_length', ax=axes[0], palette='Set2')
axes[0].set_title('Sepal Length by Species')
sns.scatterplot(data=df, x='petal_length', y='petal_width',
                hue='species', ax=axes[1], palette='Set1')
axes[1].set_title('Petal Length vs Width')
plt.tight_layout()
plt.savefig('assignment4_iris_wrangling.png', dpi=150, bbox_inches='tight')
plt.show()

# ════════════════════════════════════════════════════════════
# PART B: PySpark + SQL on Book Dataset
# ════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("PART B: PySpark + SQL — Book Dataset")
print("="*55)

spark = SparkSession.builder \
    .appName("BookDatasetAnalysis") \
    .master("local[*]") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
print("✅ SparkSession created")

# Create Book Dataset
book_data = [
    (1, "The Great Gatsby",         "F. Scott Fitzgerald", "Fiction",    1925, 4.1, 500000),
    (2, "To Kill a Mockingbird",    "Harper Lee",          "Fiction",    1960, 4.3, 750000),
    (3, "1984",                     "George Orwell",       "Dystopian",  1949, 4.5, 800000),
    (4, "Pride and Prejudice",      "Jane Austen",         "Romance",    1813, 4.2, 600000),
    (5, "The Catcher in the Rye",   "J.D. Salinger",       "Fiction",    1951, 3.8, 400000),
    (6, "Brave New World",          "Aldous Huxley",       "Dystopian",  1932, 4.0, 350000),
    (7, "The Hobbit",               "J.R.R. Tolkien",      "Fantasy",    1937, 4.6, 900000),
    (8, "Harry Potter",             "J.K. Rowling",        "Fantasy",    1997, 4.8, 1200000),
    (9, "The Da Vinci Code",        "Dan Brown",           "Thriller",   2003, 3.7, 450000),
    (10,"Sapiens",                  "Yuval Noah Harari",   "Non-Fiction",2011, 4.4, 650000),
]
schema = ["book_id","title","author","genre","year","rating","copies_sold"]
books_df = spark.createDataFrame(book_data, schema=schema)

books_df.registerTempTable("books")
print("\n🔹 Schema:"); books_df.printSchema()
print("\n🔹 All Books:"); books_df.show(truncate=False)

# SQL Query 1: Top-rated books
print("🔹 SQL — Top 5 Rated Books:")
spark.sql("""
    SELECT title, author, rating
    FROM books
    ORDER BY rating DESC
    LIMIT 5
""").show(truncate=False)

# SQL Query 2: Genre-wise average rating
print("🔹 SQL — Average Rating per Genre:")
spark.sql("""
    SELECT genre,
           COUNT(*) as book_count,
           ROUND(AVG(rating), 2) as avg_rating,
           SUM(copies_sold) as total_copies
    FROM books
    GROUP BY genre
    ORDER BY avg_rating DESC
""").show()

# SQL Query 3: Books after 1950
print("🔹 SQL — Books Published After 1950:")
spark.sql("SELECT title, year, genre FROM books WHERE year > 1950 ORDER BY year").show(truncate=False)

# DataFrame API — filter + transform
print("🔹 DataFrame API — Fantasy/Dystopian books with rating > 4.0:")
books_df.filter(
    (col("genre").isin("Fantasy","Dystopian")) & (col("rating") > 4.0)
).select("title","genre","rating","copies_sold").show(truncate=False)

spark.stop()
```

---

---

## Assignment 5

## 📉 Linear Regression — Boston Housing Dataset

### 🧠 Algorithm
```
1. Load / generate Boston-style housing dataset
2. Exploratory analysis — correlations, scatter plots
3. Split data: 80% train, 20% test
4. Fit Linear Regression model (sklearn)
5. Predict on test set
6. Evaluate: MAE, MSE, RMSE, R²
7. Plot: Actual vs Predicted, Residuals, KDE of residuals
8. Interpret coefficients
```

### ▶️ How to Run
1. Paste code into Colab cell and run

---

```python
# ============================================================
# ASSIGNMENT 5: Linear Regression — Boston Housing
# ============================================================

# ── INSTALL ────────────────────────────────────────────────
!pip install numpy pandas matplotlib seaborn scikit-learn --quiet

# ── IMPORTS ────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

print("✅ Libraries loaded")
np.random.seed(42)

# ── DATASET (Boston-style synthetic — avoids deprecation) ──
n = 506
crime_rate  = np.random.exponential(3.6, n)
rooms       = np.random.normal(6.3, 0.7, n)
age         = np.random.uniform(2, 100, n)
distance    = np.random.exponential(3.8, n)
tax         = np.random.normal(408, 170, n)
ptratio     = np.random.normal(18.5, 2.1, n)

# House price (target) with realistic weights + noise
price = (
    -0.9 * crime_rate
    + 5.8 * rooms
    - 0.05 * age
    - 0.8 * distance
    - 0.01 * tax
    - 0.5 * ptratio
    + np.random.normal(0, 3, n)
    + 10
)

df = pd.DataFrame({
    'CRIM': crime_rate, 'RM': rooms, 'AGE': age,
    'DIS': distance, 'TAX': tax, 'PTRATIO': ptratio,
    'MEDV': price
})

print("\n🔹 Dataset shape:", df.shape)
print("\n🔹 Descriptive Stats:\n", df.describe().round(3))

# ── CORRELATION ────────────────────────────────────────────
print("\n🔹 Correlation with MEDV:\n",
      df.corr()['MEDV'].sort_values(ascending=False).round(3))

# ── TRAIN/TEST SPLIT ───────────────────────────────────────
X = df.drop('MEDV', axis=1)
y = df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── MODEL ──────────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train_sc, y_train)
y_pred = model.predict(X_test_sc)

# ── METRICS ────────────────────────────────────────────────
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print("\n📊 Model Performance:")
print(f"  MAE  = {mae:.4f}")
print(f"  MSE  = {mse:.4f}")
print(f"  RMSE = {rmse:.4f}")
print(f"  R²   = {r2:.4f}")

# Coefficients
coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
print("\n🔹 Coefficients:\n", coef_df.sort_values('Coefficient', ascending=False).to_string(index=False))

# ── VISUALISATION ──────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Assignment 5 — Linear Regression: Boston Housing', fontsize=15, fontweight='bold')

# 1. Scatter: RM vs MEDV
axes[0,0].scatter(df['RM'], df['MEDV'], alpha=0.4, color='steelblue')
m, b = np.polyfit(df['RM'], df['MEDV'], 1)
xr = np.linspace(df['RM'].min(), df['RM'].max(), 100)
axes[0,0].plot(xr, m*xr+b, 'r-', linewidth=2)
axes[0,0].set_title('Scatter: Rooms vs Price')
axes[0,0].set_xlabel('RM'); axes[0,0].set_ylabel('MEDV')

# 2. Scatter: CRIM vs MEDV
axes[0,1].scatter(df['CRIM'], df['MEDV'], alpha=0.4, color='coral')
axes[0,1].set_title('Scatter: Crime Rate vs Price')
axes[0,1].set_xlabel('CRIM'); axes[0,1].set_ylabel('MEDV')

# 3. Correlation heatmap
corr = df.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[0,2], linewidths=0.5)
axes[0,2].set_title('Correlation Matrix')

# 4. Actual vs Predicted
axes[1,0].scatter(y_test, y_pred, alpha=0.5, color='purple')
mn = min(y_test.min(), y_pred.min())
mx = max(y_test.max(), y_pred.max())
axes[1,0].plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='Perfect Fit')
axes[1,0].set_title(f'Actual vs Predicted (R²={r2:.3f})')
axes[1,0].set_xlabel('Actual'); axes[1,0].set_ylabel('Predicted')
axes[1,0].legend()

# 5. Residuals
residuals = y_test - y_pred
axes[1,1].scatter(y_pred, residuals, alpha=0.5, color='darkorange')
axes[1,1].axhline(0, color='red', linestyle='--')
axes[1,1].set_title('Residual Plot')
axes[1,1].set_xlabel('Predicted'); axes[1,1].set_ylabel('Residuals')

# 6. KDE of residuals
axes[1,2].set_title('KDE — Residual Distribution')
from scipy.stats import gaussian_kde
kde_vals = gaussian_kde(residuals)
x_res = np.linspace(residuals.min(), residuals.max(), 300)
axes[1,2].plot(x_res, kde_vals(x_res), 'b-', linewidth=2, label='KDE')
axes[1,2].hist(residuals, bins=25, density=True, alpha=0.4, color='skyblue', label='Histogram')
axes[1,2].axvline(0, color='red', linestyle='--')
axes[1,2].set_xlabel('Residual'); axes[1,2].set_ylabel('Density')
axes[1,2].legend()

plt.tight_layout()
plt.savefig('assignment5_linear_regression.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

---

## Assignment 6

## 🔵 Logistic Regression + K-Means Clustering

### 🧠 Algorithm
```
Logistic Regression (IRIS):
  1. Load IRIS (binary: setosa vs non-setosa for simplicity; or multiclass)
  2. Train/test split → StandardScaler
  3. Fit LogisticRegression model
  4. Predict probabilities and classes
  5. Evaluate: Accuracy, Precision, Recall, F1, Confusion Matrix

K-Means Clustering (PySpark):
  1. Initialize SparkSession
  2. Load IRIS feature columns into Spark DataFrame
  3. Assemble features → VectorAssembler
  4. Fit KMeans(k=3)
  5. Predict cluster labels
  6. Visualise clusters
```

### ▶️ How to Run
1. Paste into Colab and run — PySpark auto-installed

---

```python
# ============================================================
# ASSIGNMENT 6: Logistic Regression (IRIS) + K-Means (PySpark)
# ============================================================

# ── INSTALL ────────────────────────────────────────────────
!apt-get install -y openjdk-11-jdk-headless -qq > /dev/null
!pip install pyspark scikit-learn pandas numpy matplotlib seaborn --quiet
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
print("✅ All dependencies installed")

# ── IMPORTS ────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler as SparkScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

print("✅ Imports done")

# ════════════════════════════════════════════════════════════
# PART A: Logistic Regression on IRIS
# ════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("PART A: Logistic Regression — IRIS Dataset")
print("="*55)

iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                      random_state=42, stratify=y)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

log_reg = LogisticRegression(max_iter=200, multi_class='multinomial', random_state=42)
log_reg.fit(X_train_sc, y_train)
y_pred = log_reg.predict(X_test_sc)

acc = accuracy_score(y_test, y_pred)
print(f"\n  Accuracy: {acc:.4f}  ({acc*100:.2f}%)")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# ════════════════════════════════════════════════════════════
# PART B: K-Means Clustering with PySpark
# ════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("PART B: K-Means Clustering — PySpark")
print("="*55)

spark = SparkSession.builder \
    .appName("KMeansIRIS") \
    .master("local[*]") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

df_pd = pd.DataFrame(X, columns=['sepal_length','sepal_width','petal_length','petal_width'])
df_pd['true_label'] = y
sdf = spark.createDataFrame(df_pd)

assembler = VectorAssembler(
    inputCols=['sepal_length','sepal_width','petal_length','petal_width'],
    outputCol='features')
sdf_assembled = assembler.transform(sdf)

spark_scaler = SparkScaler(inputCol='features', outputCol='scaled_features')
scaler_model = spark_scaler.fit(sdf_assembled)
sdf_scaled   = scaler_model.transform(sdf_assembled)

kmeans = KMeans(featuresCol='scaled_features', k=3, seed=42, maxIter=100)
km_model = kmeans.fit(sdf_scaled)
predictions = km_model.transform(sdf_scaled)

evaluator = ClusteringEvaluator(featuresCol='scaled_features')
silhouette = evaluator.evaluate(predictions)
print(f"\n  Silhouette Score (k=3): {silhouette:.4f}")

pred_pd = predictions.select('sepal_length','sepal_width','petal_length',
                               'petal_width','true_label','prediction').toPandas()
spark.stop()

# ── VISUALISATION ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Assignment 6 — Logistic Regression & K-Means', fontsize=14, fontweight='bold')

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(ax=axes[0], cmap='Blues', colorbar=False)
axes[0].set_title(f'Confusion Matrix\nAccuracy: {acc:.3f}')

# K-Means clusters (petal_length vs petal_width)
scatter_colors = ['#e74c3c','#2ecc71','#3498db']
for cluster in range(3):
    mask = pred_pd['prediction'] == cluster
    axes[1].scatter(pred_pd.loc[mask,'petal_length'],
                    pred_pd.loc[mask,'petal_width'],
                    c=scatter_colors[cluster], label=f'Cluster {cluster}', alpha=0.7, s=50)
axes[1].set_title(f'K-Means Clusters\nSilhouette={silhouette:.3f}')
axes[1].set_xlabel('Petal Length'); axes[1].set_ylabel('Petal Width')
axes[1].legend()

# True labels (ground truth)
true_colors = ['#e74c3c','#2ecc71','#3498db']
for label, name in enumerate(target_names):
    mask = pred_pd['true_label'] == label
    axes[2].scatter(pred_pd.loc[mask,'petal_length'],
                    pred_pd.loc[mask,'petal_width'],
                    c=true_colors[label], label=name, alpha=0.7, s=50)
axes[2].set_title('Ground Truth Labels')
axes[2].set_xlabel('Petal Length'); axes[2].set_ylabel('Petal Width')
axes[2].legend()

plt.tight_layout()
plt.savefig('assignment6_logistic_kmeans.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

---

## Assignment 7

## ⚡ RDD Actions & Transformations — Apache Spark

### 🧠 Algorithm
```
TRANSFORMATIONS (lazy — return new RDD):
  map, flatMap, filter, distinct, union, intersection,
  groupByKey, reduceByKey, sortBy, join

ACTIONS (eager — trigger computation, return result):
  collect, count, first, take, reduce, sum, mean,
  saveAsTextFile, countByValue

1. Create SparkContext
2. Parallelize data into RDDs
3. Apply transformations (build DAG)
4. Trigger actions (execute DAG)
5. Demonstrate word count MapReduce with RDDs
```

### ▶️ How to Run
1. Paste code into Colab cell and run

---

```python
# ============================================================
# ASSIGNMENT 7: RDD Actions & Transformations — Apache Spark
# ============================================================

# ── INSTALL ────────────────────────────────────────────────
!apt-get install -y openjdk-11-jdk-headless -qq > /dev/null
!pip install pyspark --quiet
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
print("✅ PySpark installed")

# ── IMPORTS ────────────────────────────────────────────────
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import re

conf = SparkConf().setAppName("RDD_Demo").setMaster("local[*]")
sc   = SparkContext(conf=conf)
sc.setLogLevel("ERROR")
print("✅ SparkContext created")

# ════════════════════════════════════════════════════════════
# 1. BASIC RDD CREATION
# ════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("1. Creating RDDs")
print("="*55)

nums_rdd   = sc.parallelize([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5])
fruits_rdd = sc.parallelize(["apple","banana","cherry","apple","mango","banana","apple"])
print("  Numbers RDD:", nums_rdd.collect())
print("  Fruits RDD :", fruits_rdd.collect())

# ════════════════════════════════════════════════════════════
# 2. TRANSFORMATIONS
# ════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("2. Transformations (Lazy)")
print("="*55)

# map
squared = nums_rdd.map(lambda x: x**2)
print("  map (x²)          :", squared.collect())

# filter
evens = nums_rdd.filter(lambda x: x % 2 == 0)
print("  filter (even)     :", evens.collect())

# distinct
distinct_nums = nums_rdd.distinct()
print("  distinct          :", sorted(distinct_nums.collect()))

# flatMap
words_rdd = sc.parallelize(["Hello World", "Apache Spark", "Big Data"])
flat = words_rdd.flatMap(lambda s: s.split())
print("  flatMap (words)   :", flat.collect())

# sortBy
sorted_rdd = nums_rdd.sortBy(lambda x: x, ascending=False)
print("  sortBy (desc)     :", sorted_rdd.collect())

# union
rdd_a = sc.parallelize([1, 2, 3])
rdd_b = sc.parallelize([3, 4, 5])
print("  union             :", rdd_a.union(rdd_b).collect())
print("  intersection      :", rdd_a.intersection(rdd_b).collect())

# ════════════════════════════════════════════════════════════
# 3. KEY-VALUE RDD TRANSFORMATIONS
# ════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("3. Key-Value Pair Transformations")
print("="*55)

sales = sc.parallelize([
    ("Electronics", 1500), ("Clothing", 800),  ("Electronics", 2200),
    ("Food", 400),         ("Clothing", 1200),  ("Food", 600),
    ("Electronics", 900),  ("Clothing", 300),
])

# reduceByKey
total_by_dept = sales.reduceByKey(lambda a, b: a + b)
print("  reduceByKey (total sales):")
for dept, total in sorted(total_by_dept.collect()):
    print(f"    {dept:15s}: ₹{total}")

# countByValue
fruit_counts = fruits_rdd.countByValue()
print("\n  countByValue (fruits):", dict(fruit_counts))

# groupByKey
grouped = sales.groupByKey().mapValues(list)
print("\n  groupByKey (sales values):")
for dept, vals in sorted(grouped.collect()):
    print(f"    {dept:15s}: {vals}")

# sortBy on pairs
sorted_sales = total_by_dept.sortBy(lambda x: x[1], ascending=False)
print("\n  sortBy value (descending):", sorted_sales.collect())

# ════════════════════════════════════════════════════════════
# 4. ACTIONS
# ════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("4. Actions (Eager Execution)")
print("="*55)

print(f"  count()    : {nums_rdd.count()}")
print(f"  first()    : {nums_rdd.first()}")
print(f"  take(3)    : {nums_rdd.take(3)}")
print(f"  top(3)     : {nums_rdd.top(3)}")
print(f"  sum()      : {nums_rdd.sum()}")
print(f"  mean()     : {nums_rdd.mean():.4f}")
print(f"  max()      : {nums_rdd.max()}")
print(f"  min()      : {nums_rdd.min()}")
print(f"  reduce(+)  : {nums_rdd.reduce(lambda a,b: a+b)}")

# ════════════════════════════════════════════════════════════
# 5. WORD COUNT (Classic MapReduce with RDDs)
# ════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("5. Word Count — MapReduce with RDDs")
print("="*55)

text_data = [
    "Spark is fast and reliable",
    "Spark is easy to use and fast",
    "Big data processing with Spark",
    "RDD transformations and actions in Spark"
]
text_rdd = sc.parallelize(text_data)

word_count = (
    text_rdd
    .flatMap(lambda line: line.lower().split())
    .map(lambda word: (word, 1))
    .reduceByKey(lambda a, b: a + b)
    .sortBy(lambda x: x[1], ascending=False)
)
print("  Word Counts (top 10):")
for word, count in word_count.take(10):
    print(f"    {word:20s}: {count}")

# ── PERSIST / CACHE ────────────────────────────────────────
print("\n" + "="*55)
print("6. Persist & Cache")
print("="*55)
cached = total_by_dept.cache()
print("  RDD cached. Count:", cached.count())
print("  Accessing again (from cache):", cached.collect())

sc.stop()
```

---

---

## Assignment 8

## 🗺️ MapReduce — Hadoop & PySpark

### 🧠 Algorithm
```
MapReduce Pattern:
  INPUT  →  MAP phase  →  SHUFFLE & SORT  →  REDUCE phase  →  OUTPUT

Map phase:
  Read input records
  Emit (key, value) pairs

Shuffle & Sort:
  Group all values by key (framework handles this)

Reduce phase:
  For each key, aggregate/process its value list
  Emit final (key, result)

Problems implemented:
  1. Word Count
  2. Sales Analytics (total, avg, max by category)
  3. Log File Analysis (count by log level)
  4. Student Grade Statistics
  5. Inverted Index
```

### ▶️ How to Run
1. Paste the full code into a Colab cell and run
2. Pure PySpark MapReduce patterns are used (no Hadoop install needed on Colab)

---

```python
# ============================================================
# ASSIGNMENT 8: MapReduce — Hadoop & PySpark
# ============================================================

# ── INSTALL ────────────────────────────────────────────────
!apt-get install -y openjdk-17-jdk-headless -qq

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-amd64"
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]

# ── IMPORTS ────────────────────────────────────────────────
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import re, math

conf = SparkConf().setAppName("MapReduce_Demo").setMaster("local[*]")
sc   = SparkContext(conf=conf)
sc.setLogLevel("ERROR")
print("✅ SparkContext created")

# ════════════════════════════════════════════════════════════
# MAPREDUCE 1: Word Count (Classic)
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MAPREDUCE 1: Word Frequency Count")
print("  MAP    : each word → (word, 1)")
print("  REDUCE : sum all 1s per word")
print("="*60)

documents = [
    "hadoop mapreduce processes large datasets in parallel",
    "spark provides in-memory processing for big data",
    "mapreduce divides data into map and reduce phases",
    "big data analytics requires distributed computing",
    "spark and hadoop are popular big data frameworks",
    "map phase transforms data reduce phase aggregates data",
]

text_rdd = sc.parallelize(documents)

# MAP → SHUFFLE → REDUCE
word_count_rdd = (
    text_rdd
    .flatMap(lambda line: re.findall(r'\b\w+\b', line.lower()))   # MAP
    .map(lambda word: (word, 1))                                    # (key, value)
    .reduceByKey(lambda a, b: a + b)                               # REDUCE
    .sortBy(lambda x: x[1], ascending=False)
)

print("\n  Top 15 words:")
print(f"  {'Word':<20} {'Count':>5}")
print(f"  {'-'*25}")
for word, cnt in word_count_rdd.take(15):
    print(f"  {word:<20} {cnt:>5}")

# ════════════════════════════════════════════════════════════
# MAPREDUCE 2: Sales Analytics per Category
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MAPREDUCE 2: Sales Analytics by Category")
print("  MAP    : transaction → (category, amount)")
print("  REDUCE : compute total, count, max, avg per category")
print("="*60)

transactions = [
    ("Electronics", 1500), ("Clothing", 800),  ("Electronics", 2200),
    ("Food", 400),         ("Clothing", 1200),  ("Food", 600),
    ("Electronics", 900),  ("Clothing", 300),   ("Food", 350),
    ("Electronics", 3200), ("Clothing", 750),   ("Food", 820),
    ("Books", 250),        ("Books", 380),       ("Books", 120),
]

sales_rdd = sc.parallelize(transactions)

# MAP: (category, amount) → already in key-value form
# REDUCE: aggregate stats
def reduce_stats(a, b):
    # a, b are (total, count, max_val)
    return (a[0]+b[0], a[1]+b[1], max(a[2], b[2]))

sales_stats = (
    sales_rdd
    .map(lambda x: (x[0], (x[1], 1, x[1])))           # MAP: (cat, (amt, 1, amt))
    .reduceByKey(reduce_stats)                           # REDUCE: aggregate
    .map(lambda x: (x[0], x[1][0], x[1][1],            # FORMAT output
                    round(x[1][0]/x[1][1], 2), x[1][2]))
    .sortBy(lambda x: x[1], ascending=False)
)

print(f"\n  {'Category':<15} {'Total':>8} {'Count':>6} {'Average':>9} {'Max':>7}")
print(f"  {'-'*47}")
for cat, total, cnt, avg, mx in sales_stats.collect():
    print(f"  {cat:<15} {total:>8} {cnt:>6} {avg:>9} {mx:>7}")

# ════════════════════════════════════════════════════════════
# MAPREDUCE 3: Log File Analysis
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MAPREDUCE 3: Log File Analysis")
print("  MAP    : log line → (log_level, 1)")
print("  REDUCE : count occurrences per level")
print("="*60)

log_lines = [
    "2024-01-15 10:23:01 ERROR Database connection failed",
    "2024-01-15 10:23:02 INFO  Server started on port 8080",
    "2024-01-15 10:23:03 WARN  Memory usage above 80%",
    "2024-01-15 10:23:04 INFO  Request received GET /api/users",
    "2024-01-15 10:23:05 ERROR NullPointerException in UserService",
    "2024-01-15 10:23:06 DEBUG Processing user authentication",
    "2024-01-15 10:23:07 INFO  Response sent 200 OK",
    "2024-01-15 10:23:08 WARN  Slow query detected (2.3s)",
    "2024-01-15 10:23:09 ERROR Timeout connecting to cache",
    "2024-01-15 10:23:10 INFO  User logged in: user123",
    "2024-01-15 10:23:11 DEBUG Cache miss for key user_456",
    "2024-01-15 10:23:12 INFO  Scheduled job started",
    "2024-01-15 10:23:13 WARN  API rate limit approaching",
    "2024-01-15 10:23:14 ERROR Disk space critically low",
]

log_rdd = sc.parallelize(log_lines)

log_levels = (
    log_rdd
    .map(lambda line: (line.split()[2].strip(), 1))    # MAP: extract level
    .reduceByKey(lambda a, b: a + b)                    # REDUCE: count
    .sortBy(lambda x: x[1], ascending=False)
)

total_logs = len(log_lines)
print(f"\n  {'Level':<10} {'Count':>6} {'Percentage':>12}")
print(f"  {'-'*30}")
for level, cnt in log_levels.collect():
    pct = cnt / total_logs * 100
    print(f"  {level:<10} {cnt:>6} {pct:>10.1f}%")

# ════════════════════════════════════════════════════════════
# MAPREDUCE 4: Student Grade Statistics
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MAPREDUCE 4: Student Grade Statistics by Subject")
print("  MAP    : (student, subject, marks) → (subject, marks)")
print("  REDUCE : compute avg, max, pass% per subject")
print("="*60)

student_records = [
    ("Alice",   "Math",    85), ("Bob",   "Math",    72), ("Carol", "Math",    91),
    ("Dave",    "Math",    60), ("Eve",   "Math",    78), ("Alice", "Science", 88),
    ("Bob",     "Science", 65), ("Carol", "Science", 79), ("Dave",  "Science", 55),
    ("Eve",     "Science", 92), ("Alice", "English", 76), ("Bob",   "English", 83),
    ("Carol",   "English", 68), ("Dave",  "English", 74), ("Eve",   "English", 89),
]

student_rdd = sc.parallelize(student_records)

subject_stats = (
    student_rdd
    .map(lambda x: (x[1], (x[2], 1, x[2], 1 if x[2] >= 70 else 0)))   # MAP
    .reduceByKey(lambda a, b: (a[0]+b[0], a[1]+b[1],
                                max(a[2],b[2]), a[3]+b[3]))              # REDUCE
    .map(lambda x: (x[0],
                    round(x[1][0]/x[1][1], 2),   # avg
                    x[1][2],                      # max
                    x[1][1],                      # count
                    round(x[1][3]/x[1][1]*100,1))) # pass%
    .sortBy(lambda x: x[1], ascending=False)
)

print(f"\n  {'Subject':<10} {'Avg':>6} {'Max':>5} {'Students':>9} {'Pass%':>7}")
print(f"  {'-'*40}")
for subj, avg, mx, cnt, pp in subject_stats.collect():
    print(f"  {subj:<10} {avg:>6} {mx:>5} {cnt:>9} {pp:>6}%")

# ════════════════════════════════════════════════════════════
# MAPREDUCE 5: Inverted Index
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MAPREDUCE 5: Inverted Index (word → documents)")
print("  MAP    : (doc_id, word) → (word, doc_id)")
print("  REDUCE : for each word, collect all doc_ids")
print("="*60)

docs = [
    (1, "spark hadoop mapreduce big data"),
    (2, "hadoop distributed file system hdfs"),
    (3, "spark streaming real time data processing"),
    (4, "mapreduce programming model hadoop"),
    (5, "big data analytics spark sql"),
]

docs_rdd = sc.parallelize(docs)

inverted_index = (
    docs_rdd
    .flatMap(lambda x: [(word, x[0]) for word in x[1].split()])   # MAP
    .groupByKey()                                                    # SHUFFLE
    .mapValues(lambda doc_ids: sorted(set(doc_ids)))                # REDUCE
    .sortByKey()
)

print(f"\n  {'Word':<15} {'Documents'}")
print(f"  {'-'*35}")
for word, doc_list in inverted_index.collect():
    print(f"  {word:<15} {doc_list}")

sc.stop()
```

---

## 📂 Output Files Generated

| Assignment | Output File |
|-----------|------------|
| 1 | `assignment1_eda.png` |
| 2 | `assignment2_distributions.png` |
| 3 | `assignment3_hypothesis.png` |
| 4 | `assignment4_iris_wrangling.png` |
| 5 | `assignment5_linear_regression.png` |
| 6 | `assignment6_logistic_kmeans.png` |
| 7 | *(console output)* |
| 8 | *(console output)* |

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---------|-----|
| Java not found (PySpark) | Re-run the install cell; JAVA_HOME is set automatically |
| `ModuleNotFoundError` | Run `!pip install <package> --quiet` in a new cell |
| Spark taking too long | Runtime → Restart and run all |
| Plot not showing | Add `%matplotlib inline` at top of cell |
| Memory error | Runtime → Runtime type → T4 GPU (more RAM) |

---

## 📚 References

- [PySpark Docs](https://spark.apache.org/docs/latest/api/python/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [SciPy Stats](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [Seaborn Gallery](https://seaborn.pydata.org/examples/index.html)
- [Google Colab Tips](https://colab.research.google.com/notebooks/basic_features_overview.ipynb)

---

*Made for Google Colab · All 8 assignments · Copy → Paste → Run*
