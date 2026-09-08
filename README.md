# 🛒 Market Basket Analysis & Customer Segmentation

## 📌 Project Overview

This project analyzes **Online Retail transaction data** to understand customer purchasing behavior and discover useful patterns in product purchases.

The project combines two techniques:

* 🛒 **Market Basket Analysis** using the **Apriori Algorithm**
* 👥 **Customer Segmentation** using **K-Means Clustering**

The results can help businesses understand which products are purchased together and identify different groups of customers based on their purchasing behavior.

---

## ⏩ Problem Statement

* 🛒 Retail stores generate a large amount of transaction data every day.
* 🔍 It is difficult to manually identify which products are frequently purchased together.
* 📊 Customer purchasing behavior can vary significantly from one customer to another.
* 👥 Without proper segmentation, it is difficult to understand different types of customers.
* 🔗 Market Basket Analysis can identify relationships between products using the Apriori algorithm.
* 📈 Customer Segmentation can group customers with similar purchasing behavior using K-Means clustering.
* 🎯 The project aims to discover **product associations** and **customer segments** to support better retail decision-making.

---

## 🎯 Objectives

1. 🧹 Clean and preprocess the retail transaction dataset.
2. 📊 Perform exploratory data analysis on the retail data.
3. 🔗 Find frequently purchased products using the **Apriori algorithm**.
4. 📈 Generate association rules using **Support, Confidence, and Lift**.
5. 👥 Segment customers using **K-Means clustering**.
6. 🎯 Understand different customer purchasing patterns.
7. 💡 Generate useful insights for product recommendations and marketing strategies.

---

## 📊 Dataset Information

**Dataset Name:** Online Retail Dataset
**File:** `Online Retail.xlsx`

| Column         | Description                           |
| -------------- | ------------------------------------- |
| 🧾 InvoiceNo   | Unique invoice number                 |
| 🏷️ StockCode  | Unique product code                   |
| 📝 Description | Product name or description           |
| 🔢 Quantity    | Number of products purchased          |
| 📅 InvoiceDate | Date and time of the transaction      |
| 💰 UnitPrice   | Price of one unit of the product      |
| 👤 CustomerID  | Unique customer identification number |
| 🌍 Country     | Country of the customer               |

---

## 🛒 Market Basket Analysis

Market Basket Analysis is used to discover **products that are frequently purchased together**.

### 🔗 Apriori Algorithm

The Apriori algorithm is used to find frequent itemsets from customer transactions.

The project generates:

* 📦 Frequent Itemsets
* 🔗 Association Rules
* 📊 Support
* 🎯 Confidence
* 📈 Lift

### Example

If customers frequently purchase **Product A** and **Product B** together, an association rule can help identify this relationship.

This information can be useful for:

* 🛍️ Product recommendations
* 🎁 Product bundling
* 📢 Promotional offers
* 🏪 Store arrangement

---

## 👥 Customer Segmentation

Customer Segmentation groups customers with similar purchasing behavior.

### 🤖 K-Means Clustering

The **K-Means algorithm** is used to divide customers into different clusters based on their purchasing characteristics.

This can help identify:

* 👑 High-value customers
* 🛍️ Frequent purchasers
* 💳 Customers with different spending patterns
* 📊 Groups with similar purchasing behavior

These segments can support targeted marketing and personalized strategies.

---

## 🛠️ Technologies Used

* 🐍 Python
* 📓 Jupyter Notebook / Google Colab
* 🐼 Pandas
* 🔢 NumPy
* 📊 Matplotlib
* 📈 Seaborn
* 🤖 Scikit-learn
* 🔗 Mlxtend
* 💾 Joblib / Pickle
* 🎨 Streamlit

---

## 📁 Project Files

```text
Market-Basket-Analysis-Customer-Segmentation/
│
├── 📓 Market_Basket_Analysis_and_Customer_Segmentation.ipynb
├── 📊 Online Retail.xlsx
│
├── 💾 frequent_items(2).pkl
├── 💾 association_rules(1).pkl
├── 💾 k-means_customer_segmentation_model(2).pkl
│
├── 🎨 app.py
└── 📖 README.md
```

---

## 💾 Saved Models

The trained results/models are saved using Pickle/Joblib so they can be reused in the Streamlit application without performing the complete analysis again.

### Files:

* `frequent_items(2).pkl`
  📦 Stores the frequent itemsets.

* `association_rules(1).pkl`
  🔗 Stores the generated association rules.

* `k-means_customer_segmentation_model(2).pkl`
  👥 Stores the trained K-Means customer segmentation model.

---

## 🎨 Streamlit Application

A Streamlit application can be used to provide an interactive interface for the project.

The application can allow users to:

* 🛒 Explore frequent products
* 🔗 View association rules
* 📊 Analyze Support, Confidence, and Lift
* 👥 Perform customer segmentation
* 📈 View customer clusters
* 💡 Understand retail insights

---

## ▶️ How to Run

### 1️⃣ Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn mlxtend streamlit joblib openpyxl
```

### 2️⃣ Run the Streamlit Application

```bash
streamlit run app.py
```

### 3️⃣ Open the Application

After running the command, Streamlit will provide a local URL where the application can be viewed in a web browser.

---

## 📈 Key Outcomes

The project provides two major types of insights:

### 🛒 Product-Level Insights

* Identify frequently purchased products.
* Discover product relationships.
* Generate association rules.
* Support cross-selling and product recommendations.

### 👥 Customer-Level Insights

* Group customers based on purchasing behavior.
* Identify different customer segments.
* Support targeted marketing.
* Improve customer-focused strategies.

---

## 🌟 Project Benefits

* 📊 Converts raw transaction data into useful insights.
* 🛒 Helps understand product purchasing patterns.
* 👥 Helps identify different customer groups.
* 🎯 Supports targeted marketing strategies.
* 💡 Can improve product recommendation and cross-selling decisions.
* 🚀 Provides an interactive Streamlit-based application.

---

## 👩‍💻 Project Type

**Data Science | Machine Learning | Association Rule Mining | Customer Segmentation**

### 🔑 Algorithms Used

**Apriori Algorithm + K-Means Clustering**

---

## ⭐ Conclusion

This project demonstrates how retail transaction data can be analyzed using **Market Basket Analysis and Customer Segmentation**.

The **Apriori algorithm** helps discover relationships between products, while **K-Means clustering** helps identify groups of customers with similar purchasing behavior.

Together, these techniques provide valuable insights that can help businesses make better **marketing, recommendation, and customer-targeting decisions**.

