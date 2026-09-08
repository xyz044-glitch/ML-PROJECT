import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Retail Analytics Dashboard",
    page_icon="🛍️",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f5f9ff;
}

.block-container {
    padding-top: 1.5rem;
}

.title {
    font-size: 42px;
    font-weight: 700;
    color: #1565c0;
    text-align: center;
}

.subtitle {
    font-size: 18px;
    color: #555555;
    text-align: center;
    margin-bottom: 30px;
}

.section-title {
    color: #1565c0;
    font-size: 28px;
    font-weight: 700;
}

.card {
    padding: 25px;
    border-radius: 15px;
    background-color: #ffffff;
    color: #222222;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.15);
    margin-bottom: 15px;
}

.card h2 {
    color: #1565c0 !important;
    font-size: 24px;
    font-weight: 700;
}

.card p {
    color: #333333 !important;
    font-size: 16px;
    line-height: 1.6;
}

.metric-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #e3f2fd;
    text-align: center;
}

.recommendation {
    padding: 18px;
    margin: 10px 0px;
    border-radius: 12px;
    background-color: #e8f5e9;
    color: #222222 !important;
    border-left: 5px solid #1565c0;
    font-size: 18px;
    font-weight: 600;
}

.recommendation strong,
.recommendation span,
.recommendation p {
    color: #222222 !important;
}

.footer {
    text-align: center;
    color: #777777;
    margin-top: 40px;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RULES_FILE = os.path.join(
    BASE_DIR,
    "association_rules.pkl"
)

FREQUENT_FILE = os.path.join(
    BASE_DIR,
    "frequent_items.pkl"
)

KMEANS_FILE = os.path.join(
    BASE_DIR,
    "k-means_customer_segmentation_model.pkl"
)

DATA_FILE = os.path.join(
    BASE_DIR,
    "online_retail.csv"
)


# ============================================================
# LOAD MARKET BASKET FILES
# ============================================================

@st.cache_data
def load_market_basket():

    rules = None
    frequent_items = None

    if not os.path.exists(RULES_FILE):
        return None, None, "association_rules.pkl not found"

    if not os.path.exists(FREQUENT_FILE):
        return None, None, "frequent_items.pkl not found"

    try:

        rules = joblib.load(RULES_FILE)
        frequent_items = joblib.load(FREQUENT_FILE)

        return rules, frequent_items, None

    except Exception as e:

        return None, None, str(e)


# ============================================================
# LOAD CUSTOMER SEGMENTATION DATA
# ============================================================

@st.cache_data
def load_customer_segmentation():

    if not os.path.exists(KMEANS_FILE):

        return None, None, "k-means_customer_segmentation_model.pkl not found"

    if not os.path.exists(DATA_FILE):

        return None, None, "online_retail.csv not found"

    try:

        # Load KMeans model
        kmeans_model = joblib.load(KMEANS_FILE)

        # Load retail dataset
        retail = pd.read_csv(DATA_FILE)

        # Remove missing CustomerID
        retail = retail.dropna(
            subset=["CustomerID"]
        )

        # Convert CustomerID
        retail["CustomerID"] = retail[
            "CustomerID"
        ].astype(str)

        # Convert Quantity
        retail["Quantity"] = pd.to_numeric(
            retail["Quantity"],
            errors="coerce"
        )

        # Convert UnitPrice
        retail["UnitPrice"] = pd.to_numeric(
            retail["UnitPrice"],
            errors="coerce"
        )

        # Remove invalid values
        retail = retail.dropna(
            subset=[
                "Quantity",
                "UnitPrice"
            ]
        )

        # Keep positive values
        retail = retail[
            (retail["Quantity"] > 0) &
            (retail["UnitPrice"] > 0)
        ]

        # Create TotalAmount if not present
        if "TotalAmount" not in retail.columns:

            retail["TotalAmount"] = (
                retail["Quantity"] *
                retail["UnitPrice"]
            )

        else:

            retail["TotalAmount"] = pd.to_numeric(
                retail["TotalAmount"],
                errors="coerce"
            )

            retail["TotalAmount"] = retail[
                "TotalAmount"
            ].fillna(
                retail["Quantity"] *
                retail["UnitPrice"]
            )

        # ----------------------------------------------------
        # CUSTOMER LEVEL AGGREGATION
        # ----------------------------------------------------

        customer_df = retail.groupby(
            "CustomerID"
        ).agg(

            Quantity=(
                "Quantity",
                "sum"
            ),

            TotalAmount=(
                "TotalAmount",
                "sum"
            )

        ).reset_index()

        # ----------------------------------------------------
        # FEATURES FOR K-MEANS
        # ----------------------------------------------------

        X = customer_df[
            [
                "Quantity",
                "TotalAmount"
            ]
        ]

        # ----------------------------------------------------
        # STANDARD SCALING
        # ----------------------------------------------------

        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X)

        # ----------------------------------------------------
        # PREDICT CLUSTERS
        # ----------------------------------------------------

        customer_df["Cluster"] = (
            kmeans_model.predict(X_scaled)
        )

        return customer_df, kmeans_model, None

    except Exception as e:

        return None, None, str(e)


# ============================================================
# LOAD DATA
# ============================================================

rules, frequent_items, basket_error = load_market_basket()

customer_df, kmeans_model, customer_error = (
    load_customer_segmentation()
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🛍️ Retail Analytics")

st.sidebar.markdown(
    "### Dashboard Menu"
)

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Home",
        "🛒 Market Basket Analysis",
        "👥 Customer Segmentation"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "This dashboard combines Market Basket Analysis "
    "and Customer Segmentation using Machine Learning."
)


# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    st.markdown(
        '<div class="title">🛍️ Retail Analytics Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Market Basket Analysis + Customer Segmentation'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    # --------------------------------------------------------
    # INTRODUCTION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">📊 Project Overview</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        This project analyzes retail customer purchasing behavior
        using two Machine Learning techniques:

        **1. Market Basket Analysis**
        
        Finds products that are frequently purchased together.

        **2. Customer Segmentation**
        
        Groups customers into different segments based on their
        purchasing quantity and total spending.
        """
    )

    st.markdown("---")

    # --------------------------------------------------------
    # TWO FEATURES
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            <div class="card">

            <h2>🛒 Market Basket Analysis</h2>

            <p>
            Uses Apriori and Association Rules to discover
            relationships between products.
            </p>

            <p>
            Select one product and the dashboard will suggest
            products frequently purchased with it.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="card">

            <h2>👥 Customer Segmentation</h2>

            <p>
            Uses K-Means clustering to divide customers into
            different groups.
            </p>

            <p>
            Segmentation is based on Quantity purchased and
            Total Amount spent.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # --------------------------------------------------------
    # DATA STATUS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">📁 Project File Status</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        if rules is not None:
            st.success("✅ Association Rules")
        else:
            st.error("❌ Association Rules")

    with col2:

        if frequent_items is not None:
            st.success("✅ Frequent Items")
        else:
            st.error("❌ Frequent Items")

    with col3:

        if kmeans_model is not None:
            st.success("✅ K-Means Model")
        else:
            st.error("❌ K-Means Model")

    with col4:

        if customer_df is not None:
            st.success("✅ Retail Dataset")
        else:
            st.error("❌ Retail Dataset")

    st.markdown("---")

    st.info(
        "💡 Use the menu on the left to explore the analysis."
    )


# ============================================================
# MARKET BASKET ANALYSIS
# ============================================================

elif page == "🛒 Market Basket Analysis":

    st.markdown(
        '<div class="title">🛒 Market Basket Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Product Recommendation using Association Rules'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    if rules is None:

        st.error(
            "❌ Market Basket files could not be loaded."
        )

        st.warning(
            f"Error: {basket_error}"
        )

        st.info(
            "Make sure the following files are in the same "
            "folder as app.py:\n\n"
            "association_rules(2).pkl\n\n"
            "frequent_items(3).pkl"
        )

        st.stop()

    # --------------------------------------------------------
    # CLEAN RULES
    # --------------------------------------------------------

    rules = rules.copy()

    def convert_to_text(value):

        try:

            if isinstance(
                value,
                (set, frozenset, list, tuple)
            ):

                return ", ".join(
                    sorted(
                        [str(x) for x in value]
                    )
                )

            return str(value)

        except:

            return str(value)

    rules["Antecedents_Text"] = rules[
        "antecedents"
    ].apply(convert_to_text)

    rules["Consequents_Text"] = rules[
        "consequents"
    ].apply(convert_to_text)

    # --------------------------------------------------------
    # CREATE PRODUCT LIST
    # --------------------------------------------------------

    products = set()

    for item in rules["antecedents"]:

        if isinstance(
            item,
            (set, frozenset, list, tuple)
        ):

            for product in item:

                products.add(str(product))

        else:

            products.add(str(item))

    for item in rules["consequents"]:

        if isinstance(
            item,
            (set, frozenset, list, tuple)
        ):

            for product in item:

                products.add(str(product))

        else:

            products.add(str(item))

    products = sorted(products)

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "📦 Total Products",
            len(products)
        )

    with col2:

        st.metric(
            "🔗 Association Rules",
            len(rules)
        )

    with col3:

        if "lift" in rules.columns:

            st.metric(
                "📈 Highest Lift",
                round(
                    rules["lift"].max(),
                    2
                )
            )

    st.markdown("---")

    # --------------------------------------------------------
    # PRODUCT SELECTION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        '🔍 Product Recommendation'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Select one product to find products frequently "
        "purchased with it."
    )

    selected_product = st.selectbox(
        "Select Product",
        products
    )

    # --------------------------------------------------------
    # FIND RECOMMENDATIONS
    # --------------------------------------------------------

    recommendations = []

    for _, row in rules.iterrows():

        antecedent = row["antecedents"]

        consequent = row["consequents"]

        try:

            if selected_product in antecedent:

                for item in consequent:

                    recommendations.append(
                        {
                            "Product": str(item),
                            "Support": row["support"],
                            "Confidence": row["confidence"],
                            "Lift": row["lift"]
                        }
                    )

        except:

            pass

    # --------------------------------------------------------
    # RECOMMENDATION RESULTS
    # --------------------------------------------------------

    if len(recommendations) == 0:

        st.warning(
            "No recommendation was found for this product."
        )

    else:

        recommendation_df = pd.DataFrame(
            recommendations
        )

        # Remove duplicate products
        recommendation_df = (
            recommendation_df
            .sort_values(
                by="Lift",
                ascending=False
            )
            .drop_duplicates(
                subset=["Product"]
            )
        )

        st.success(
            f"🎯 Recommended products for "
            f"**{selected_product}**"
        )

        # ----------------------------------------------------
        # TOP 5 RECOMMENDATIONS
        # ----------------------------------------------------

        top_recommendations = (
            recommendation_df.head(5)
        )

        cols = st.columns(
            len(top_recommendations)
        )

        for i, (_, row) in enumerate(
            top_recommendations.iterrows()
        ):

            with cols[i]:

                st.markdown(
                    f"""
                    <div class="recommendation">

                    🛍️ {row["Product"]}

                    <br><br>

                    📈 Lift: {row["Lift"]:.2f}

                    <br>

                    🎯 Confidence:
                    {row["Confidence"]:.2%}

                    <br>

                    📊 Support:
                    {row["Support"]:.2%}

                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.markdown("---")

        # ----------------------------------------------------
        # RECOMMENDATION TABLE
        # ----------------------------------------------------

        st.subheader(
            "📋 Recommendation Details"
        )

        display_recommendations = (
            recommendation_df.copy()
        )

        display_recommendations[
            "Support"
        ] = display_recommendations[
            "Support"
        ].round(4)

        display_recommendations[
            "Confidence"
        ] = display_recommendations[
            "Confidence"
        ].round(4)

        display_recommendations[
            "Lift"
        ] = display_recommendations[
            "Lift"
        ].round(2)

        st.dataframe(
            display_recommendations,
            use_container_width=True
        )

        # ----------------------------------------------------
        # LIFT CHART
        # ----------------------------------------------------

        st.subheader(
            "📈 Recommendation by Lift"
        )

        chart_data = (
            recommendation_df
            .head(10)
            .sort_values(
                "Lift",
                ascending=True
            )
        )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.barh(
            chart_data["Product"],
            chart_data["Lift"]
        )

        ax.set_xlabel(
            "Lift"
        )

        ax.set_ylabel(
            "Product"
        )

        ax.set_title(
            f"Top Recommendations for {selected_product}"
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

    st.markdown("---")

    # --------------------------------------------------------
    # TOP ASSOCIATION RULES
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        '🏆 Top Association Rules'
        '</div>',
        unsafe_allow_html=True
    )

    top_rules = rules.copy()

    if "lift" in top_rules.columns:

        top_rules = top_rules.sort_values(
            by="lift",
            ascending=False
        )

    top_rules = top_rules.head(20)

    display_rules = pd.DataFrame()

    display_rules[
        "Antecedent"
    ] = top_rules[
        "Antecedents_Text"
    ]

    display_rules[
        "Consequent"
    ] = top_rules[
        "Consequents_Text"
    ]

    if "support" in top_rules.columns:

        display_rules[
            "Support"
        ] = top_rules[
            "support"
        ].round(4)

    if "confidence" in top_rules.columns:

        display_rules[
            "Confidence"
        ] = top_rules[
            "confidence"
        ].round(4)

    if "lift" in top_rules.columns:

        display_rules[
            "Lift"
        ] = top_rules[
            "lift"
        ].round(2)

    st.dataframe(
        display_rules,
        use_container_width=True
    )

    # --------------------------------------------------------
    # EXPLANATION
    # --------------------------------------------------------

    st.markdown("---")

    st.markdown(
        '<div class="section-title">'
        '📚 Association Rule Metrics'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.info(
            """
            **Support**

            Shows how frequently a product combination
            appears in all transactions.
            """
        )

    with col2:

        st.info(
            """
            **Confidence**

            Shows how often the consequent product is
            purchased when the selected product is purchased.
            """
        )

    with col3:

        st.info(
            """
            **Lift**

            Shows how strongly two products are associated.

            Higher lift generally indicates a stronger
            purchasing relationship.
            """
        )


# ============================================================
# CUSTOMER SEGMENTATION
# ============================================================

elif page == "👥 Customer Segmentation":

    st.markdown(
        '<div class="title">👥 Customer Segmentation</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Customer Groups using K-Means Clustering'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    # --------------------------------------------------------
    # CHECK DATA
    # --------------------------------------------------------

    if customer_df is None:

        st.error(
            "❌ Customer Segmentation files could not be loaded."
        )

        st.warning(
            f"Error: {customer_error}"
        )

        st.info(
            "Make sure these files are in the same folder "
            "as app.py:\n\n"
            "k-means_customer_segmentation_model(3).pkl\n\n"
            "online_retail (1)(1).csv"
        )

        st.stop()

    # --------------------------------------------------------
    # BASIC INFORMATION
    # --------------------------------------------------------

    total_customers = (
        customer_df["CustomerID"]
        .nunique()
    )

    total_quantity = (
        customer_df["Quantity"]
        .sum()
    )

    total_sales = (
        customer_df["TotalAmount"]
        .sum()
    )

    number_clusters = (
        customer_df["Cluster"]
        .nunique()
    )

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "👥 Total Customers",
            f"{total_customers:,}"
        )

    with col2:

        st.metric(
            "📦 Total Quantity",
            f"{total_quantity:,.0f}"
        )

    with col3:

        st.metric(
            "💰 Total Sales",
            f"{total_sales:,.2f}"
        )

    with col4:

        st.metric(
            "🎯 Number of Clusters",
            number_clusters
        )

    st.markdown("---")

    # ========================================================
    # CLUSTER DISTRIBUTION
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '📊 Customer Distribution by Cluster'
        '</div>',
        unsafe_allow_html=True
    )

    cluster_counts = (
        customer_df["Cluster"]
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.bar(
        cluster_counts.index.astype(str),
        cluster_counts.values
    )

    ax.set_xlabel(
        "Cluster"
    )

    ax.set_ylabel(
        "Number of Customers"
    )

    ax.set_title(
        "Customers in Each Cluster"
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    # ========================================================
    # AVERAGE SPENDING
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '💰 Average Spending by Cluster'
        '</div>',
        unsafe_allow_html=True
    )

    avg_spending = (
        customer_df
        .groupby("Cluster")["TotalAmount"]
        .mean()
        .sort_index()
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.bar(
        avg_spending.index.astype(str),
        avg_spending.values
    )

    ax.set_xlabel(
        "Cluster"
    )

    ax.set_ylabel(
        "Average Spending"
    )

    ax.set_title(
        "Average Customer Spending by Cluster"
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    # ========================================================
    # QUANTITY VS TOTAL AMOUNT
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '📦 Quantity vs Total Amount'
        '</div>',
        unsafe_allow_html=True
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    clusters = sorted(
        customer_df["Cluster"].unique()
    )

    for cluster in clusters:

        data = customer_df[
            customer_df["Cluster"] == cluster
        ]

        ax.scatter(
            data["Quantity"],
            data["TotalAmount"],
            label=f"Cluster {cluster}",
            alpha=0.6
        )

    ax.set_xlabel(
        "Quantity Purchased"
    )

    ax.set_ylabel(
        "Total Amount Spent"
    )

    ax.set_title(
        "Customer Segmentation"
    )

    ax.legend()

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    # ========================================================
    # CLUSTER SUMMARY
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '📋 Cluster Summary'
        '</div>',
        unsafe_allow_html=True
    )

    cluster_summary = (
        customer_df
        .groupby("Cluster")
        .agg(
            Customers=(
                "CustomerID",
                "count"
            ),

            Total_Quantity=(
                "Quantity",
                "sum"
            ),

            Total_Sales=(
                "TotalAmount",
                "sum"
            ),

            Avg_Quantity=(
                "Quantity",
                "mean"
            ),

            Avg_Spending=(
                "TotalAmount",
                "mean"
            )
        )
        .reset_index()
    )

    cluster_summary[
        [
            "Total_Quantity",
            "Total_Sales",
            "Avg_Quantity",
            "Avg_Spending"
        ]
    ] = cluster_summary[
        [
            "Total_Quantity",
            "Total_Sales",
            "Avg_Quantity",
            "Avg_Spending"
        ]
    ].round(2)

    st.dataframe(
        cluster_summary,
        use_container_width=True
    )

    # ========================================================
    # SELECT CLUSTER
    # ========================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">'
        '🔎 Explore a Customer Segment'
        '</div>',
        unsafe_allow_html=True
    )

    selected_cluster = st.selectbox(
        "Select Cluster",
        sorted(
            customer_df["Cluster"]
            .unique()
        )
    )

    selected_data = customer_df[
        customer_df["Cluster"] ==
        selected_cluster
    ]

    # ========================================================
    # SELECTED CLUSTER METRICS
    # ========================================================

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "👥 Customers",
            len(selected_data)
        )

    with col2:

        st.metric(
            "📦 Average Quantity",
            f"{selected_data['Quantity'].mean():,.2f}"
        )

    with col3:

        st.metric(
            "💰 Average Spending",
            f"{selected_data['TotalAmount'].mean():,.2f}"
        )

    # ========================================================
    # CUSTOMER TABLE
    # ========================================================

    st.subheader(
        f"👥 Customers in Cluster {selected_cluster}"
    )

    customer_display = selected_data[
        [
            "CustomerID",
            "Quantity",
            "TotalAmount",
            "Cluster"
        ]
    ].copy()

    customer_display[
        "Quantity"
    ] = customer_display[
        "Quantity"
    ].round(2)

    customer_display[
        "TotalAmount"
    ] = customer_display[
        "TotalAmount"
    ].round(2)

    st.dataframe(
        customer_display,
        use_container_width=True
    )

    # ========================================================
    # SEGMENT INTERPRETATION
    # ========================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">'
        '💡 Customer Segment Interpretation'
        '</div>',
        unsafe_allow_html=True
    )

    overall_avg_quantity = (
        customer_df["Quantity"].mean()
    )

    overall_avg_spending = (
        customer_df["TotalAmount"].mean()
    )

    cluster_avg_quantity = (
        selected_data["Quantity"].mean()
    )

    cluster_avg_spending = (
        selected_data["TotalAmount"].mean()
    )

    # Determine customer segment
    if (
        cluster_avg_quantity > overall_avg_quantity
        and
        cluster_avg_spending > overall_avg_spending
    ):

        segment_name = "High-Value Customers"

        explanation = (
            "These customers purchase larger quantities "
            "and spend more than the average customer."
        )

    elif (
        cluster_avg_quantity < overall_avg_quantity
        and
        cluster_avg_spending < overall_avg_spending
    ):

        segment_name = "Low-Value Customers"

        explanation = (
            "These customers purchase smaller quantities "
            "and spend less than the average customer."
        )

    elif (
        cluster_avg_quantity > overall_avg_quantity
        and
        cluster_avg_spending <= overall_avg_spending
    ):

        segment_name = "High-Quantity Customers"

        explanation = (
            "These customers purchase larger quantities "
            "but their average spending is not very high."
        )

    else:

        segment_name = "Moderate-Value Customers"

        explanation = (
            "These customers have a moderate purchasing "
            "quantity and spending pattern."
        )

    st.success(
        f"🎯 Cluster {selected_cluster}: "
        f"**{segment_name}**"
    )

    st.write(
        explanation
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

    🛍️ Retail Analytics Dashboard  
    <br>
    Market Basket Analysis + Customer Segmentation  
    <br><br>
    Built using Python, Streamlit, Apriori and K-Means

    </div>
    """,
    unsafe_allow_html=True
)