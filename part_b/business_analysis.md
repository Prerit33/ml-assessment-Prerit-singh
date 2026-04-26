### B1. Problem Formulation

**(a) Machine Learning Problem Formulation**

* **Target Variable:** The total number of items sold (sales volume) at a specific store during the one-month promotion period.
* **Candidate Input Features:**
    * *Store Attributes:* Store size (sq ft), location type (urban, semi-urban, rural), local competition density (e.g., number of competitors within a 5-mile radius).
    * *Baseline Metrics:* Average monthly footfall, historical baseline sales volume (without promotions).
    * *Customer Demographics:* Average local income, primary age demographic, loyalty program penetration rate at that specific store.
    * *Promotion/Context Variables:* The specific promotion deployed (Categorical: Flat Discount, BOGO, etc.), and the month/season (to account for holiday spikes or seasonal lulls).
* **Problem Type:** **Supervised Regression**. 
    * *Justification:* Because the goal is to predict a continuous numerical value (number of items sold), regression is the correct approach. In practice, we would input a store's features into the model, simulate the prediction for all five promotion types, and deploy the promotion that yields the highest predicted output. *(Note: As the team matures, this could evolve into **Uplift Modeling**, which specifically predicts the incremental increase in sales caused by the treatment/promotion rather than just total sales).*

**(b) Target Variable Selection: Items Sold vs. Revenue**

* **Why Items Sold is More Reliable:** Revenue is inherently confounded by the mechanics of the promotions themselves. Revenue is $Price \times Volume$. A "Flat Discount" or "BOGO" drastically reduces the price per item. A highly successful BOGO promotion might double the number of items sold, but because the effective price per item drops by 50%, the total revenue might remain completely flat. If we train a model on revenue, it will penalize aggressive discount promotions even if they successfully drove massive customer engagement and cleared inventory. 
* **The Broader Principle:** This illustrates the principle of **Business Objective Alignment** (or construct validity) in ML design. The target variable must represent the *true* behavioral outcome the business wants to optimize (moving inventory/driving volume) and be isolated from mathematical artifacts or internal pricing levers that distort the result.

**(c) Modeling Strategy: Beyond the Single Global Model**

* **Critique of a Global Model:** A single global model across all 50 stores risks "underfitting" the geographical nuances. Urban stores with high footfall might dominate the dataset, causing the model to learn what works best for city shoppers while completely misinterpreting the behavior of rural shoppers. 
* **Proposed Alternative: Cluster-Based (Segmented) Modeling.**
    * *The Strategy:* Instead of one model, group the stores into distinct clusters based on their underlying characteristics (e.g., location type, competition density, demographics) using an unsupervised algorithm like K-Means clustering. Then, train a separate regression model for each cluster.
    * *Justification:* This strikes the perfect balance between highly localized accuracy and data volume. Training 50 separate models (one for each store) would lead to overfitting due to a lack of data per store. However, training 3 to 5 cluster-specific models (e.g., an "Urban High-Competition" model and a "Rural Low-Competition" model) ensures that the machine learning algorithm picks up on the specific promotional sensitivities of those customer segments without the signals getting drowned out by the rest of the chain.
 
Here is a comprehensive strategy for preparing and exploring the data to solve the promotion effectiveness problem.

### B2. Data and EDA Strategy

**(a) Data Integration, Grain, and Aggregations**

* **Joining Strategy:** 1.  **Aggregate Transactions:** First, roll up the granular `transactions` table (which is likely at the receipt/item level) to the desired time/location level. 
    2.  **Join the Rest:** Once aggregated, we will left-join the `store_attributes` using `store_id`. We will join the `promotion_details` using `store_id` and the specific `month_year` the promotion was active. Finally, join the aggregated `calendar` features using the `month_year`.
* **The Final Grain:** The grain of the final modeling dataset should be **Store-Month** (one row = one specific store in one specific month). This matches the business decision cycle ("determine which promotion should be deployed in each store each month").
* **Key Aggregations (from daily/transactional to monthly):**
    * *Transactions:* Sum of quantity (`items_sold` - target variable), sum of revenue, count of unique transactions (footfall proxy).
    * *Calendar:* Count of weekend days in that month, count of festival/holiday days in that month.
    * *Promotion Details:* If promotions don't perfectly align with the 1st to 30th of a month, we might need an aggregation like "percentage of days in the month under promotion X."

**(b) Exploratory Data Analysis (EDA)**

Before training models, I would execute these four analyses:

1.  **Distribution of Target Variable (Items Sold) by Store Size/Type**
    * *Chart:* Overlaid histograms or boxplots of `items_sold`, faceted by `location_type` (urban, semi-urban, rural).
    * *What to look for:* Extreme outliers or heavy skewness. Are urban stores selling 10x more than rural stores? 
    * *Influence on Modeling:* If the volume variance is massive, an absolute error metric (like RMSE) will focus the model entirely on the biggest stores. This would justify log-transforming the target variable or switching the target to a relative metric, like "percentage increase over baseline."
2.  **Average Items Sold vs. Promotion Type**
    * *Chart:* Bar chart showing the mean/median `items_sold` for each of the 5 promotion types, plus the "No Promotion" baseline.
    * *What to look for:* Which promotion performs best on average? Are there promotions that perform worse than having no promotion at all?
    * *Influence on Feature Engineering:* Establishes a global baseline. If two promotions (e.g., Category Offer and Flat Discount) yield nearly identical volume curves, we might consider combining them into a single category if data is sparse.
3.  **Promotion Effectiveness by Location (Interaction Analysis)**
    * *Chart:* A heatmap or grouped bar chart showing the average `items_sold` on the Y-axis, grouped by `location_type` on the X-axis, with different colors for each `promotion_type`.
    * *What to look for:* Interaction effects. Does BOGO cause a massive spike in Urban stores but fall flat in Rural ones? Does Loyalty Bonus only work in Semi-Urban areas?
    * *Influence on Modeling:* If strong interactions exist, this strictly validates the "Cluster-Based Modeling" strategy discussed in B1. If a single model is used instead, this dictates that explicit interaction features (e.g., `is_Urban * is_BOGO`) must be engineered.
4.  **Time Series & Calendar Effects**
    * *Chart:* A line chart of total chain-wide `items_sold` over time (e.g., the last 24 months), with vertical shaded regions highlighting festival months.
    * *What to look for:* Strong seasonality (e.g., summer lulls, winter spikes) and the isolated impact of festivals. 
    * *Influence on Feature Engineering:* If festivals dictate the majority of sales, calendar features will be the most important inputs. We will need to engineer lag features (e.g., `sales_volume_last_month` or `sales_volume_same_month_last_year`) to give the model a rolling baseline.

**(c) Addressing the 80% "No Promotion" Imbalance**

* **The Effect on the Model:** Since 80% of the dataset represents business-as-usual (no promotion), the algorithm will dedicate the vast majority of its learning capacity to predicting baseline sales. It will become highly accurate at guessing what a store sells on a normal Tuesday, but will severely under-learn the subtle nuances of how the 5 specific promotions behave. It treats the promotions as edge cases rather than the core focus of the analysis.
* **Steps to Address It:**
    1.  **Sample Weighting:** During model training (especially if using algorithms like XGBoost or Random Forest), assign higher weights to the rows where a promotion was active. For example, give promotional rows a weight of 4 and non-promotional rows a weight of 1. This forces the model to pay equal attention to the promotional patterns.
    2.  **Two-Stage Modeling (Baseline + Lift):** * *Stage 1:* Train a model on *all* the data to predict "Baseline Sales" (what a store should sell given the month, weather, and location, assuming no promotion).
        * *Stage 2:* Filter the dataset to *only* the 20% of rows where a promotion occurred. Train a second model to predict the **Lift** (Actual Items Sold / Predicted Baseline Sales). This directly isolates the promotion's effect from the store's normal performance, completely bypassing the imbalance issue.


Here is the breakdown for evaluating, explaining, and deploying the promotion effectiveness model.

### B3. Model Evaluation and Deployment

**(a) Train-Test Split Strategy and Metrics**

* **The Split Strategy (Out-of-Time Validation):** Because this is panel data (cross-sectional store data over a time series), we must use a time-based split. For example, use Month 1 through Month 24 (Years 1 and 2) for training, Months 25 through 30 for validation (hyperparameter tuning), and Months 31 through 36 (Year 3) as the final holdout test set.
* **Why a Random Split is Inappropriate:** A random split causes **data leakage**. If we randomly shuffle the rows, the model might train on data from December 2023 and use it to predict November 2023. In the real world, we cannot use future knowledge to predict the past. Retail data is also heavily autocorrelated and seasonal; random splitting destroys the temporal continuity the model needs to learn real-world forecasting.
* **Evaluation Metrics:**
    * **Mean Absolute Error (MAE):** This represents the average absolute difference between the predicted items sold and the actual items sold. 
        * *Interpretation:* "On average, our model's prediction is off by +/- 50 items per store per month." This is highly intuitive for business stakeholders.
    * **Weighted Mean Absolute Percentage Error (WMAPE):** * *Interpretation:* Because we have stores of varying sizes (urban vs. rural), a 50-item error is disastrous for a small store but negligible for a massive one. WMAPE scales the error relative to the total volume, giving we an aggregate percentage error (e.g., "Our predictions are 8% off overall") without being overly skewed by low-volume stores.
    * **Root Mean Squared Error (RMSE):** * *Interpretation:* RMSE penalizes large errors heavily. If wildly overestimating sales leads the business to overstock perishable or highly seasonal fashion items (causing massive financial loss), we would use RMSE to train a model that avoids catastrophic misses.

**(b) Investigating Feature Importance for Differing Recommendations**

To explain why the model suggested different promotions for Store 12 in December versus March, I would use **Local Feature Importance** techniques, specifically **SHAP (SHapley Additive exPlanations)** values. 

* **The Investigation:** Global feature importance tells us what drives the model *overall*, but SHAP allows us to break down a *single, specific prediction*. I would generate a SHAP waterfall plot for Store 12's December prediction and another for its March prediction. 
* **The Communication:** I would show the marketing team how the baseline prediction is modified by specific local features. For example:
    * "In December, the model sees that `Month = Dec` (holiday shopping) and `Historical_Footfall = High` strongly push the Loyalty Points Bonus up. Shoppers are already buying multiple gifts, so offering points incentivizes them to consolidate all their holiday shopping at our store."
    * "In March, the `Month = March` feature pushes Loyalty down, but `Competitor_Density = High` pushes the Flat Discount up. The model learned that during slow, off-season months, shoppers are highly price-sensitive and easily swayed by local competitors, making an immediate Flat Discount the only way to drive volume."

**(c) End-to-End Deployment and Monitoring Process**

To operationalize this model for monthly batch predictions, the pipeline would look like this:

**1. Model Serialization and Storage**
Once trained and validated, the final model object (along with any pre-processing steps like scalers or one-hot encoders) is serialized (e.g., saved as a Pickle or ONNX file) and stored in a secure model registry (like MLflow, AWS SageMaker, or a cloud storage bucket). 

**2. The Monthly Inference Pipeline**
* **Data Preparation (ETL):** On the 1st of every month, an automated script (e.g., orchestrated by Apache Airflow) runs a SQL query to gather the latest data. It fetches the previous month's rolling footfall, the current store attributes, and the calendar flags for the *upcoming* month.
* **Simulation Matrix:** For each of the 50 stores, the script generates 5 distinct rows—one for each possible promotion type—resulting in a 250-row prediction dataset.
* **Scoring & Recommendation:** The script loads the saved model and predicts the `items_sold` for all 250 rows. It then groups the results by store, identifies which of the 5 promotions yielded the highest predicted volume, and writes those 50 winning recommendations to a database connected to the marketing team's dashboard.

**3. Monitoring and Retraining**
* **Data Drift Monitoring:** Track the incoming feature distributions. If a new competitor opens near 10 stores, or if a specific store's footfall drops by 40% due to construction, an alert should trigger indicating the input data no longer matches the training data.
* **Concept Drift (Performance) Monitoring:** At the end of every month, when the actual `items_sold` data arrives, the system will automatically calculate the MAE and WMAPE of the model's predictions from the start of the month.
* **Retraining Trigger:** If the error metric exceeds a predefined threshold (e.g., WMAPE rises above 15% for two consecutive months), an automated retraining pipeline is triggered. This pipeline will pull the most recent historical data, retrain the model, evaluate it against the old model, and deploy the new version if it performs better.
