#Libraries

import pandas as pd
import datetime as dt
from sklearn.preprocessing import StandardScaler

#Load Dataset

df = pd.read_csv("Cleaned_Dataset.Csv")

#Function for Feature Engineering

def feature_engineering(df):
    
    # Ensure InvoiceDate is datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Create Revenue
    df['Revenue'] = df['Quantity'] * df['UnitPrice']
    
    # Snapshot date (one day after last transaction)
    snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
    
    # Create RFM
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'Revenue': 'sum'
    })
    
    rfm.rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'Revenue': 'Monetary'
    }, inplace=True)
    
    # Behavioral features
    rfm['Avg_Order_Value'] = rfm['Monetary'] / rfm['Frequency']
    rfm['Purchase_Rate'] = rfm['Frequency'] / rfm['Recency']
    
    return rfm

def scale_features(rfm):
    scaler = StandardScaler()
    
    rfm_numeric = rfm.select_dtypes(include=['int64', 'float64'])
    
    scaled_data = scaler.fit_transform(rfm_numeric)
    
    rfm_scaled = pd.DataFrame(
        scaled_data,
        columns=rfm_numeric.columns,
        index=rfm.index
    )
    
    return rfm_scaled

# Derived Ratios
def add_derived_ratios(rfm):

    rfm = rfm.copy()

    # Avoid division by zero
    rfm['Avg_Order_Value'] = rfm['Monetary'] / (rfm['Frequency'] + 1e-6)
    rfm['Purchase_Rate'] = rfm['Frequency'] / (rfm['Recency'] + 1e-6)

    # Additional advanced ratios
    rfm['Value_per_Day'] = rfm['Monetary'] / (rfm['Recency'] + 1e-6)
    rfm['Engagement_Score'] = rfm['Frequency'] * rfm['Monetary']

    return rfm

behavioural_features = feature_engineering(df)
scaled_features = scale_features(behavioural_features)
rfm = add_derived_ratios(behavioural_features)
print("Behavioural:",behavioural_features,"\n")
print("Scaled:",scaled_features)
print(rfm.head())

#Advanced Features 

#1. Customer Lifetime Value
def customer_lifetime_value_segmentation(rfm):

    rfm = rfm.copy()

    # Simple CLV formula
    rfm['CLV'] = rfm['Monetary'] * rfm['Frequency']

    # segments based on CLV
    rfm['CLV_Segment'] = pd.qcut(
        rfm['CLV'],
        q=4,
        labels=['Low Value', 'Medium Value', 'High Value', 'Premium']
    )

    return rfm

#2. Time Based 
def time_based_segmentation(rfm):

    rfm = rfm.copy()

    rfm['Customer_Type'] = pd.cut(
        rfm['Recency'],
        bins=[-1, 30, 90, 180, 3650],
        labels=['Active', 'Warm', 'Cold', 'Lost']
    )

    return rfm


rfm = customer_lifetime_value_segmentation(rfm)

print(rfm[['CLV', 'CLV_Segment']].head())

rfm = time_based_segmentation(rfm)

print(rfm[['Recency', 'Customer_Type']].head())