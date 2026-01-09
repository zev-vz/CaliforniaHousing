# 🏠 California Housing Market Dashboard

A modern, interactive dashboard built with Dash and Plotly to explore the 1990 California Census housing dataset. Visualize housing characteristics, analyze feature correlations, and generate AI-powered house value predictions using multiple machine learning models.

https://californiahousing-u67d.onrender.com/

## 🎯 Executive Summary

This dashboard allows users to:

- Explore California housing data across 20,640 census block groups
- Analyze key housing metrics: median house value, income, house age, and more
- Visualize feature distributions, correlations, and scatter plots
- Generate predictions for house values using:
  - Gradient Boosting
  - Ridge
  - Lasso
  - Linear Regression

## 📊 Dataset Overview

**Source:** California Housing dataset (from sklearn.datasets)  
**Coverage:** Census data across California block groups, 1990

### Key Variables

| Variable | Description |
|----------|-------------|
| MedHouseVal | Median house value (hundreds of thousands of dollars) |
| MedInc | Median income of households (tens of thousands of dollars) |
| HouseAge | Median age of houses in the block group |
| AveRooms | Average rooms per household |
| AveBedrms | Average bedrooms per household |
| Population | Total population in the block group |

### Derived Features

- **PriceCategory** (Budget, Mid-range, Expensive, Luxury)
- **RoomsPerPerson** (AveRooms / AveOccup)
- **IncomeLevel** (Low, Medium, High, Very High)

## ⚙️ Dashboard Features

### Interactive Controls

- **Color Map By:** Median House Value, Income, House Age, Population Density, or Average Rooms
- **Income Level Filter:** Low, Medium, High, Very High
- **Price Range Filter:** Adjustable with slider

### Visualizations

- **Map View:** Zoomable California map showing house values and other features
- **Distribution & Correlation:** Histograms and correlation heatmaps
- **Scatter & Feature Comparison:** Income vs. house value scatter plots and income distribution by price category
- **Predictive Analytics:** Input house characteristics to predict values using multiple ML models with visualization

## 🔬 Predictive Modeling

- **Models Included:** Gradient Boosting, Ridge, Lasso, Linear Regression
- **Features Used for Prediction:** HouseAge, AveRooms, AveBedrms
- **Prediction Output:** Estimated house value with visualization against market distribution
- **Confidence Handling:** Missing inputs automatically filled with median values

## 🏆 Key Insights

- **Price Drivers:** Income, house age, and number of rooms/bedrooms are the strongest predictors
- **Correlation Patterns:** Income correlates positively with median house value
- **Distribution Patterns:** Most houses fall in the mid-range pricing category
