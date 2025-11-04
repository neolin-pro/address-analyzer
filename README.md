# Address Analyzer

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-green)
![Healthcare](https://img.shields.io/badge/Healthcare-Analytics-red)

A comprehensive Streamlit-based web application for analyzing patients' address data. This tool provides powerful address validation, geocoding, descriptive analytics, visualization, and machine learning capabilities for healthcare address intelligence.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Application Modules](#application-modules)
- [Data Processing Pipeline](#data-processing-pipeline)
- [API Integrations](#api-integrations)
- [Machine Learning Models](#machine-learning-models)
- [Technologies](#technologies)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Sample Use Cases](#sample-use-cases)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

## 🎯 Overview

The BC Address Analyzer is a specialized healthcare analytics platform designed to process, validate, and analyze patient address data for precariously housed populations in British Columbia. The application combines:

- **Address Validation**: Canada Post AddressComplete API integration for address standardization
- **Geocoding**: Google Maps API for latitude/longitude coordinate generation
- **Healthcare Analytics**: ED facility matching, move frequency analysis, and patient tracking
- **Machine Learning**: Predictive models for patient behavior and risk assessment
- **Interactive Visualization**: Comprehensive charts and plots for data exploration

This tool helps healthcare providers understand patient mobility patterns, identify unstable housing situations, and improve care coordination.

## ✨ Key Features

### 🔧 Data Processing
- **Multi-file CSV/TSV Upload**: Drag-and-drop interface for multiple datasets
- **Address Elementization**: Automated parsing of addresses into standardized components
  - Building number, street name, street type, city, province, postal code
  - Canada Post API integration with fallback parsing
- **Geocoding**: Postal code to latitude/longitude conversion via Google Maps API
- **Data Merging**: Intelligent merge of emergency visit and discharge records
- **Date-based Operations**: Calculate days between visits, chronological sorting

### 📊 Descriptive Analytics
- **Move Frequency Analysis**: Track patient address changes over time
- **Missing Data Detection**: Identify incomplete address records
- **Residential vs Commercial Classification**: Distinguish address types
- **ED Facility Matching**: Two-tier matching (ED facilities + Long-Term Care)
- **Distance Calculations**: Compute distances between address changes
- **Shared Address Detection**: Identify multiple patients at same location

### 📈 Visualization
- **10+ Chart Types**: Bar, line, scatter, box, histogram, pie, heatmap, count plots
- **Interactive Plotting**: Seaborn and Matplotlib integration
- **Distribution Analysis**: Density plots and pair plots
- **Correlation Heatmaps**: Feature relationship visualization

### 🤖 Machine Learning
- **Multiple Algorithms**: 
  - Classification: Logistic Regression, Random Forest, XGBoost, CatBoost, LightGBM
  - Regression: Random Forest, XGBoost, CatBoost, LightGBM
- **Hyperparameter Tuning**: Grid Search CV for optimal parameters
- **Automated Preprocessing**: StandardScaler, OneHotEncoder, missing value imputation
- **Performance Metrics**:
  - Classification: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
  - Regression: R², MSE, MAE, RMSE, Residual Plots
- **Feature Importance**: Visual ranking of predictive features

### 🎨 User Interface
- **Light/Dark Theme Toggle**: Customizable UI appearance
- **5 Tabbed Interface**: Upload, Preprocessing, Descriptive, Visualization, Modeling
- **Session State Management**: Persistent data across interactions
- **Real-time Progress Indicators**: Loading bars and status messages
- **Responsive Design**: Wide layout optimized for data tables

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Web Interface                   │
│  ┌──────────┬──────────────┬────────────┬─────────┬─────┐  │
│  │ Upload   │ Preprocessing │ Descriptive│ Visualiz│ ML  │  │
│  └──────────┴──────────────┴────────────┴─────────┴─────┘  │
└────────────────────┬────────────────────────────────────────┘
                     │
         ┌───────────┼───────────┐
         ▼           ▼           ▼
    ┌────────┐  ┌────────┐  ┌──────────┐
    │Canada  │  │Google  │  │ML Models │
    │Post API│  │Maps API│  │ Engine   │
    └────────┘  └────────┘  └──────────┘
         │           │           │
         └───────────┴───────────┘
                     │
         ┌───────────┴────────────┐
         ▼                        ▼
    ┌────────────┐         ┌────────────┐
    │ Validated  │         │ Predictions│
    │  Addresses │         │  & Insights│
    └────────────┘         └────────────┘
```

## 🚀 Installation

### Prerequisites

- **Python**: 3.8 or higher
- **API Keys**:
  - Canada Post AddressComplete API key (for address validation)
  - Google Maps Geocoding API key (for coordinate generation)

### Step 1: Clone the Repository

```bash
git clone https://github.com/nayzawlin/address-analyzer.git
cd address-analyzer
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv

# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure API Keys

Edit the `address_analyzer.py` file and add your API keys:

```python
# Line ~181: Google Maps API Key
API_KEY = 'YOUR_GOOGLE_MAPS_API_KEY'

# Line ~491: Canada Post API Key
api_key = "YOUR_CANADA_POST_API_KEY"
```

**Note**: Keep your API keys secure and never commit them to version control.

### Step 5: Prepare Logo (Optional)

Place a `logo.png` file in the project root directory for sidebar branding.

## 💻 Usage

### Starting the Application

Run the Streamlit app from the project directory:

```bash
streamlit run address_analyzer.py
```

The application will open in your default web browser at `http://localhost:8501`

### Basic Workflow

#### 1. **Upload Data** (Tab 1)
- Drag and drop CSV/TSV files containing patient address data
- Multiple files supported (automatically named Table1, Table2, etc.)
- View uploaded data preview

#### 2. **Data Preprocessing** (Tab 2)

**Address Elementization:**
```
Input: "1666 W 75th Ave, Vancouver, BC V6P 6G2"
↓
Output Columns:
- E_SubBuilding: (unit number if present)
- E_BuildingNumber: 1666
- E_Street: W 75th
- E_StreetType: Ave
- E_City: Vancouver
- E_ProvinceCode: BC
- E_PostalCode: V6P 6G2
```

**Geocoding:**
- Convert postal codes to latitude/longitude
- Auto-save results to new table with coordinates

**Merge Operations:**
- Merge emergency visit and discharge records
- Date-based matching with configurable tolerance
- Calculate days between events

#### 3. **Descriptive Analysis** (Tab 3)

Available analyses:
- **Move Frequency**: Number of address changes per patient
- **Missing Fields**: Incomplete address detection
- **Residential vs Commercial**: Address type classification
- **ED Facility Matching**: Postal code matching with facility lists
- **Distance Moved**: Kilometers between address changes (requires date column)
- **Shared Addresses**: Count patients at same location

#### 4. **Visualization** (Tab 4)

Create charts with customizable options:
- Select table and columns
- Choose from 10 plot types
- Interactive parameter configuration
- Export-ready high-quality plots

#### 5. **Machine Learning** (Tab 5)

Build predictive models:
1. Select table and target variable
2. Choose features (independent variables)
3. Auto-detect problem type (classification/regression)
4. Select ML algorithm
5. Configure hyperparameters
6. Train with optional Grid Search
7. View performance metrics and visualizations

## 📂 Application Modules

### Module 1: Upload & Session Management
- Multi-file uploader with validation
- Session state persistence
- Original table backup system
- Reset functionality

### Module 2: Address Processing
- **Canada Post Integration**: 
  - `find_address()`: Search for address suggestions
  - `retrieve_full_address()`: Get complete parsed address
  - `parse_address_fallback()`: Regex-based backup parser
- **Geocoding**: 
  - `get_lat_lon()`: Google Maps API wrapper
  - Rate limiting (0.2s delay between requests)

### Module 3: Date Utilities
- **Centralized Date Functions**:
  - `calculate_days_between()`: Robust date difference calculator
  - Handles multiple date formats
  - Error handling for invalid dates

### Module 4: Data Merging
- Client ID + date-based matching
- Configurable date tolerance window
- Automatic column conflict resolution
- Merge statistics reporting

### Module 5: Analytics Engine
- Address frequency aggregation
- Pattern detection algorithms
- Facility matching logic (two-tier system)
- Geographic distance calculations (Haversine formula via geopy)

### Module 6: Visualization Engine
- Matplotlib/Seaborn integration
- Automatic plot type selection based on data types
- Customizable aesthetics (colors, labels, sizes)
- Export-ready figure generation

### Module 7: ML Pipeline
- Automated preprocessing (scaling, encoding, imputation)
- Model selection based on problem type
- Hyperparameter grid search
- Cross-validation (5-fold stratified)
- Feature importance extraction
- Performance visualization

## 🔗 API Integrations

### Canada Post AddressComplete API

**Purpose**: Validate and parse addresses into standardized components

**Endpoints Used**:
- `Find/v2.10/xmla.ws`: Search for address suggestions
- `Retrieve/v2.11/xmla.ws`: Get complete address details

**Rate Limits**: Check your API plan (typical: 100-500 requests/day free tier)

**Response Format**:
```xml
<Row Id="..." 
     BuildingNumber="1666" 
     Street="W 75th" 
     StreetType="Ave" 
     City="Vancouver" 
     ProvinceCode="BC" 
     PostalCode="V6P 6G2" />
```

### Google Maps Geocoding API

**Purpose**: Convert postal codes to geographic coordinates

**Endpoint**: `https://maps.googleapis.com/maps/api/geocode/json`

**Rate Limits**: 
- Free tier: $200 credit/month (~40,000 requests)
- Recommended: Enable billing and set budget alerts

**Response Format**:
```json
{
  "results": [{
    "geometry": {
      "location": {
        "lat": 49.2827,
        "lng": -123.1207
      }
    }
  }]
}
```

## 🤖 Machine Learning Models

### Supported Algorithms

#### Classification Models

1. **Logistic Regression**
   - Best for: Binary/multiclass classification with linear decision boundaries
   - Hyperparameters: C (regularization), solver, max_iter

2. **Random Forest Classifier**
   - Best for: Non-linear patterns, feature importance ranking
   - Hyperparameters: n_estimators, max_depth, min_samples_split

3. **XGBoost Classifier**
   - Best for: High performance, handles missing data well
   - Hyperparameters: n_estimators, max_depth, learning_rate, subsample

4. **CatBoost Classifier**
   - Best for: Categorical features (no encoding needed)
   - Hyperparameters: iterations, depth, learning_rate

5. **LightGBM Classifier**
   - Best for: Large datasets, fast training
   - Hyperparameters: n_estimators, max_depth, learning_rate, num_leaves

#### Regression Models

1. **Random Forest Regressor**
2. **XGBoost Regressor**
3. **CatBoost Regressor**
4. **LightGBM Regressor**

### Example Use Cases

**Classification Examples**:
- Predict patient readmission risk (High/Low)
- Classify housing stability (Stable/Unstable)
- Identify ED frequent users (Yes/No)

**Regression Examples**:
- Predict number of future address changes
- Estimate days until next ED visit
- Calculate distance to next move

### Model Evaluation

**Classification Metrics**:
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual breakdown of predictions

**Regression Metrics**:
- **R² Score**: Proportion of variance explained (0-1, higher is better)
- **MSE**: Mean Squared Error (lower is better)
- **MAE**: Mean Absolute Error (interpretable in original units)
- **RMSE**: Root Mean Squared Error (same units as target variable)

## 🛠️ Technologies

### Core Framework
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing

### Data Processing
- **requests**: HTTP API calls
- **xml.dom.minidom**: XML parsing (Canada Post responses)
- **urllib**: URL encoding and requests
- **re**: Regular expressions for address parsing

### Geospatial
- **geopy**: Geographic distance calculations (geodesic)
- **Google Maps API**: Geocoding service

### Visualization
- **Matplotlib**: Core plotting library
- **Seaborn**: Statistical visualizations

### Machine Learning
- **scikit-learn**: 
  - Preprocessing: StandardScaler, OneHotEncoder, LabelEncoder
  - Models: LogisticRegression, RandomForestClassifier/Regressor
  - Model Selection: train_test_split, GridSearchCV
  - Metrics: accuracy_score, confusion_matrix, r2_score
- **XGBoost**: Gradient boosting models
- **LightGBM**: Fast gradient boosting
- **CatBoost**: Categorical boosting

## 📁 Project Structure

```
address-analyzer/
├── address_analyzer.py    # Main application file (2118 lines)
├── requirements.txt             # Python dependencies
├── logo.png                     # Sidebar branding image
├── data/
│   └── sample_addresses.csv     # Example dataset
├── README.md                    # This file
└── .gitignore                   # Git exclusions (API keys, venv, etc.)
```

## ⚙️ Configuration

### Theme Settings

Toggle between light and dark mode in the sidebar:
- **Dark Mode**: Optimized for low-light environments
- **Light Mode**: Default clean interface

### API Configuration

**Canada Post API**:
```python
# Free tier: https://www.canadapost.ca/cpo/mc/business/productsservices/developers/services/addresscomplete/default.jsf
# Register for API key at Canada Post Developer Portal
api_key = "YOUR_KEY_HERE"
```

**Google Maps API**:
```python
# Enable Geocoding API in Google Cloud Console
# Set billing (free $200/month credit)
# Create API key with Geocoding API access
API_KEY = "YOUR_KEY_HERE"
```

### Performance Tuning

**Address Elementization**:
- Processes ~100 addresses/minute (API dependent)
- Implements 0.2s rate limiting for Google API
- Caches results in session state

**Machine Learning**:
- Grid Search: Enable for small datasets (<10,000 rows)
- Single parameters: Faster for large datasets
- Cross-validation: 5-fold default (adjustable in code)

## 📖 Sample Use Cases

### Use Case 1: Patient Housing Stability Analysis

**Objective**: Identify patients with unstable housing situations

**Workflow**:
1. Upload patient address history CSV
2. Elementize addresses with Canada Post API
3. Generate coordinates with Google API
4. Run "Move Frequency Relative to Patient" analysis
5. Run "Distance Moved Between Address Changes"
6. Visualize move frequency distribution (histogram)
7. Build ML classification model to predict high vs low mobility

**Outcome**: List of high-risk patients requiring housing support services

### Use Case 2: ED Facility Utilization Patterns

**Objective**: Understand which ED facilities serve precariously housed patients

**Workflow**:
1. Upload ED visit records with postal codes
2. Upload ED facility location CSV with postal codes
3. Upload Long-Term Care facility CSV
4. Run "Check ED Visit Centers Postal Code match"
5. Visualize facility match rates (pie chart)
6. Create count plot of top facilities

**Outcome**: Identify facilities needing specialized homeless patient services

### Use Case 3: Readmission Risk Prediction

**Objective**: Predict 30-day readmission risk for discharge planning

**Workflow**:
1. Upload emergency visit and discharge records
2. Merge datasets with date-based matching
3. Calculate "Days_Between_Visit" feature
4. Add move frequency and missing address fields
5. Select target: "Readmitted_30Days" (Yes/No)
6. Choose features: MoveFrequency, MissingFrequency, Days_Between_Visit
7. Train XGBoost classifier with Grid Search
8. Review feature importance (e.g., MoveFrequency most predictive)

**Outcome**: Risk scores for discharge planners to prioritize follow-ups

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### Development Setup

```bash
# Fork and clone the repo
git clone https://github.com/YOUR_USERNAME/address-analyzer.git
cd address-analyzer

# Create feature branch
git checkout -b feature/amazing-feature

# Make changes and test
streamlit run address_analyzer.py

# Commit with descriptive message
git commit -m "Add amazing feature"

# Push and create Pull Request
git push origin feature/amazing-feature
```

### Code Style
- Follow PEP 8 style guide
- Use descriptive variable names
- Add docstrings for complex functions
- Comment non-obvious logic

### Testing
- Test with sample datasets before PR
- Verify all 5 tabs function correctly
- Check both light and dark themes
- Test with/without API keys (fallback behavior)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Nay Zaw Lin**

- GitHub: [@neolin-pro](https://github.com/neolin-pro)
- Email: nayzawlin07@gmail.com

## 🙏 Acknowledgments

- **Vancouver Coastal Health**: For providing the use case and requirements
- **Canada Post**: AddressComplete API for address validation
- **Google Maps**: Geocoding API for coordinate generation
- **Streamlit Community**: For excellent documentation and examples

## 📚 References

### Healthcare & Housing Research
- Hwang, S. W. (2001). *Homelessness and health*. Canadian Medical Association Journal, 164(2), 229-233.
- Fazel, S., et al. (2014). *The health of homeless people in high-income countries*. The Lancet, 384(9953), 1529-1540.

### Geospatial Analysis
- Canada Post AddressComplete Documentation: https://www.canadapost.ca/pca/
- Google Maps Platform Documentation: https://developers.google.com/maps/documentation

### Machine Learning
- Pedregosa, F., et al. (2011). *Scikit-learn: Machine learning in Python*. JMLR, 12, 2825-2830.
- Chen, T., & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system*. KDD '16.

---

## 🚀 Quick Start Guide

```bash
# 1. Install
git clone https://github.com/nayzawlin/bc-address-analyzer.git
cd bc-address-analyzer
pip install -r requirements.txt

# 2. Configure API keys in ddress_analyzer.py

# 3. Run
streamlit run address_analyzer.py

# 4. Open browser at http://localhost:8501

# 5. Upload CSV, process, and analyze!
```

---

⭐ **If you find this project helpful, please consider giving it a star!**

💡 **Have questions or suggestions?** Open an issue or reach out via email.
