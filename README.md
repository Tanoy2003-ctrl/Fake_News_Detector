# 🔍 Fake News Detector

A machine learning-powered web application built with Streamlit to detect fake news articles using Natural Language Processing and Random Forest classification.

## 🚀 Features

- **🔍 Single Article Analysis**: Analyze individual news articles for authenticity
- **📊 Model Performance Dashboard**: View detailed performance metrics and confusion matrix
- **🧪 Test Examples**: Run predefined test cases to validate model performance
- **📁 Batch Analysis**: Analyze multiple articles from uploaded text files
- **📈 Interactive Visualizations**: Confidence gauges, probability charts, and performance metrics
- **💾 Export Results**: Download analysis results as CSV files

## 📋 Requirements

- Python 3.8+
- Required packages (install via `pip install -r requirements.txt`):
  - streamlit>=1.47.0
  - plotly>=6.2.0
  - pandas>=2.3.0
  - scikit-learn>=1.5.0
  - numpy>=1.23.0

## 📁 Data Setup

Before running the application, ensure you have the following CSV files in your Downloads folder:

- `Fake.csv` - Dataset containing fake news articles
- `True.csv` - Dataset containing real news articles

These datasets should have at least the following columns:
- `title` - Article title
- `text` - Article content

## 🏃‍♂️ How to Run

### Option 1: Streamlit Web Interface (Recommended)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Streamlit app:**
   ```bash
   streamlit run run_app.py
   ```

3. **Open your browser** and navigate to `http://localhost:8501`

4. **Load & Train Model** by clicking the button in the sidebar

5. **Start analyzing news articles!**

### Option 2: Command Line Interface

1. **Run the original script:**
   ```bash
   python Main.py --interactive
   ```

2. **Other CLI options:**
   ```bash
   # Analyze single text
   python Main.py --text "Your news article here"
   
   # Quick analysis
   python Main.py --quick "Breaking news text"
   
   # Analyze from file
   python Main.py --file articles.txt
   ```

## 🎯 How It Works

### 1. Data Preprocessing
- Loads and balances fake and real news datasets
- Cleans text by removing special characters and normalizing case
- Combines title and content for comprehensive analysis

### 2. Model Training
- Uses TF-IDF vectorization with optimized parameters
- Trains a Random Forest classifier with balanced class weights
- Implements cross-validation for robust performance

### 3. Enhanced Prediction
- Combines machine learning predictions with keyword-based analysis
- Uses domain-specific indicators for improved accuracy
- Provides confidence scores and probability distributions

### 4. Real-time Analysis
- Processes user input in real-time
- Displays results with confidence levels and explanations
- Offers batch processing for multiple articles

## 📊 Model Performance

The model typically achieves:
- **Accuracy**: 85-95% on test data
- **Precision**: High precision for both fake and real news detection
- **Recall**: Balanced recall across both classes
- **F1-Score**: Optimized for balanced performance

## 🎨 Web Interface Features

### 🔍 Analyze News Tab
- **Text Input**: Type or paste news articles
- **File Upload**: Upload text files for analysis
- **Real-time Results**: Instant prediction with confidence scores
- **Visual Feedback**: Color-coded results and confidence gauges

### 📊 Model Performance Tab
- **Accuracy Metrics**: Overall model accuracy
- **Confusion Matrix**: Interactive visualization
- **Detailed Metrics**: Precision, recall, and F1-scores
- **Performance Charts**: Visual performance breakdown

### 🧪 Test Examples Tab
- **Predefined Tests**: Run sample articles to validate model
- **Batch Testing**: Test multiple examples at once
- **Results Summary**: Accuracy and confidence statistics

### 📁 Batch Analysis Tab
- **File Upload**: Process multiple articles from text files
- **Progress Tracking**: Real-time processing updates
- **Results Export**: Download results as CSV
- **Summary Statistics**: Overview of analysis results

## 🔧 Customization

### Adding New Indicators
You can customize the keyword-based analysis by modifying the indicator lists in the `predict_news_enhanced` function:

```python
# Add to real news indicators
strong_real_indicators = [
    'your_custom_indicator',
    # ... existing indicators
]

# Add to fake news indicators
strong_fake_indicators = [
    'your_custom_indicator',
    # ... existing indicators
]
```

### Model Parameters
Adjust the Random Forest parameters in the `train_model` function:

```python
model = RandomForestClassifier(
    n_estimators=150,        # Number of trees
    max_depth=25,           # Maximum tree depth
    min_samples_split=3,    # Minimum samples to split
    # ... other parameters
)
```

## 🚨 Important Notes

1. **Data Privacy**: The application processes text locally - no data is sent to external servers
2. **Model Limitations**: Performance depends on the quality and diversity of training data
3. **Continuous Learning**: Consider retraining with new data periodically
4. **Context Matters**: The model works best with complete news articles rather than headlines only

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Troubleshooting

### Common Issues

1. **"Dataset files not found" error**
   - Ensure `Fake.csv` and `True.csv` are in your Downloads folder
   - Check file permissions

2. **Memory issues during training**
   - Reduce `max_features` in TF-IDF vectorizer
   - Use a smaller subset of training data

3. **Slow performance**
   - Install watchdog for better file monitoring: `pip install watchdog`
   - Reduce `n_estimators` in Random Forest

4. **Port already in use**
   - Use a different port: `streamlit run streamlit_app.py --server.port 8502`

## 📞 Support

For issues, questions, or contributions, please create an issue in the repository or contact the development team.

---

**Happy Fake News Detection! 🔍📰**
