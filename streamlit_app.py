import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import pickle
import os
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Pick 4 Pattern Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🎯 Pick 4 Pattern Predictor")
st.markdown("### Advanced Pattern Analysis for Lottery Predictions")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Home", "Predictions", "Backtest", "Pattern Analysis", "Settings"])

# Initialize session state
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "backtest_results" not in st.session_state:
    st.session_state.backtest_results = None

def load_model():
    """Load the trained model"""
    try:
        model_path = "model.pkl"
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            st.session_state.model_loaded = True
            return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
    return None

def format_predictions_as_text(predictions_df):
    """Format predictions DataFrame as detailed text with headers and separators"""
    text_output = []
    text_output.append("=" * 80)
    text_output.append("PICK 4 PATTERN PREDICTIONS")
    text_output.append("=" * 80)
    text_output.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    text_output.append("")
    
    if predictions_df is None or len(predictions_df) == 0:
        text_output.append("No predictions available.")
        return "\n".join(text_output)
    
    # Header row
    text_output.append("-" * 80)
    text_output.append(f"{'Index':<8} {'Pattern':<15} {'Confidence':<15} {'Probability':<15} {'Status':<20}")
    text_output.append("-" * 80)
    
    # Data rows
    for idx, row in predictions_df.iterrows():
        pattern = str(row.get('pattern', 'N/A'))
        confidence = f"{float(row.get('confidence', 0)):.4f}" if 'confidence' in row else "N/A"
        probability = f"{float(row.get('probability', 0)):.4f}" if 'probability' in row else "N/A"
        status = str(row.get('status', 'Pending'))
        
        text_output.append(f"{idx:<8} {pattern:<15} {confidence:<15} {probability:<15} {status:<20}")
    
    text_output.append("-" * 80)
    text_output.append("")
    text_output.append("ANALYSIS GUIDE:")
    text_output.append("-" * 80)
    text_output.append("• Confidence: Measure of pattern strength (0.0 - 1.0)")
    text_output.append("• Probability: Predicted likelihood of occurrence (0.0 - 1.0)")
    text_output.append("• Status: Current validation status of the prediction")
    text_output.append("")
    text_output.append("RECOMMENDATIONS FOR IMPROVEMENT:")
    text_output.append("-" * 80)
    text_output.append("• Focus on patterns with confidence > 0.7 for best results")
    text_output.append("• Monitor multiple patterns simultaneously for redundancy")
    text_output.append("• Track historical accuracy to adjust selection criteria")
    text_output.append("• Use probability scores to rank pattern likelihood")
    text_output.append("=" * 80)
    
    return "\n".join(text_output)

def format_backtest_as_text(backtest_df, summary_stats=None):
    """Format backtest results as detailed text with comprehensive analysis"""
    text_output = []
    text_output.append("=" * 100)
    text_output.append("BACKTEST RESULTS - PATTERN PERFORMANCE ANALYSIS")
    text_output.append("=" * 100)
    text_output.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    text_output.append("")
    
    if backtest_df is None or len(backtest_df) == 0:
        text_output.append("No backtest results available.")
        return "\n".join(text_output)
    
    # Summary Statistics Section
    if summary_stats:
        text_output.append("SUMMARY STATISTICS")
        text_output.append("-" * 100)
        text_output.append(f"Total Tests Run:        {summary_stats.get('total_tests', 0)}")
        text_output.append(f"Successful Predictions: {summary_stats.get('successful', 0)}")
        text_output.append(f"Failed Predictions:     {summary_stats.get('failed', 0)}")
        text_output.append(f"Success Rate:           {summary_stats.get('success_rate', 0):.2%}")
        text_output.append(f"Average Confidence:     {summary_stats.get('avg_confidence', 0):.4f}")
        text_output.append(f"Average Probability:    {summary_stats.get('avg_probability', 0):.4f}")
        text_output.append("")
    
    # Detailed Results Table
    text_output.append("DETAILED RESULTS")
    text_output.append("-" * 100)
    text_output.append(f"{'Date':<12} {'Pattern':<12} {'Predicted':<12} {'Actual':<12} {'Match':<8} {'Confidence':<12} {'Notes':<25}")
    text_output.append("-" * 100)
    
    for idx, row in backtest_df.iterrows():
        date = str(row.get('date', 'N/A'))[:10]
        pattern = str(row.get('pattern', 'N/A'))
        predicted = str(row.get('predicted', 'N/A'))
        actual = str(row.get('actual', 'N/A'))
        match = "✓ YES" if row.get('match', False) else "✗ NO"
        confidence = f"{float(row.get('confidence', 0)):.4f}" if 'confidence' in row else "N/A"
        notes = str(row.get('notes', ''))[:20]
        
        text_output.append(f"{date:<12} {pattern:<12} {predicted:<12} {actual:<12} {match:<8} {confidence:<12} {notes:<25}")
    
    text_output.append("-" * 100)
    text_output.append("")
    text_output.append("INTERPRETATION GUIDE")
    text_output.append("-" * 100)
    text_output.append("Match: Indicates whether the prediction matched the actual result")
    text_output.append("Confidence: Model's confidence in the prediction (higher is better)")
    text_output.append("Success Rate: Percentage of correct predictions")
    text_output.append("")
    text_output.append("LEARNING RECOMMENDATIONS")
    text_output.append("-" * 100)
    text_output.append("1. PATTERN SELECTION:")
    text_output.append("   - Identify which patterns have highest success rates")
    text_output.append("   - Focus resources on patterns with >60% accuracy")
    text_output.append("   - Avoid patterns that consistently underperform")
    text_output.append("")
    text_output.append("2. TIMING ANALYSIS:")
    text_output.append("   - Review predictions by day of week for patterns")
    text_output.append("   - Check for seasonal variations in success rates")
    text_output.append("   - Track performance trends over time")
    text_output.append("")
    text_output.append("3. CONFIDENCE THRESHOLDS:")
    text_output.append("   - Set minimum confidence levels for actual play")
    text_output.append("   - Higher confidence should correlate with better results")
    text_output.append("   - Adjust thresholds based on your risk tolerance")
    text_output.append("")
    text_output.append("4. RISK MANAGEMENT:")
    text_output.append("   - Diversify across multiple patterns")
    text_output.append("   - Adjust bet amounts based on confidence scores")
    text_output.append("   - Track ROI for each pattern category")
    text_output.append("=" * 100)
    
    return "\n".join(text_output)

# PAGE: Home
if page == "Home":
    st.markdown("## Welcome to Pick 4 Pattern Predictor")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Features
        - 🎯 Advanced pattern recognition
        - 📊 Backtest historical data
        - 📈 Confidence scoring
        - 💾 Export detailed reports
        - 🔍 Pattern analysis tools
        """)
    
    with col2:
        st.markdown("""
        ### How to Use
        1. Navigate to **Predictions** for next draws
        2. Use **Backtest** to validate performance
        3. Analyze patterns in **Pattern Analysis**
        4. Configure settings in **Settings**
        5. Export results as formatted text
        """)
    
    st.markdown("---")
    st.info("💡 **Tip**: Always backtest new patterns before using them in production!")

# PAGE: Predictions
elif page == "Predictions":
    st.markdown("## Predictions")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### Generate Next Draw Predictions")
    
    with col2:
        if st.button("🔄 Generate Predictions", key="gen_pred"):
            # Simulate loading a model
            model = load_model()
            
            # Generate sample predictions
            sample_data = {
                'pattern': ['1234', '5678', '2468', '1357', '4567'],
                'confidence': [0.8234, 0.7156, 0.6892, 0.7543, 0.7891],
                'probability': [0.7821, 0.6934, 0.6234, 0.7123, 0.7654],
                'status': ['Strong', 'Good', 'Fair', 'Good', 'Strong']
            }
            st.session_state.predictions = pd.DataFrame(sample_data)
            st.success("✓ Predictions generated!")
    
    # Display predictions if available
    if st.session_state.predictions is not None:
        st.markdown("### Current Predictions")
        st.dataframe(st.session_state.predictions, use_container_width=True)
        
        # Download button for formatted text
        predictions_text = format_predictions_as_text(st.session_state.predictions)
        st.download_button(
            label="📥 Download Predictions (Text Format)",
            data=predictions_text,
            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            key="download_pred"
        )
        
        st.markdown("---")
        st.markdown("### Predictions Preview")
        st.text(predictions_text)
    else:
        st.info("Click 'Generate Predictions' to create new predictions for the next draw.")

# PAGE: Backtest
elif page == "Backtest":
    st.markdown("## Backtest Results")
    
    st.markdown("### Backtest Historical Performance")
    
    backtest_start_col, backtest_end_col, backtest_btn_col = st.columns([1, 1, 1])
    
    with backtest_start_col:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
    
    with backtest_end_col:
        end_date = st.date_input("End Date", value=datetime.now())
    
    with backtest_btn_col:
        if st.button("▶ Run Backtest", key="run_backtest"):
            # Generate sample backtest data
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            sample_backtest = {
                'date': dates[:20],
                'pattern': ['1234', '5678', '2468', '1357', '4567'] * 4,
                'predicted': ['1234', '5678', '2468', '1357', '4567'] * 4,
                'actual': ['1234', '5679', '2468', '1357', '4568'] + ['1234', '5678', '2468', '1357', '4567'] * 3 + ['1234', '5679'],
                'match': [True, False, True, True, False] * 4,
                'confidence': [0.8234, 0.7156, 0.6892, 0.7543, 0.7891] * 4,
                'notes': ['Correct', 'Off by 1', 'Correct', 'Correct', 'Off by 1'] * 4
            }
            
            st.session_state.backtest_results = pd.DataFrame(sample_backtest)
            
            # Calculate summary stats
            summary = {
                'total_tests': len(sample_backtest['date']),
                'successful': sum(sample_backtest['match']),
                'failed': len(sample_backtest['match']) - sum(sample_backtest['match']),
                'success_rate': sum(sample_backtest['match']) / len(sample_backtest['match']),
                'avg_confidence': np.mean(sample_backtest['confidence']),
                'avg_probability': np.mean(sample_backtest['confidence'])
            }
            st.session_state.backtest_summary = summary
            st.success("✓ Backtest completed!")
    
    # Display backtest results
    if st.session_state.backtest_results is not None:
        st.markdown("### Results Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        summary = st.session_state.backtest_summary
        with col1:
            st.metric("Total Tests", summary['total_tests'])
        with col2:
            st.metric("Successful", summary['successful'])
        with col3:
            st.metric("Failed", summary['failed'])
        with col4:
            st.metric("Success Rate", f"{summary['success_rate']:.1%}")
        
        st.markdown("---")
        st.markdown("### Detailed Results")
        st.dataframe(st.session_state.backtest_results, use_container_width=True)
        
        # Download button for formatted text
        backtest_text = format_backtest_as_text(st.session_state.backtest_results, summary)
        st.download_button(
            label="📥 Download Backtest Report (Text Format)",
            data=backtest_text,
            file_name=f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            key="download_backtest"
        )
        
        st.markdown("---")
        st.markdown("### Full Report Preview")
        st.text(backtest_text)
    else:
        st.info("Set date range and click 'Run Backtest' to analyze historical performance.")

# PAGE: Pattern Analysis
elif page == "Pattern Analysis":
    st.markdown("## Pattern Analysis")
    
    st.markdown("### Analyze Specific Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pattern_input = st.text_input("Enter Pattern (e.g., 1234)", placeholder="Enter 4-digit pattern")
    
    with col2:
        if st.button("🔍 Analyze Pattern", key="analyze_pattern"):
            if pattern_input and len(pattern_input) == 4:
                st.success(f"Analyzing pattern: {pattern_input}")
                
                # Sample analysis data
                analysis_data = {
                    'metric': ['Historical Frequency', 'Confidence Score', 'Probability', 'Last Seen', 'Performance'],
                    'value': ['2.3% of draws', '0.7894', '0.7234', '5 days ago', 'Strong']
                }
                
                st.dataframe(pd.DataFrame(analysis_data), use_container_width=True)
                
                st.markdown("### Pattern Trends")
                # Placeholder for chart
                st.info("Chart visualization would appear here")
            else:
                st.error("Please enter a valid 4-digit pattern")

# PAGE: Settings
elif page == "Settings":
    st.markdown("## Settings")
    
    st.markdown("### Application Settings")
    
    with st.expander("Model Configuration", expanded=True):
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7)
        min_samples = st.number_input("Minimum Sample Size", 10, 1000, 100)
        st.success(f"Threshold set to {confidence_threshold:.2f}")
    
    with st.expander("Export Settings", expanded=False):
        export_format = st.radio("Default Export Format", ["Text Format", "JSON", "Plain Text"])
        include_headers = st.checkbox("Include Analysis Headers", value=True)
        include_recommendations = st.checkbox("Include Recommendations", value=True)
        st.success("Export settings updated!")
    
    with st.expander("Display Preferences", expanded=False):
        show_confidence = st.checkbox("Show Confidence Scores", value=True)
        show_probability = st.checkbox("Show Probability", value=True)
        decimal_places = st.number_input("Decimal Places", 2, 6, 4)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Pick 4 Pattern Predictor v1.0 | Disclaimer: For entertainment purposes only</p>
    <p><small>Report any issues or suggestions to improve the application</small></p>
</div>
""", unsafe_allow_html=True)
