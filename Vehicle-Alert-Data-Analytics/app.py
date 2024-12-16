from flask import Flask, render_template, jsonify, send_from_directory
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.io as pio
import json

# Initialize Flask application
app = Flask(__name__)

# Function to load and preprocess data
def load_data():
    # Read CSV data files into pandas DataFrames
    df = pd.read_csv('data/iraste_nxt_cas.csv')
    df1 = pd.read_csv('data/iraste_nxt_casdms.csv')
    
    # Concatenate both DataFrames and clean the data
    df = pd.concat([df, df1], axis=0)
    df = df.drop_duplicates()  # Remove duplicates
    df = df.dropna()  # Remove missing values
    df = df.sample(frac=0.01, random_state=42)  # Take a random sample of 1% of the data
    return df

# Route to perform spatial analysis and display map of alert occurrences
@app.route('/spatial-analysis')
def spatial_analysis():
    df = load_data()
    # Create a heatmap of alert occurrences based on latitude and longitude
    fig = px.density_mapbox(df, lat='Lat', lon='Long', radius=10, zoom=5, 
                             mapbox_style='carto-positron',
                             title='Spatial Distribution of Alert Occurrences')
    fig.update_layout(mapbox_center={'lat': df['Lat'].mean(), 'lon': df['Long'].mean()})
    
    return jsonify(fig.to_json())

# Route to analyze alert frequency and vehicle alert comparison
@app.route('/alert-frequency')
def alert_frequency():
    df = load_data()
    df['Date'] = pd.to_datetime(df['Date'])
    df['DayOfWeek'] = df['Date'].dt.day_name()
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    df['Speed'] = df['Speed'].astype(float)
    
    # Alert Frequency Analysis by Day of Week
    fig1 = px.histogram(df, x='DayOfWeek', color='Alert', 
                        title='Alert Frequency by Day of Week')
    fig1.update_layout(xaxis={'categoryorder':'total descending'})
    
    # Alert Frequency Comparison Across Different Vehicles
    fig2 = px.scatter(df, x='Speed', color='Alert', 
                      title='Alert Frequency Comparison Across Different Vehicles')
    fig2.update_layout(xaxis_title='Speed', yaxis_title='Alert Frequency')
    
    # Speed vs Time with Alert Events
    fig_speed_time = px.scatter(df, x='Time', y='Speed', color='Alert', 
                                title='Speed vs. Time with Alert Events')
    fig_speed_time_json = json.loads(fig_speed_time.to_json())
    
    return jsonify({
        'day_of_week_chart': fig1.to_json(),
        'speed_alert_chart': fig2.to_json(),
        'speed_time': fig_speed_time_json
    })

# Route to analyze vehicle speed and categorize alerts based on speed
@app.route('/speed-analysis')
def speed_analysis():
    df = load_data()
    df['Speed'] = df['Speed'].astype(float)
    df_sorted = df.sort_values(by='Time')
    df_sorted['Time'] = pd.to_datetime(df_sorted['Time'], errors='coerce')
    
    # Speed categorization function
    def categorize_speed(speed):
        if speed < 60:
            return 'Low'
        elif 60 <= speed < 80:
            return 'Medium'
        else:
            return 'High'
    
    df_sorted['Speed_Category'] = df_sorted['Speed'].apply(categorize_speed)
    
    # Speed vs Time Scatter Plot
    fig1 = px.scatter(df_sorted, x='Time', y='Speed', color='Alert', 
                      title='Speed vs. Time with Alert Events')
    
    # Speed Distribution Histogram
    fig2 = px.histogram(df_sorted, x='Speed', nbins=20, 
                        title='Distribution of Speed')
    
    # Alerts by Speed Category
    grouped_data = df_sorted.groupby(['Speed_Category', 'Alert']).size().reset_index(name='Count')
    fig3 = px.bar(grouped_data, x='Speed_Category', y='Count', color='Alert', 
                  barmode='group', title='Alerts Count by Speed Category')
    
    return jsonify({
        'speed_time_chart': fig1.to_json(),
        'speed_distribution_chart': fig2.to_json(),
        'speed_category_chart': fig3.to_json()
    })

# Route to calculate and display correlation analysis
@app.route('/correlation-analysis')
def correlation_analysis():
    df = load_data()
    df1 = df.copy()
    
    # Prepare data for correlation calculation
    df1['Alert'] = df1['Alert'].astype('category').cat.codes
    df1['Date'] = pd.to_datetime(df1['Date'])
    df1['DayOfWeek'] = df1['Date'].dt.day_name()
    df1['HourOfDay'] = df1['Date'].dt.hour
    df1['Date'] = df1['Date'].astype('category').cat.codes
    df1['Time'] = pd.to_datetime(df1['Time'], errors='coerce')
    df1['DayOfWeek'] = df1['DayOfWeek'].astype('category').cat.codes
    df1['HourOfDay'] = df1['HourOfDay'].astype('category').cat.codes
    df1.drop(['HourOfDay'], axis=1, inplace=True)

    # Calculate correlation matrix
    correlation_matrix = df1.corr()
    
    # Create a heatmap for the correlation matrix
    fig = ff.create_annotated_heatmap(
        z=correlation_matrix.values,
        x=list(correlation_matrix.columns),
        y=list(correlation_matrix.index),
        colorscale='Viridis'
    )
    fig.update_layout(title='Correlation Between Alert Occurrence and Road Conditions')
    
    return jsonify(fig.to_json())

# Route to analyze driver behavior based on alert counts
@app.route('/driver-behavior')
def driver_behavior():
    df = load_data()
    alert_counts = df['Alert'].value_counts().reset_index()
    alert_counts.columns = ['Alert', 'Frequency']
    
    # Create a pie chart of alert frequencies
    fig = px.pie(alert_counts, values='Frequency', names='Alert', 
                 title='Distribution of Driver Alerts')
    
    return jsonify(fig.to_json())

# Route for analyzing safety-related alerts
@app.route('/safety-impact')
def safety_impact():
    df = load_data()
    safety_df = df[df['Alert'].isin(['cas_ldw', 'cas_hmw', 'hard_brake', 'cas_pcw', 'cas_fcw'])]
    
    # Speed vs Frequency of Safety-Related Alerts
    fig1 = px.scatter(safety_df.groupby('Speed')['Alert'].count().reset_index(), 
                      x='Speed', y='Alert',
                      title='Speed vs. Frequency of Safety-Related Alerts')
    
    # Box plot of Speed Distribution during Safety Alerts
    fig2 = px.box(safety_df, x='Alert', y='Speed', 
                  title='Speed Distribution During Safety Alerts')
    
    return jsonify({
        'safety_speed_frequency': fig1.to_json(),
        'safety_speed_distribution': fig2.to_json()
    })

# Route for combined safety analysis and alerts visualization
@app.route('/safety_analysis')
def safety_analysis():
    df = load_data()
    
    # Filter for safety-related alerts
    safety_alerts = ['cas_ldw', 'cas_hmw', 'hard_brake', 'cas_pcw', 'cas_fcw']
    safety_df = df[df['Alert'].isin(safety_alerts)]

    # Safety-related Alerts Frequency
    safety_counts = safety_df['Alert'].value_counts().reset_index()
    safety_counts.columns = ['Alert', 'Frequency']
    
    # Pie chart for safety-related alerts distribution
    fig_safety_pie = px.pie(safety_counts, values='Frequency', names='Alert', 
                            title='Distribution of Safety-Related Alerts')
    fig_safety_pie_json = json.loads(fig_safety_pie.to_json())

    # Speed vs Frequency of Safety Alerts
    safety_speed_freq = safety_df.groupby('Speed')['Alert'].count().reset_index()
    fig_safety_speed = px.scatter(safety_speed_freq, x='Speed', y='Alert', 
                                  title='Speed vs. Frequency of Safety-Related Alerts', 
                                  trendline='ols')
    fig_safety_speed_json = json.loads(fig_safety_speed.to_json())

    return jsonify({
        'safety_pie': fig_safety_pie_json,
        'safety_speed': fig_safety_speed_json
    })

# Home route to show summary and visualizations
@app.route('/')
def home():
    df = load_data()
    summary = df.describe(include='all').to_html(classes='dataframe', border=0)

    # Create heatmap for spatial analysis
    fig1 = px.density_mapbox(df, lat='Lat', lon='Long', radius=10, zoom=5, mapbox_style='carto-positron',
                             title='Spatial Distribution of Alert Occurrences')
    fig1.update_layout(mapbox_center={'lat': df['Lat'].mean(), 'lon': df['Long'].mean()})
    heatmap_html = pio.to_html(fig1, full_html=False)

    # Alert frequency by day of the week
    df['Date'] = pd.to_datetime(df['Date'])
    df['DayOfWeek'] = df['Date'].dt.day_name()
    freq_data = df['DayOfWeek'].value_counts().reset_index()
    freq_data.columns = ['Day', 'Frequency']
    fig2 = px.bar(freq_data, x='Day', y='Frequency', title='Alert Frequency by Day of the Week',
                  color='Frequency', labels={'Frequency': 'Number of Alerts'})
    alert_frequency_html = pio.to_html(fig2, full_html=False)

    # Speed analysis
    fig3 = px.histogram(df, x='Speed', nbins=30, title='Vehicle Speed Distribution',
                        labels={'Speed': 'Speed (km/h)', 'count': 'Number of Vehicles'})
    speed_analysis_html = pio.to_html(fig3, full_html=False)

    locations = [{
        'lat': row['Lat'], 
        'lng': row['Long'], 
        'alert': row['Alert'],
        'date': row['Date'],
        'time': row['Time'],
        'vehicle': row['Vehicle'],
        'speed': row['Speed']
    } for _, row in df.iterrows()]

    return render_template('index.html', summary=summary, 
                           heatmap=heatmap_html, 
                           alert_frequency=alert_frequency_html, 
                           speed_analysis=speed_analysis_html,
                           locations=locations)

# Route to get cleaned data for coordinates
@app.route('/data/coordinates')
def get_coordinates():
    df = load_data()
    data = []
    for _, row in df.iterrows():
        record={
           "alert": row['Alert'] if pd.notna(row['Alert']) else '',
            "date": row['Date'] if pd.notna(row['Date']) else '',
            "time": row['Time'] if pd.notna(row['Time']) else '',
            "lat": row['Lat'] if pd.notna(row['Lat']) else None,  # Handle NaN latitude
            "long": row['Long'] if pd.notna(row['Long']) else None,  # Handle NaN longitude
            "vehicle": row['Vehicle'] if pd.notna(row['Vehicle']) else '',
            "speed": row['Speed'] if pd.notna(row['Speed']) else 0  # Handle NaN speed
        }
        data.append(record)
    return jsonify(data)

# Ensure the app runs when accessed directly
if __name__ == '__main__':
    app.run(debug=True)
