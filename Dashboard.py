import streamlit as st
import pandas as pd
import datetime as dt

# Load data once outside of callbacks
yield_df = pd.read_csv('./data/yield_df.csv')

st.title('Crop Yield Dashboard')
st.write('')
# Initialize input_data
input_data = {
    'Area': sorted(yield_df['Area'].unique())[0],  # Default to first area
    'Year': dt.datetime.now().year
}

with st.form(key='analytical-crop-yield'):
    area = st.selectbox('Area', sorted(yield_df['Area'].unique()))
    year = st.slider('Year',
                     min_value=int(yield_df['Year'].min()),
                     max_value=int(yield_df['Year'].max()),
                     value=int(yield_df['Year'].max()),
                     step=1)

    submit_btn = st.form_submit_button(label='Create Analytical Crop')

    if submit_btn:
        # Update input data when form is submitted
        input_data = {
            'Area': area,
            'Year': year,
        }

# Only process data and display charts if we have valid inputs
if input_data:
    summary_data = yield_df[yield_df['Area'] == input_data['Area']].groupby(['Year', 'Item'])[
        'hg/ha_yield'].mean().reset_index()

    # Display historical data for the selected area
    st.subheader(f'Historical Yield Data for {input_data["Area"]}')
    # st.line_chart(summary_data, x="Year", y="hg/ha_yield",
    #              y_label='Year', x_label='Crop yield hg/ha',
    #              color="Item", horizontal=True, use_container_width=True)
    multi_line_data = summary_data.pivot(index="Year", columns="Item", values="hg/ha_yield")
    st.line_chart(multi_line_data)


    # Filter data for the selected year
    year_data = summary_data[summary_data['Year'] == input_data['Year']]

    if not year_data.empty:
        st.subheader(f'Crop Yields for {input_data["Area"]} in {input_data["Year"]}')
        # Bar chart is more suitable for comparing different items in a single year
        st.bar_chart(year_data.set_index('Item')['hg/ha_yield'],
                     x_label='Year', y_label='Crop yield hg/ha', use_container_width=True)
    else:
        st.warning(f"No data available for {input_data['Area']} in {input_data['Year']}")

    # Add some general statistics
    st.subheader('Statistics')
    avg_yield = summary_data.groupby('Item')['hg/ha_yield'].mean().reset_index()
    avg_yield.columns = ['Crop', 'Average Yield (hg/ha)']
    st.dataframe(avg_yield)
