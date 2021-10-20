from mimetypes import init
import pandas as pd
import streamlit as st
from prophet import Prophet, forecaster
import datetime
from plotly import graph_objects as go
import numpy as np
import plotly.graph_objects as go
from stock_object import Stock

st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.markdown("# Train forecasting models for stock prices")
# ------ layout setting---------------------------
window_selection_c = st.sidebar.container()
window_selection_c.markdown("## Insights")

sub_columns = window_selection_c.columns(2)
change_c = st.sidebar.container()

# ----------Time window selection-----------------
YESTERDAY = Stock.nearest_business_day(datetime.date.today()-datetime.timedelta(days=1))
DEFAULT_START = Stock.nearest_business_day(YESTERDAY - datetime.timedelta(days=700))


START = sub_columns[0].date_input(
    "From", value=DEFAULT_START, max_value=YESTERDAY - datetime.timedelta(days=1)
)
START = Stock.nearest_business_day(START)
END = sub_columns[1].date_input("To", value=YESTERDAY, max_value=YESTERDAY, min_value=START)
END = Stock.nearest_business_day(END)


# ---------------stock selection------------------
STOCKS = np.array(["AAPL", "GOOG", "MSFT", "GME", "FB",'TSLA'])  # TODO : include all stocks
SYMB = window_selection_c.selectbox("select stock", STOCKS)

chart_width = st.expander(label="chart width").slider("", 1000, 2800, 1400)


# ------------------------Plot stock linecharts--------------------


fig = go.Figure()
stock = Stock(symbol=SYMB)
stock.load_data(START, END, inplace=True)
fig = stock.plot_raw_data(fig)
with change_c:
    stock.show_delta()

fig.update_layout(
            width=chart_width,
            margin=dict(l=0, r=0, t=0, b=0, pad=0),
            legend=dict(
                x=0,
                y=0.99,
                traceorder="normal",
                font=dict(size=12),
            ),
            autosize=False,
            template="plotly_dark",
)




st.write(fig)

# ---------------------------------------------------------------------------------

if "TEST_INTERVAL_LENGTH" not in st.session_state:
    # set the initial default value of the slider widget
    st.session_state.TEST_INTERVAL_LENGTH = 100

if "TRAIN_INTERVAL_LENGTH" not in st.session_state:
    # set the initial default value of the slider widget
    st.session_state.TRAIN_INTERVAL_LENGTH = 200

if "HORIZON" not in st.session_state:
    # set the initial default value of the slider widget
    st.session_state.HORIZON = 7

if "train_job" not in st.session_state:
    # set the initial default value of the slider widget
    st.session_state.train_job = False
# ------------------------------------------Training configurations-------------------------------------
st.sidebar.markdown("## Forecasts")
form = st.sidebar.form(key="train_dataset")




form.markdown("## Select interval lengths")
HORIZON = form.number_input(
    "Inference horizon", min_value=7, max_value=200, key="HORIZON"
)
TEST_INTERVAL_LENGTH = form.number_input(
    "number of days to test on and visualize",
    min_value=7,
    key="TEST_INTERVAL_LENGTH",
)

TRAIN_INTERVAL_LENGTH = form.number_input(
    "number of  day to use for training",
    min_value=60,
    key="TRAIN_INTERVAL_LENGTH",
)


form.form_submit_button(
    label="Train",
    on_click=Stock.launch_training
)

Stock.train_forecast_report(chart_width, SYMB)