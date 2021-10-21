from prophet.forecaster import Prophet
import yfinance as yf
import datetime
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error

class Stock:
    """
    This class enables data loading, plotting and statistical analysis of a given stock,
     upon initialization load a sample of data to check if stock exists. 
        
    """
    params={
    'changepoint_prior_scale':0.0018298282889708827,
    'holidays_prior_scale':0.00011949782374119523,
    'seasonality_mode':'additive',
    'seasonality_prior_scale':4.240162804451275
        }

    def __init__(self, symbol="GOOG"):
       
        self.end = datetime.datetime.today()
        self.start = self.end - datetime.timedelta(days=4)
        self.symbol = symbol
        self.data = self.load_data(self.start, self.end)

    @st.cache(show_spinner=False)
    def load_data(self, start, end, inplace=False):
        """
        takes a start and end dates, download data do some processing and returns dataframe
        """

        data = yf.download(self.symbol, start, end + datetime.timedelta(days=1))
        try:
            assert len(data) > 0
        except AssertionError:
            print("Cannot fetch data, check spelling or time window")
        data.reset_index(inplace=True)
        data.rename(columns={"Date": "datetime"}, inplace=True)
        data["date"] = data.apply(lambda raw: raw["datetime"].date(), axis=1)

        data = data[["date", 'Close']]
        if inplace:
            self.data = data
            self.start = start
            self.end = end
            return True
        return data

    def plot_raw_data(self, fig):
        """
        Plot time-serie line chart of closing price on a given plotly.graph_objects.Figure object
        """
        fig = fig.add_trace(
            go.Scatter(
                x=self.data.date,
                y=self.data['Close'],
                mode="lines",
                name=self.symbol,
            )
        )
        return fig

    @staticmethod
    def nearest_business_day(DATE: datetime.date):
        """
        Takes a date and transform it to the nearest business day, 
        static because we would like to use it without a stock object.
        """
        if DATE.weekday() == 5:
            DATE = DATE - datetime.timedelta(days=1)

        if DATE.weekday() == 6:
            DATE = DATE + datetime.timedelta(days=1)
        return DATE

    def show_delta(self):
        """
        Visualize a summary of the stock change over the specified time period
        """
        epsilon = 1e-6
        i = self.start
        j = self.end
        s = self.data.query("date==@i")['Close'].values[0]
        e = self.data.query("date==@j")['Close'].values[0]

        difference = round(e - s, 2)
        change = round(difference / (s + epsilon) * 100, 2)
        e = round(e, 2)
        cols = st.columns(2)
        (color, marker) = ("green", "+") if difference >= 0 else ("red", "-")

        cols[0].markdown(
            f"""<p style="font-size: 90%;margin-left:5px">{self.symbol} \t {e}</p>""",
            unsafe_allow_html=True,
        )
        cols[1].markdown(
            f"""<p style="color:{color};font-size:90%;margin-right:5px">{marker} \t {difference} {marker} {change} % </p>""",
            unsafe_allow_html=True
        ) 



    @staticmethod
    def for_prophet(df: pd.DataFrame, date_column="date", y_column="Close") -> pd.DataFrame:
        return df.rename(columns={date_column: "ds", y_column: "y"})

    @st.cache(show_spinner=False)
    def load_train_test_data(self, TEST_INTERVAL_LENGTH, TRAIN_INTERVAL_LENGTH):
        """Returns two dataframes for testing and training"""
        TODAY = Stock.nearest_business_day(datetime.date.today())
        TEST_END = Stock.nearest_business_day(TODAY)
        TEST_START = Stock.nearest_business_day(
            TEST_END - datetime.timedelta(days=TEST_INTERVAL_LENGTH)
        )

        TRAIN_END = Stock.nearest_business_day(TEST_START - datetime.timedelta(days=1))
        TRAIN_START = Stock.nearest_business_day(
            TRAIN_END - datetime.timedelta(days=TRAIN_INTERVAL_LENGTH)
        )

        train_data = self.load_data(TRAIN_START, TRAIN_END)
        test_data = self.load_data(TEST_START, TEST_END)

        train_data = Stock.for_prophet(train_data)
        test_data = Stock.for_prophet(test_data)
        self.train_data = train_data
        self.test_data = test_data



    @st.cache(show_spinner=False)
    def train_prophet(self): 
        
        m = Prophet(**self.params)
        m.fit(self.train_data)
        self.model = m
        forecasts = m.predict(self.test_data)
        self.test_data = self.test_data.join(
            forecasts[["yhat_lower", "yhat", "yhat_upper"]]
        )
        self.test_mape = mean_absolute_percentage_error(
            self.test_data["y"], self.test_data["yhat"]
        )


    def plot_test(self):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.test_data["ds"],
                y=self.test_data["y"],
                mode="lines",
                name="True Closing price",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=self.test_data["ds"],
                y=self.test_data["yhat"],
                mode="lines",
                name="Predicted CLosing price",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=self.test_data["ds"],
                y=self.test_data["yhat_upper"],
                fill=None,
                mode="lines",
                name="CI+",
                line_color="orange",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=self.test_data["ds"],
                y=self.test_data["yhat_lower"],
                fill="tonexty",
                fillcolor='rgba(100,69,0,0.2)',
                mode="lines",
                line_color="orange",
                name="CI-",
            )
        )
        fig.update_layout(
            width=st.session_state.CHART_WIDTH,
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

        return fig


    def plot_inference(self) -> go.Figure:
        """
        Generate forecasts for the given horizon and plots them, returns a plotly.graph_objects.Figure
        """
        all_df=pd.concat([self.train_data,self.test_data[['ds','y']]])
        m=Prophet(**self.params)
        m.fit(all_df)
        self.model=m
        future=self.model.make_future_dataframe(periods=st.session_state.HORIZON,include_history=False)
        forecasts=self.model.predict(future)

        fig=go.Figure()
        fig.add_trace(
        go.Scatter(
            x=forecasts["ds"],
            y=forecasts["yhat"],
            mode="lines",
            name="Predicted CLosing price",
        )
        )

        fig.add_trace(
        go.Scatter(
            x=forecasts["ds"],
            y=forecasts["yhat_upper"],
            fill=None,
            mode="lines",
            name="CI+",
            line_color="orange",
        )
        )

        fig.add_trace(
        go.Scatter(
            x=forecasts["ds"],
            y=forecasts["yhat_lower"],
            fill="tonexty",
            fillcolor='rgba(100,69,0,0.2)',
            mode="lines",
            line_color="orange",
            name="CI-",
        )
        )
        fig.update_layout(
        width=st.session_state.CHART_WIDTH,
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
        return fig


    @staticmethod
    def train_test_forecast_report(symb): 
        """Launch training and plot testing results and reports MAPE error, finally it plots forecasts up to the specified horizon"""
        if st.session_state.TRAIN_JOB or st.session_state.TRAINED:
            text=st.empty() # Because streamlit adds widgets sequentially, we have to reserve a place at the top (after the chart of part 1)
            bar=st.empty() # Reserve a place for a progess bar
            
            text.write('Training model ... ') 
            bar=st.progress(0)

            stock = Stock(symb) 
            bar.progress(10)
            TEST_INTERVAL_LENGTH=st.session_state.TEST_INTERVAL_LENGTH #Retrieve input from the user
            TRAIN_INTERVAL_LENGTH=st.session_state.TRAIN_INTERVAL_LENGTH

            stock.load_train_test_data(TEST_INTERVAL_LENGTH, TRAIN_INTERVAL_LENGTH) #load train test data into the stock object, it's using cache
            bar.progress(30)
            stock.train_prophet() #this is also using cache
            bar.progress(70)
            text.write('Plotting test results ...')
            fig = stock.plot_test()
            bar.progress(100)
            bar.empty() #Turn the progress bar object back to what it was before and empty container
            st.markdown(
                f"## {symb} stock forecasts on testing set, Testing error {round(stock.test_mape*100,2)}%"
            )
            st.plotly_chart(fig)
            text.write('Generating forecasts ... ')
            fig2=stock.plot_inference() #Generate forecasts and plot them (no cache but figures are not updated if their data didn't change)
            st.markdown(f'## Forecasts for the next {st.session_state.HORIZON} days')
            st.plotly_chart(fig2)
            text.empty()
            """The button click will trigger this code to run only once, 
               the following flag TRAINED will keep this block of code executing even after the click,
               it won't redo everything however because we are using cache. 
               this flag needs to be initialized to False in the session state in main.py before the button"""

            st.session_state.TRAINED=True 
        else:
            st.markdown('Setup training job and hit Train')

        
    def save_forecasts(self,path):
        self.forecasts.to_csv(path)
