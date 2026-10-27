import streamlit as st, pandas as pd, joblib, plotly.express as px

st.set_page_config(page_title="Food Waste Predictor", layout="wide")
st.title("Food Waste Prediction and Consumption Dashboard")

bundle = joblib.load("waste_model.joblib")
model, feats = bundle["model"], bundle["features"]
df = pd.read_csv("daily_data.csv", parse_dates=["date"])

st.sidebar.header("Tomorrow's conditions")
next_date = df.date.max() + pd.Timedelta(days=1)
temp = st.sidebar.slider("Max temperature (C)", 20.0, 42.0, 32.0)
rain = st.sidebar.slider("Rainfall (mm)", 0.0, 100.0, 0.0)
hol = st.sidebar.checkbox("Holiday / no-class day")

row = {"lag_1": df.waste_units.iloc[-1], "lag_7": df.waste_units.iloc[-7],
       "roll7": df.waste_units.iloc[-7:].mean(), "dow": next_date.dayofweek, "month": next_date.month,
       "is_weekend": int(next_date.dayofweek >= 5), "temp_max": temp, "rain_mm": rain, "is_holiday": int(hol)}
pred = float(model.predict(pd.DataFrame([row])[feats])[0])
usual_prep = df.prepared.tail(30).mean()

c1, c2, c3 = st.columns(3)
c1.metric("Forecast date", next_date.strftime("%d %b %Y"))
c2.metric("Predicted waste (servings)", f"{pred:.0f}")
c3.metric("Suggested cut in preparation", f"{pred*0.8:.0f} servings",
          delta=f"-{pred*0.8/usual_prep*100:.1f}% of usual {usual_prep:.0f}")
if pred > df.waste_units.quantile(0.75):
    st.warning("High waste risk: reduce preparation or plan surplus redistribution.")

st.subheader("Recent waste trend")
st.plotly_chart(px.line(df.tail(90), x="date", y="waste_units"), use_container_width=True)
st.subheader("Average waste by weekday")
st.plotly_chart(px.bar(df.groupby("dow").waste_units.mean().reset_index(), x="dow", y="waste_units"),
                use_container_width=True)
