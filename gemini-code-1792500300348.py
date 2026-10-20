import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Operations Dashboard", page_icon="☕", layout="wide")

try:
    from cafe import CafeSim
    from agent import train, get_best_action
except Exception as e:
    st.error("Missing cafe.py or agent.py files.")
    st.stop()

# --- UI Styling ---
st.markdown("""
    <style>
    .stApp { background-color: #F8F9FA; }
    h1, h2, h3 { color: #1E293B; font-family: 'Inter', sans-serif; }
    .stButton > button {
        background-color: #0F172A; color: white; border-radius: 6px;
        font-weight: 600; width: 100%; transition: 0.3s;
    }
    .stButton > button:hover { background-color: #334155; }
    .stAlert { border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

# --- State Management ---
if 'Q' not in st.session_state:
    with st.spinner("Initializing AI Core (Training model on 15,000 days of simulated data)..."):
        env = CafeSim()
        Q, curve = train(env, episodes=15000)
        st.session_state['Q'] = Q
        st.session_state['ledger'] = []
        st.session_state['env'] = env

Q = st.session_state['Q']
env = st.session_state['env']
days_map = {0:"Monday", 1:"Tuesday", 2:"Wednesday", 3:"Thursday", 4:"Friday", 5:"Saturday", 6:"Sunday"}
reverse_days = {v:k for k,v in days_map.items()}

# --- Header ---
st.title("☕ The Smart Café: AI Operations Manager")
st.markdown("Use this briefing to make optimized daily procurement decisions.")
st.markdown("---")

# --- Layout ---
col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.header("🌅 Morning Briefing")
    st.markdown("Enter today's real-world parameters to get your order recommendation.")
    
    with st.container(border=True):
        current_stock = st.number_input("Leftover Stock (Units)", min_value=0, max_value=15, value=3)
        today_str = st.selectbox("Day of the Week", list(days_map.values()), index=4) # Default Friday
        is_event = st.checkbox("🎉 Local Event / High Footfall Expected Today?")
        
        dow_int = reverse_days[today_str]
        event_int = 1 if is_event else 0
        
        # Get AI Recommendation
        ai_recommendation = get_best_action(Q, current_stock, dow_int, event_int)
        
        st.markdown(f"### **🤖 AI Recommendation: Order {ai_recommendation} units**")
        st.caption("This minimizes combined risk of spoilage costs and lost sales based on learned patterns.")
        
        st.markdown("---")
        st.markdown("#### Execute Order")
        actual_order = st.number_input("Override AI Recommendation (Optional)", min_value=0, max_value=15, value=ai_recommendation)
        
        if st.button("Submit Order & Advance Day"):
            # Simulate the day's result based on the manager's choice
            env.stock = current_stock
            env.day_of_week = dow_int
            env.event_today = event_int
            
            # Predict AI's hypothetical profit to compare against actual
            env_copy = CafeSim()
            env_copy.stock, env_copy.day_of_week, env_copy.event_today = current_stock, dow_int, event_int
            env_copy.step(ai_recommendation) # Run dummy step
            
            # Run actual step
            profit, next_state, _, info = env.step(actual_order)
            
            # Log results
            log_entry = {
                "Day": today_str,
                "Event": "Yes" if is_event else "No",
                "Starting Stock": current_stock,
                "AI Suggested": ai_recommendation,
                "You Ordered": actual_order,
                "Demand": info['demand'],
                "Spoiled": info['spoiled'],
                "Stockout": info['stockout'],
                "Profit (AED)": info['profit'],
                "AI Overridden?": "Yes" if ai_recommendation != actual_order else "No"
            }
            st.session_state['ledger'].append(log_entry)
            st.success(f"Day concluded! Made AED {info['profit']}.")
            st.rerun()

with col2:
    st.header("📊 Executive Ledger")
    
    if st.session_state['ledger']:
        df = pd.DataFrame(st.session_state['ledger'])
        
        # Metrics
        total_profit = df['Profit (AED)'].sum()
        total_lost_sales = df['Stockout'].sum()
        total_spoilage = df['Spoiled'].sum()
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Profit", f"AED {total_profit:,.0f}")
        m2.metric("Total Spoiled Units", total_spoilage)
        m3.metric("Missed Customers", total_lost_sales)
        
        st.markdown("#### Recent Operations History")
        st.dataframe(df.tail(10), use_container_width=True, hide_index=True)
        
    else:
        st.info("The ledger is empty. Process your first morning order on the left to begin tracking operations.")