import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load model and feature names
model = joblib.load("linkedin_model.pkl")
feature_cols = joblib.load("feature_columns.pkl")

# ================================
# LOAD & CLEAN DATA FOR EDA (FIX)
# ================================
s = pd.read_csv("social_media_usage.csv")

def clean_sm(x):
    return np.where(x == 1, 1, 0)

ss = pd.DataFrame({
    "sm_li": clean_sm(s["web1h"]),
    "income": s["income"].where(s["income"] <= 9, np.nan),
    "education": s["educ2"].where(s["educ2"] <= 8, np.nan),
    "parent": s["par"],
    "married": s["marital"],
    "female": s["gender"].replace({1: 0, 2: 1}),
    "age": s["age"].where(s["age"] <= 98, np.nan)
})

ss = ss.dropna()

# ============================
# CUSTOM SIDEBAR THEME
# ============================
st.markdown(
    """
    <style>
    /* Sidebar background (matching Apple Midnight accent blue) */
    section[data-testid="stSidebar"] {
        background-color: #4DA3FF !important;
    }

    /* Inner sidebar container */
    section[data-testid="stSidebar"] > div {
        background-color: #4DA3FF !important;
    }

    /* Sidebar text (white for perfect contrast) */
    section[data-testid="stSidebar"] * {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Sidebar Navigation
page = st.sidebar.radio(
    "Navigate",
    [
        "Who's a User?",
        "Exploratory Insights",
        "Make a Prediction"
    ]
)

# ========= THEME COLORS =========
ACCENT_BLUE = "#4DA3FF"      
ACCENT_GRAY = "#9AA5B1"      
BAR_GRAY = "#CBD5E1"         
BG_CARD = "rgba(255, 255, 255, 0.05)"

# Add chart container wrapper 

def chart_card(chart):
    st.markdown(
        f"""
        <div style="
            background:{BG_CARD};
            padding:18px;
            border-radius:12px;
            border:1px solid rgba(255,255,255,0.1);
            margin-bottom:30px;
        ">
        """,
        unsafe_allow_html=True
    )
    st.altair_chart(chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# PAGE 1: PROJECT INTRODUCTION (NO HTML/CSS)
# -------------------------
if page == "Who's a User?":

    # Override the font color

    st.markdown(
        """
        <style>
        .custom-title {
            color: #FFFFFF !important; 
            font-size: 46px !important;
            font-weight: 800 !important;
            margin-bottom: 8px !important;
            letter-spacing: -1px !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Hero-style header
    
    st.markdown(
        """
        <h1 class='custom-title' style='text-align:center; line-height: 1.1;'>
            Who's a User?<br>
            <span style='font-size:44px; font-weight:600;'>Predicting LinkedIn Usage</span>
         </h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    
    st.markdown(
        """
        ### A Machine-Learning Dashboard for LinkedIn Targeting

        LinkedIn is a widely used professional networking platform, but not everyone uses it.  
        We have set out to understand **which types of people** are most likely to be LinkedIn users so that the organization can make smarter decisions about where and how to run marketing campaigns.
        """
    )

    st.markdown("---")

    # Two-column layout for a more "dashboard" feel
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Project Objective")
        st.markdown(
            """
            - Use real survey data on social media habits in the U.S.  
            - Build a **logistic regression model** to predict whether someone uses LinkedIn.  
            - Focus on individual demographics and attributes, including:
              - Income  
              - Education  
              - Age  
              - Gender  
              - Marital status  
              - Parental status  
            """
        )

    with col2:
        st.subheader("Why This Matters for Marketing")
        st.markdown(
            """
            - Identify segments **most likely** to be active on LinkedIn.  
            - Prioritize platforms for future campaigns.  
            - Support **data-driven decisions** about audience targeting and messaging.  
            - Help leadership evaluate the tradeoffs between different digital channels.
            """
        )

    st.markdown("---")
    st.subheader("How to Use This App")

    st.markdown(
        """
        - **Exploratory Data Insights**  
          Explore charts that show how LinkedIn usage varies by income, education, age,
          gender, marital status, and parental status.

        - **Make a Prediction**  
          Enter a hypothetical person’s profile (income, education, age, etc.) and the app will:
          - Predict whether they are classified as a LinkedIn **User** or **Non-User**, and  
          - Provide the **probability** that this person uses LinkedIn.

        Use the navigation menu on the left to move between pages.
        """
    )

#Global Altair styling upgrade

alt.themes.enable("dark")  # Start with dark base

def alt_style(chart):
    return (
        chart.configure_view(
            fill="rgba(30, 30, 40, 0.6)",
            strokeWidth=0
        )
        .configure_axis(
            labelColor="#E2E8F0",
            titleColor="#FFFFFF",
            grid=False
        )
        .configure_title(
            color="#FFFFFF",
            fontSize=18,
            anchor="start"
        )
        .configure_legend(
            titleColor="#FFFFFF",
            labelColor="#FFFFFF",
            orient="top-left"
        )
    )

# -------------------------
# PAGE 2: EDA
# -------------------------
if page == "Exploratory Insights":
   
    st.markdown("<h1 class='eda-title'>Exploratory Data Insights</h1>", unsafe_allow_html=True)

    st.markdown("---")
    
    st.markdown("### Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Samples", "1,260")
    col2.metric("LinkedIn Users", "33.3%")
    col3.metric("Features Used", "6")

    st.markdown("---")
    st.markdown("### Key Insights")

    insight_cols = st.columns(3)

    insight_cols[0].info(
        "**Education is the Strongest Predictor**\n\n"
        "A one-unit increase in education increases odds of LinkedIn usage by 40%."
    )
    insight_cols[1].info(
        "**Income is Highly Influential**\n\n"
        "Higher-income individuals show more professional networking activity."
    )
    insight_cols[2].info(
        "**Model Stability**\n\n"
        "Low multicollinearity supports effective logistic regression."
    )

    st.markdown("---")
    st.markdown("### Visualizations Dashboard")

    # Georgetown University theme
    GU_BLUE = "#0055A4"     # brighter GU blue
    GU_GRAY = "#B6B8BA"     # vibrant gray
       # ---------- Income Chart (Altair) ----------
    ss["li_label"] = ss["sm_li"].map({0: "No", 1: "Yes"})
    
    income_chart = alt.Chart(ss).mark_bar(opacity=0.7).encode(
    x=alt.X("income:O", title="Income Level (1–9)"),
    y=alt.Y("count()", title="Count"),
    color=alt.Color(
        "li_label:N",
        title="LinkedIn User",
        scale=alt.Scale(
            domain=["No", "Yes"],
            range=[GU_GRAY, GU_BLUE]   # Not User = Gray, User = Blue
        )
    ),
    tooltip=[
        alt.Tooltip("income:O", title="Income Level"),
        alt.Tooltip("li_label:N", title="LinkedIn User"),
        alt.Tooltip("count()", title="Count")
    ]
).properties(
    title="Income Distribution: LinkedIn Users vs Non-Users",
    width=600,
    height=400
)

    st.subheader("LinkedIn Usage by Income")
    st.altair_chart(income_chart, use_container_width=True)


    # ---------- Education Chart (Altair) ----------
    ss["li_label"] = ss["sm_li"].map({0: "No", 1: "Yes"})

    # Interactive selection
    selection = alt.selection_point(fields=["li_label"])

    education_chart = alt.Chart(ss).mark_bar().encode(
    x=alt.X("education:O", title="Education Level (1–8)"),
    y=alt.Y("count()", title="Count"),
    color=alt.Color(
        "li_label:N",
        title="LinkedIn User",
        scale=alt.Scale(
            domain=["No", "Yes"],       # legend order
            range=[GU_GRAY, GU_BLUE]           # color selection
        )
    ),
    opacity=alt.condition(selection, alt.value(1), alt.value(0.25)),
    tooltip=[
        alt.Tooltip("education:O", title="Education Level"),
        alt.Tooltip("li_label:N", title="LinkedIn User"),
        alt.Tooltip("count()", title="Count")
    ]
    ).add_params(
        selection
    ).properties(
        title="Education Distribution: LinkedIn Users vs Non-Users",
        width=600,
        height=400
    )

    st.subheader("LinkedIn Usage by Education Level")
    st.altair_chart(education_chart, use_container_width=True)

    # ---------- Parental Chart (Altair) ----------

     # Prepare dataset for visualization
    ss_plot = ss.copy()

    # Normalize parent coding
    # If parent is coded 2 instead of 0, remap:
    ss_plot["parent"] = ss_plot["parent"].replace({2: 0})  # If needed

    # Keep only valid values
    ss_plot = ss_plot[ss_plot["parent"].isin([0,1])]

    # Convert to readable labels
    ss_plot["parent_label"] = ss_plot["parent"].map({0: "Not a Parent", 1: "Parent"})
    ss_plot["sm_li_label"] = ss_plot["sm_li"].map({0: "No", 1: "Yes"})

    parental_chart = alt.Chart(ss_plot).mark_bar().encode(
    x=alt.X("parent_label:N", title="Parental Status"),
    y=alt.Y("count()", title="Total Count"),
    color=alt.Color(
        "sm_li_label:N",
        title="LinkedIn User",
        scale=alt.Scale(domain=["No", "Yes"], range=[GU_GRAY, GU_BLUE])
    ),
    tooltip=[
        alt.Tooltip("parent_label:N", title="Parental Status"),
        alt.Tooltip("sm_li_label:N", title="LinkedIn User"),
        alt.Tooltip("count():Q", title="Total Count")
        ]
    ).properties(
    title="LinkedIn Usage by Parental Status (Total Count)",
    width=450, height=350
    )

    st.subheader("LinkedIn Usage by Parental Status")
    st.altair_chart(parental_chart, use_container_width=True)

    # ---------- Marital Chart (Altair) ----------

    # Prepare dataset for visualization
    ss_marital = ss.copy()

    # Convert marital status into married = 1, not married = 0
    ss_marital["married"] = ss_marital["married"].replace({
        1: 1,        # Married
        2: 0,        # Living with partner
        3: 0,        # Divorced
        4: 0,        # Separated
        5: 0,        # Widowed
        6: 0,         # Never married
        8: 0,	     # Don't know
    	9: 0,	     # Refused
    })

    ss_marital["married_label"] = ss_marital["married"].map({
        0: "Not Married",
        1: "Married"
    })

    ss_marital["sm_li_label"] = ss_marital["sm_li"].map({
        0: "No",
        1: "Yes"
    })

    marital_chart = alt.Chart(ss_marital).mark_bar().encode(
    x=alt.X(
        "married_label:N",
        title="Marital Status",
        scale=alt.Scale(domain=["Not Married", "Married"])
    ),
    y=alt.Y("count()", title="Total Count"),
    color=alt.Color(
        "sm_li_label:N",
        title="LinkedIn User",
        scale=alt.Scale(domain=["No", "Yes"], range=[GU_GRAY, GU_BLUE])
    ),
    tooltip=[
        alt.Tooltip("married_label:N", title="Marital Status"),
        alt.Tooltip("sm_li_label:N", title="LinkedIn User"),
        alt.Tooltip("count():Q", title="Total Count")
    ]
).properties(
    title="LinkedIn Usage by Marital Status (Total Count)",
    width=450,
    height=350
)

    st.subheader("LinkedIn Usage by Marital Status")
    st.altair_chart(marital_chart, use_container_width=True)

    # ---------- Gender Chart (Altair) ----------

        # Prepare dataset for visualization
    ss_gender = ss.copy()

    # Map readable labels
    ss_gender["gender_label"] = ss_gender["female"].map({
        0: "Male",
        1: "Female"
    })

    ss_gender["sm_li_label"] = ss_gender["sm_li"].map({
        0: "No",
        1: "Yes"
    })

    gender_chart = alt.Chart(ss_gender).mark_bar().encode(
    x=alt.X(
        "gender_label:N",
        title="Gender",
        scale=alt.Scale(domain=["Male", "Female"])
    ),
    y=alt.Y("count()", title="Total Count"),
    color=alt.Color(
        "sm_li_label:N",
        title="LinkedIn User",
        scale=alt.Scale(
            domain=["No", "Yes"],
            range=[GU_GRAY, GU_BLUE]
        )
    ),
    tooltip=[
        alt.Tooltip("gender_label:N", title="Gender"),
        alt.Tooltip("sm_li_label:N", title="LinkedIn User"),
        alt.Tooltip("count():Q", title="Total Count")
    ]
    ).properties(
        title="LinkedIn Usage by Gender (Total Count)",
        width=450,
        height=350
    )

    st.subheader("LinkedIn Usage by Gender")
    st.altair_chart(gender_chart, use_container_width=True)

    # ---------- Age Chart (Altair) ----------

    # Prepare dataset for visualization
    ss_age = ss.copy()
    ss_age["sm_li_label"] = ss_age["sm_li"].map({0:"No", 1:"Yes"})

    # GU colors
    GU_BLUE = "#0055A4"
    GU_GRAY = "#B6B8BA"

    age_chart = alt.Chart(ss_age).mark_bar(opacity=0.75).encode(
    x=alt.X("age:Q", bin=alt.Bin(maxbins=20), title="Age"),
    y=alt.Y("count()", title="Count"),
    color=alt.Color(
        "sm_li_label:N",
        title="LinkedIn User",
        scale=alt.Scale(domain=["No", "Yes"], range=[GU_GRAY, GU_BLUE])
    ),
    tooltip=[
        alt.Tooltip("age:Q", title="Age"),
        alt.Tooltip("sm_li_label:N", title="LinkedIn User"),
        alt.Tooltip("count():Q", title="Count")
    ]
    ).properties(
        width=550,
        height=350,
        title="Age Distribution: LinkedIn Users vs Non-Users"
    )

    st.subheader("LinkedIn Usage Based on Age")
    st.altair_chart(age_chart, use_container_width=True)

    # ---------- Age Histogram (Seaborn) ----------

    # Georgetown University theme
    GU_BLUE = "#0055A4"     # brighter GU blue
    GU_GRAY = "#B6B8BA"     # vibrant gray

    # Palette for histogram bars (0 = Non-user, 1 = User)
    gu_palette = [GU_GRAY, GU_BLUE]

    # Histogram with red & green KDE lines
    sns.histplot(
        data=ss,
        x="age",
        hue="sm_li",
        bins=20,
        palette=gu_palette,
        kde=True
    )

    # Change KDE line colors
    ax = plt.gca()
    lines = ax.get_lines()

    # KDE lines appear last. Ensure at least two exist.
    if len(lines) >= 2:
        lines[0].set_color("red")     # KDE line for NON-LinkedIn users
        lines[1].set_color("green")   # KDE line for LinkedIn users

    # Build a custom legend explaining everything
    custom_handles = [
        plt.Line2D([], [], color=GU_GRAY, marker="s", linestyle="", markersize=10, label="Non-User (Bars)"),
        plt.Line2D([], [], color=GU_BLUE, marker="s", linestyle="", markersize=10, label="User (Bars)"),
        plt.Line2D([], [], color="red", linestyle="solid", label="Non-User KDE (Red Line)"),
        plt.Line2D([], [], color="green", linestyle="solid", label="User KDE (Green Line)")
    ]

    ax.legend(handles=custom_handles, title="Legend")

    plt.title("Age Distribution: LinkedIn Users vs Non-Users", color=GU_BLUE)
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.show()

# -------------------------
# PAGE 3: MAKE A PREDICTION
# -------------------------
if page == "Make a Prediction":
    st.title("LinkedIn User Prediction Tool")

    ACCENT = "#0054A6"  # Georgetown Blue

    # ---------------------------
    # Income Dropdown
    # ---------------------------
    income_options = {
    "Less than $10,000": 1,
    "$10,000–$20,000": 2,
    "$20,000–$30,000": 3,
    "$30,000–$40,000": 4,
    "$40,000–$50,000": 5,
    "$50,000–$75,000": 6,
    "$75,000–$100,000": 7,
    "$100,000–$150,000": 8,
    "$150,000+": 9
    }

    income_label = st.selectbox("Income Level", list(income_options.keys()))
    income = income_options[income_label]

    # ---------------------------
    # Education Dropdown
    # ---------------------------
    education_options = {
    "Less than high school": 1,
    "High school incomplete": 2,
    "High school graduate": 3,
    "Some college, no degree": 4,
    "Two-year associate degree": 5,
    "Four-year college degree": 6,
    "Some graduate school, no degree": 7,
    "Graduate degree (MA, MS, PhD, MD, JD)": 8
    }

    education_label = st.selectbox("Education Level", list(education_options.keys()))
    education = education_options[education_label]
    
    parent = st.radio("Are You a Parent?", ["No", "Yes"])
    married = st.radio("Are You Married?", ["No", "Yes"])
    gender = st.radio("What is Your Gender?", ["Male", "Female"])

    # Age input method
    st.subheader("Age Input")
    age_method = st.radio("How do you want to enter age?", ["Enter Age", "Enter Birthday"])

    if age_method == "Enter Age":
        age = st.number_input("Age", min_value=18, max_value=98, value=30)
    else:
        dob = st.date_input("Enter Date of Birth")
        age = int((pd.Timestamp.today() - pd.Timestamp(dob)).days / 365.25)

    # Predict button
    if st.button("Predict LinkedIn Usage"):
        input_data = np.array([[
            income,
            education,
            0 if parent == "No" else 1,
            0 if married == "No" else 1,
             0 if gender == "Male" else 1,
            age
        ]])

        prob = model.predict_proba(input_data)[0][1]
        pred = "User" if prob >= 0.5 else "Non-User"

        # NEW COLORS
        if pred == "User":
            glow_color = "#32CD32"   # Green
            pred_color = "#32CD32"
        else:
            glow_color = "#FFD700"   # Yellow (Gold)
            pred_color = "#FFD700"

        # Prediction card with glow
        st.markdown(
            f"""
            <style>
            @keyframes glow {{
                0% {{ box-shadow: 0 0 8px {glow_color}; }}
                50% {{ box-shadow: 0 0 22px {glow_color}; }}
                100% {{ box-shadow: 0 0 8px {glow_color}; }}
            }}
            .prediction-box {{
                animation: glow 2s infinite;
            }}
            </style>

            <div class="prediction-box" style="
                padding:20px;
                margin-top:20px;
                border-radius:15px;
                border:3px solid {glow_color};
                background-color:rgba(255, 255, 255, 0.08);
                backdrop-filter: blur(12px);
                text-align:center;
                font-size:22px;
                font-weight:600;
                color:white;
            ">
                Prediction: <span style="color:{pred_color}">{pred}</span><br>
                Probability: {prob*100:.0f}% 
            </div>
            """,
            unsafe_allow_html=True
        )


# Footer

st.markdown(
    "<hr><p style='text-align:center;color:gray'>LinkedIn Usage Prediction Model <br> Georgetown MSBA • 2025</p>",
    unsafe_allow_html=True
)

# Theme

st.markdown(
    """
    <style>

        /* MAIN BACKGROUND (Apple Midnight) */
        .stApp {
            background-color: #0B0E14 !important;
            color: #E5E7EB !important;
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", sans-serif;
        }

        html, body, [class*="st-"] {
            color: #E5E7EB !important;
        }

        /* TITLES — Apple neon blue */
        h1, h2, h3, h4 {
            color: #76B7FF !important;
            font-weight: 600 !important;
        }

        /* ------------------------------------------------------------
           BRIGHT FROSTED GLASS SIDEBAR + SOFT GLOW BORDER
           ------------------------------------------------------------ */
        section[data-testid="stSidebar"] {
            background: rgba(240, 244, 255, 0.65) !important;  /* bright glass tint */
            backdrop-filter: blur(32px) saturate(220%) brightness(145%) !important;
            -webkit-backdrop-filter: blur(32px) saturate(220%) brightness(145%) !important;

            /* Soft glowing border */
            border-right: 2px solid rgba(118, 183, 255, 0.55) !important;
            box-shadow:
                0 0 18px rgba(118, 183, 255, 0.35),   /* outer glow */
                inset 0 0 12px rgba(118, 183, 255, 0.15); /* inner soft glow */
        }

        /* SIDEBAR TEXT — original dark */
        section[data-testid="stSidebar"] * {
            color: #0B0E14 !important;
            font-weight: 600 !important;
        }

        /* BUTTONS */
        .stButton>button {
            background-color: #76B7FF !important;
            color: #0B0E14 !important;
            border-radius: 10px !important;
            padding: 10px 18px;
            font-size: 16px;
            font-weight: 600;
            border: none;
            box-shadow: 0 3px 10px rgba(118, 183, 255, 0.3);
            transition: 0.2s ease;
        }
        .stButton>button:hover {
            background-color: #A3CCFF !important;
            box-shadow: 0 6px 18px rgba(163, 204, 255, 0.4);
        }

        /* INPUTS (glass-style fields) */
        .stSelectbox div[data-baseweb="select"] > div,
        .stNumberInput input,
        .stDateInput input {
            background-color: rgba(255, 255, 255, 0.07) !important;
            backdrop-filter: blur(16px) !important;
            color: #FFFFFF !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            border-radius: 8px !important;
        }

        /* RADIO LABELS */
        div[role="radiogroup"] label {
            color: #0B0E14 !important;
        }

        /* METRICS */
        [data-testid="stMetricValue"] {
            color: #76B7FF !important;
            font-weight: 800 !important;
        }
        [data-testid="stMetricLabel"] {
            color: #9CA3AF !important;
        }

        /* CHART TITLES */
        .vega-typography,
        .mark-title text {
            fill: #76B7FF !important;
            font-weight: 600 !important;
        }

    </style>
    """,
    unsafe_allow_html=True
)

