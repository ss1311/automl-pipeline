
import os
import streamlit as st
import requests
import pandas as pd
import json
import time

# ============================================================
# CONFIGURATION
# ============================================================

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8010"
)

st.set_page_config(
    page_title="AutoML Pipeline Hub",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AutoML Pipeline Hub")

st.markdown(
    "Automated Preprocessing, Optuna Tuning, "
    "MLflow Tracking, Model Monitoring & Retraining"
)


# ============================================================
# SESSION STATE
# ============================================================

if "sample_json" not in st.session_state:

    st.session_state.sample_json = json.dumps(
        [
            {
                "tenure_months": 12,
                "total_orders": 10,
                "total_spend": 269.41,
                "support_calls": 2,
                "last_login_days_ago": 27
            }
        ],
        indent=2
    )

if "target_col" not in st.session_state:
    st.session_state.target_col = None

if "retrain_job_id" not in st.session_state:
    st.session_state.retrain_job_id = None

if "gate_metric" not in st.session_state:
    st.session_state.gate_metric = "Accuracy"


# ============================================================
# API HELPER
# ============================================================

def get_available_models():

    try:

        response = requests.get(
            f"{API_URL}/automl/models",
            timeout=10
        )

        if response.status_code == 200:

            return response.json().get(
                "models",
                []
            )

        return []

    except requests.RequestException:

        return []


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🚀 Train Model",
        "🔮 Predict",
        "📊 Model Insights",
        "🔄 Monitor & Retrain"
    ]
)


# ============================================================
# TAB 1: TRAIN MODEL
# ============================================================

with tab1:

    st.header("1. Upload Data & Train")

    uploaded_file = st.file_uploader(
        "Upload your dataset (.csv)",
        type=["csv"],
        key="train_file"
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.write("**Data Preview:**")

        st.dataframe(
            df.head(),
            use_container_width=True
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            target_col = st.selectbox(
                "Select Target Column",
                df.columns
            )

        with col2:

            exp_name = st.text_input(
                "Experiment Name",
                value="Ecommerce_Churn"
            )

        with col3:

            n_trials = st.number_input(
                "Number of Optuna Trials",
                min_value=1,
                max_value=100,
                value=10
            )

        if st.button(
            "Start Training 🚀",
            type="primary"
        ):

            st.session_state.target_col = target_col

            try:

                sample_row = (
                    df.drop(columns=[target_col])
                    .iloc[0]
                    .to_dict()
                )

                sample_row = {
                    k: v
                    for k, v in sample_row.items()
                    if k.lower() != "id"
                    and not k.lower().endswith("_id")
                }

                st.session_state.sample_json = json.dumps(
                    [sample_row],
                    indent=2,
                    default=str
                )

            except Exception:

                pass

            with st.spinner(
                "Sending data to backend..."
            ):

                uploaded_file.seek(0)

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file,
                        "text/csv"
                    )
                }

                data = {
                    "target_col": target_col,
                    "experiment_name": exp_name,
                    "n_trials": n_trials
                }

                try:

                    response = requests.post(
                        f"{API_URL}/automl/train",
                        files=files,
                        data=data,
                        timeout=30
                    )

                    if response.status_code == 200:

                        st.success(
                            "Training started successfully!"
                        )

                        st.info(
                            "💡 Check your backend terminal "
                            "for the training logs."
                        )

                    else:

                        st.error(
                            f"Error: {response.text}"
                        )

                except requests.RequestException as e:

                    st.error(
                        f"Connection failed: {e}"
                    )


# ============================================================
# TAB 2: PREDICT
# ============================================================

with tab2:

    st.header("2. Predict with Champion Model")

    col1, col2 = st.columns([1, 2])

    with col1:

        st.button(
            "🔄 Refresh Model List",
            key="refresh_predict"
        )

        available_models = get_available_models()

        if available_models:

            model_name = st.selectbox(
                "Select Trained Model",
                available_models,
                key="predict_model"
            )

        else:

            st.warning(
                "No models found. Train a model first."
            )

            model_name = st.text_input(
                "Model Name",
                value="Ecommerce_Churn_champion"
            )

        st.markdown(
            "**Input Features (JSON Format):**"
        )

        json_input = st.text_area(
            "Enter JSON array of records",
            value=st.session_state.sample_json,
            height=250
        )

        predict_btn = st.button(
            "Predict 🔮",
            type="primary"
        )

    with col2:

        if predict_btn:

            if not model_name:

                st.error(
                    "Please select a model."
                )

            else:

                try:

                    parsed_data = json.loads(
                        json_input
                    )

                    payload = {
                        "data": parsed_data
                    }

                    with st.spinner(
                        "Getting predictions..."
                    ):

                        response = requests.post(
                            f"{API_URL}/automl/predict/{model_name}",
                            json=payload,
                            timeout=30
                        )

                    if response.status_code == 200:

                        result = response.json()

                        st.success(
                            "Prediction Successful!"
                        )

                        res_df = pd.DataFrame(
                            parsed_data
                        )

                        res_df["Prediction"] = (
                            result["predictions"]
                        )

                        st.dataframe(
                            res_df,
                            use_container_width=True
                        )

                    else:

                        error_detail = (
                            response.json().get(
                                "detail",
                                "Unknown error"
                            )
                        )

                        st.error(
                            f"Error {response.status_code}: "
                            f"{error_detail}"
                        )

                except json.JSONDecodeError:

                    st.warning(
                        "Invalid JSON format."
                    )

                except requests.RequestException as e:

                    st.error(
                        f"Connection failed: {e}"
                    )

                except Exception as e:

                    st.error(
                        f"Prediction error: {e}"
                    )


# ============================================================
# TAB 3: MODEL INSIGHTS
# ============================================================

with tab3:

    st.header("3. Model Insights")

    st.markdown(
        "Explore registered models and MLflow experiment tracking."
    )

    available_models = get_available_models()

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Registered Models",
            len(available_models)
        )

    with col2:

        st.metric(
            "MLflow Tracking",
            "Configured"
        )

    st.markdown("---")

    st.subheader("Registered Models")

    if available_models:

        model_df = pd.DataFrame(
            available_models,
            columns=["Model Name"]
        )

        st.dataframe(
            model_df,
            use_container_width=True
        )

        selected_insight_model = st.selectbox(
            "Select Model",
            available_models,
            key="insight_model"
        )

        st.markdown("### Selected Model")

        st.code(
            selected_insight_model
        )

        if selected_insight_model.endswith(
            "_champion"
        ):

            st.success(
                "This is a Champion model."
            )

        elif selected_insight_model.endswith(
            "_candidate"
        ):

            st.info(
                "This is a Candidate model."
            )

        st.markdown(
            "### MLflow Experiment Tracking"
        )

        st.markdown(
            "Use the MLflow UI to inspect parameters, "
            "metrics, experiment runs and registered model versions."
        )

        st.markdown(
            "[Open MLflow Dashboard](http://127.0.0.1:5000)"
        )

    else:

        st.warning(
            "No registered models available. "
            "Train a model first."
        )


# ============================================================
# TAB 4: MONITORING & RETRAINING
# ============================================================

with tab4:

    st.header(
        "🔄 Model Monitoring & Retraining"
    )

    st.markdown(
        "Evaluate new labelled data, retrain candidate models "
        "and promote models through a quality gate."
    )

    available_models = get_available_models()

    if not available_models:

        st.warning(
            "No models found. Train a model first."
        )

    else:

        # ====================================================
        # TASK A & B: EVALUATE NEW DATA
        # ====================================================

        with st.container(border=True):

            st.subheader(
                "Task A & B: Evaluate Champion on New Batch"
            )

            col1, col2 = st.columns(2)

            with col1:

                mon_model = st.selectbox(
                    "Select Champion Model",
                    available_models,
                    key="mon_model"
                )

            with col2:

                mon_target = st.text_input(
                    "Target Column Name",
                    value=(
                        st.session_state.target_col
                        or "target"
                    ),
                    key="mon_target"
                )

            mon_file = st.file_uploader(
                "Upload New Labelled Batch (CSV)",
                type=["csv"],
                key="mon_file"
            )

            if st.button(
                "Evaluate New Data 📊",
                key="evaluate_new_data"
            ):

                if mon_file is None:

                    st.warning(
                        "Please upload a CSV file."
                    )

                else:

                    with st.spinner(
                        "Evaluating model on new data..."
                    ):

                        try:

                            files = {
                                "file": (
                                    mon_file.name,
                                    mon_file,
                                    "text/csv"
                                )
                            }

                            data = {
                                "model_name": mon_model,
                                "target_col": mon_target
                            }

                            response = requests.post(
                                f"{API_URL}/monitor/evaluate",
                                files=files,
                                data=data,
                                timeout=120
                            )

                            if response.status_code == 200:

                                result = response.json()

                                st.success(
                                    "Evaluation Complete!"
                                )

                                st.markdown(
                                    "### Data Summary"
                                )

                                st.json(
                                    result.get(
                                        "summary",
                                        {}
                                    )
                                )

                                st.markdown(
                                    "### Metrics on New Data"
                                )

                                metrics = result.get(
                                    "metrics",
                                    {}
                                )

                                if metrics:

                                    metric_cols = st.columns(
                                        len(metrics)
                                    )

                                    for i, (key, value) in enumerate(
                                        metrics.items()
                                    ):

                                        metric_cols[i].metric(
                                            key.upper(),
                                            f"{value:.4f}"
                                        )

                                    st.session_state.gate_metric = (
                                        list(metrics.keys())[0]
                                    )

                                else:

                                    st.warning(
                                        "No metrics returned."
                                    )

                            else:

                                try:

                                    error_detail = (
                                        response.json().get(
                                            "detail",
                                            response.text
                                        )
                                    )

                                except Exception:

                                    error_detail = response.text

                                st.error(
                                    f"Error: {error_detail}"
                                )

                        except requests.RequestException as e:

                            st.error(
                                f"Connection failed: {e}"
                            )


        # ====================================================
        # TASK C & D: RETRAIN + QUALITY GATE
        # ====================================================

        with st.container(border=True):

            st.subheader(
                "Task C & D: Retrain & Quality Gate"
            )

            st.info(
                "💡 This will combine the original data with "
                "the new batch, train a candidate model, "
                "and compare the metrics."
            )

            col1, col2 = st.columns(2)

            with col1:

                retrain_exp = st.text_input(
                    "Experiment Name",
                    value=mon_model.replace(
                        "_champion",
                        ""
                    ),
                    key="retrain_exp"
                )

            with col2:

                gate_metric = st.selectbox(
                    "Metric for Quality Gate",
                    [
                        "Accuracy",
                        "R2_Score",
                        "test_score"
                    ],
                    key="gate_metric_select"
                )

            n_trials_retrain = st.number_input(
                "Retraining Optuna Trials",
                min_value=1,
                max_value=100,
                value=3,
                key="retrain_trials"
            )

            if st.button(
                "Retrain Candidate Model 🚀",
                type="primary",
                key="start_retraining"
            ):

                if mon_file is None:

                    st.warning(
                        "Please upload a labelled CSV "
                        "in the evaluation section above."
                    )

                else:

                    with st.spinner(
                        "Uploading and starting retraining..."
                    ):

                        try:

                            mon_file.seek(0)

                            files = {
                                "file": (
                                    mon_file.name,
                                    mon_file,
                                    "text/csv"
                                )
                            }

                            data = {
                                "experiment_name": retrain_exp,
                                "target_col": mon_target,
                                "current_metric": gate_metric,
                                "n_trials": n_trials_retrain
                            }

                            response = requests.post(
                                f"{API_URL}/monitor/retrain",
                                files=files,
                                data=data,
                                timeout=60
                            )

                            if response.status_code == 200:

                                st.session_state.retrain_job_id = (
                                    response.json().get(
                                        "job_id"
                                    )
                                )

                                st.success(
                                    "Retraining started in background!"
                                )

                            else:

                                st.error(
                                    f"Error: {response.text}"
                                )

                        except requests.RequestException as e:

                            st.error(
                                f"Connection failed: {e}"
                            )


            # =================================================
            # LIVE RETRAINING STATUS
            # =================================================

            if st.session_state.retrain_job_id:

                st.markdown("---")

                st.markdown(
                    "### Retraining Status"
                )

                job_id = (
                    st.session_state.retrain_job_id
                )

                retrain_status = st.empty()

                status_placeholder = st.empty()

                check_status = st.button(
                    "🔄 Check Status",
                    key="check_retrain_status"
                )

                if check_status:

                    try:

                        response = requests.get(
                            f"{API_URL}/automl/status/{job_id}",
                            timeout=15
                        )

                        if response.status_code == 200:

                            info = response.json()

                            job_status = info.get(
                                "status",
                                "unknown"
                            )

                            job_message = info.get(
                                "message",
                                "No message"
                            )

                            if job_status == "running":

                                retrain_status.info(
                                    f"⏳ {job_message}"
                                )

                            elif job_status == "completed":

                                retrain_status.success(
                                    job_message
                                )

                                st.balloons()

                                st.session_state.retrain_job_id = None

                            elif job_status == "failed":

                                retrain_status.error(
                                    f"❌ {job_message}"
                                )

                                st.session_state.retrain_job_id = None

                            else:

                                status_placeholder.warning(
                                    f"Status: {job_status}"
                                )

                        else:

                            retrain_status.error(
                                f"Status API error: "
                                f"{response.status_code}"
                            )

                    except requests.RequestException as e:

                        retrain_status.error(
                            f"Connection failed: {e}"
                        )

                st.caption(
                    f"Job ID: {job_id}"
                )

                st.info(
                    "Click 'Check Status' to check the latest "
                    "retraining progress."
                )
