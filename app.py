import streamlit as st
import sys
import os
import torch
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import time
import random
import plotly.express as px
import pickle      

# ==========================================
# 0. SYSTEM CONFIG
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Fix tqdm for Streamlit
try:
    import tqdm
except ImportError:
    from types import ModuleType
    dummy_tqdm = ModuleType("tqdm")
    def no_op_tqdm(iterable, *args, **kwargs): return iterable
    dummy_tqdm.tqdm = no_op_tqdm
    sys.modules["tqdm"] = dummy_tqdm

st.set_page_config(page_title="KraftgeneAI | Grid Guardian", layout="wide", page_icon="⚡")

# ==========================================
# 1. VISUAL STYLE
# ==========================================
st.markdown("""
<style>
    /* MAIN CONTAINER */
    .reportview-container { background: #050505; }
    .main { background-color: #050505; }
    
    /* BUTTONS */
    .stButton>button {
        width: 100%;
        border-radius: 6px;
        height: 3.5em;
        font-weight: 700 !important;
        border: 1px solid #333;
        background-color: #1f1f1f;
        color: #ddd;
        transition: all 0.2s;
    }
    .stButton>button:not(:disabled):hover {
        border-color: #FF4B4B;
        color: #FF4B4B;
        background-color: #261111;
    }
    
    /* ALERTS */
    .hazard-alert {
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 25px;
        text-align: center;
        font-weight: bold;
        font-size: 1.4em;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        animation: pulse-border 2s infinite;
    }
    @keyframes pulse-border {
        0% { box-shadow: 0 0 0 0 rgba(255, 75, 75, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(255, 75, 75, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 75, 75, 0); }
    }
    .hazard-critical { background: linear-gradient(90deg, #3a0000 0%, #590000 100%); border: 2px solid #ff4b4b; color: #ff9999; }
    .hazard-warning { background: linear-gradient(90deg, #3a2a00 0%, #594400 100%); border: 2px solid #ffcc4b; color: #ffeb99; }
    .hazard-safe { background: linear-gradient(90deg, #002b0e 0%, #004d1a 100%); border: 2px solid #4bff86; color: #ccffdb; }

    /* TEXT BOXES */
    .info-box {
        background-color: #111;
        padding: 15px;
        border-radius: 5px;
        border-left: 3px solid #FF4B4B;
        font-family: monospace;
        font-size: 0.9em;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. BULLETPROOF RESOURCES
# ==========================================
@st.cache_resource
def load_topology(path="grid_topology.pkl"):
    G = nx.Graph()
    positions = {}
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, list): data = data[0]

        raw_pos = data.get('positions', {})
        iter_pos = raw_pos.items() if hasattr(raw_pos, 'items') else enumerate(raw_pos)
        
        for k, v in iter_pos:
            try:
                if hasattr(k, 'item'): node_id = int(k.item())
                else: node_id = int(k)
                if hasattr(v, 'tolist'): coords = v.tolist()
                elif isinstance(v, np.ndarray): coords = v.tolist()
                else: coords = v
                if len(coords) >= 2: positions[node_id] = (float(coords[0]), float(coords[1]))
            except: pass 
        
        edge_index = data.get('edge_index', None)
        if edge_index is not None:
            if hasattr(edge_index, 'numpy'): edge_index = edge_index.numpy()
            if isinstance(edge_index, np.ndarray):
                src, dst = (edge_index[0], edge_index[1]) if edge_index.shape[0] == 2 else (edge_index[:, 0], edge_index[:, 1])
                for s, d in zip(src, dst): G.add_edge(int(s), int(d))
        G.add_nodes_from(positions.keys())
        if len(positions) == 0: positions = nx.spring_layout(G, seed=42)
        return G, positions
    except Exception as e:
        st.error(f"TOPOLOGY ERROR: {e}")
        G = nx.fast_gnp_random_graph(20, 0.2)
        return G, nx.spring_layout(G)

@st.cache_resource
def get_predictor():
    try:
        from inference import CascadePredictor
        MODEL_PATH = "best_f1_model.pth"
        if not os.path.exists(MODEL_PATH): return None
        return CascadePredictor(MODEL_PATH, "grid_topology.pkl", device="cuda" if torch.cuda.is_available() else "cpu", base_mva=100.0, base_freq=60.0)
    except: return None

# ==========================================
# 3. INTERACTIVE VISUALIZATION
# ==========================================
def draw_interactive_grid(G, pos, predicted_path_data, risk_scores, gt_path_data=None, title="Grid State"):
    if len(pos) == 0:
        if len(G.nodes()) > 0: pos = nx.spring_layout(G) 
        else: return go.Figure(layout=dict(title="Empty Grid Data"))

    # Map Node ID -> Sequence Order
    pred_map = {item['node_id']: item['order'] for item in predicted_path_data}
    gt_set = set()
    gt_map = {}
    if gt_path_data:
        for idx, item in enumerate(gt_path_data):
            gt_map[item['node_id']] = idx + 1
            gt_set.add(item['node_id'])

    # Edges
    edge_x, edge_y = [], []
    for edge in G.edges():
        u, v = int(edge[0]), int(edge[1])
        if u in pos and v in pos:
            edge_x.extend([pos[u][0], pos[v][0], None])
            edge_y.extend([pos[u][1], pos[v][1], None])

    base_edges = go.Scatter(
        x=edge_x, y=edge_y, 
        line=dict(width=1, color='rgba(60, 60, 60, 0.4)'), 
        hoverinfo='none', mode='lines', name='Lines'
    )

    # Trajectory
    path_x, path_y = [], []
    sorted_pred = sorted(predicted_path_data, key=lambda x: x['order'])
    for i in range(len(sorted_pred) - 1):
        u, v = sorted_pred[i]['node_id'], sorted_pred[i+1]['node_id']
        if u in pos and v in pos:
            path_x.extend([pos[u][0], pos[v][0], None])
            path_y.extend([pos[u][1], pos[v][1], None])

    trajectory_trace = go.Scatter(
        x=path_x, y=path_y,
        line=dict(width=4, color='rgba(255, 75, 75, 0.5)', dash='dot'),
        mode='lines', name='Flow', hoverinfo='none'
    )

    # Nodes
    node_x, node_y, node_colors, node_lines, node_sizes, node_texts = [], [], [], [], [], []
    def get_risk(n): return risk_scores.get(n, 0.0) if isinstance(risk_scores, dict) else (risk_scores[n] if n < len(risk_scores) else 0.0)

    for node_id, coords in pos.items():
        node_id = int(node_id)
        node_x.append(coords[0])
        node_y.append(coords[1])
        
        is_pred = node_id in pred_map
        is_gt = node_id in gt_set
        risk = get_risk(node_id)
        
        color, line_color, size = '#1f77b4', '#1f77b4', 6
        label = f"Node {node_id}<br>Status: Nominal"
        
        if gt_path_data is not None: 
            if is_pred and is_gt:
                color, line_color, size = '#FF4B4B', '#00FF00', 18
                label = f"<b>TRUE POSITIVE</b><br>Node {node_id}"
            elif is_pred and not is_gt:
                color, line_color, size = '#FFA500', '#FF4B4B', 14
                label = f"<b>FALSE ALARM</b><br>Node {node_id}"
            elif not is_pred and is_gt:
                color, line_color, size = '#050505', '#FF0000', 14
                label = f"<b>MISSED</b><br>Node {node_id}"
        else:
            if is_pred:
                color, line_color, size = '#FF4B4B', '#FFFFFF', 15
                label = f"<b>PREDICTED</b><br>Node {node_id}"
            elif risk > 0.4:
                color, line_color, size = '#FFA500', '#FFA500', 10

        node_colors.append(color)
        node_lines.append(line_color)
        node_sizes.append(size)
        node_texts.append(label)

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_texts,
        marker=dict(color=node_colors, size=node_sizes, line=dict(width=2, color=node_lines))
    )
    
    # Sequence Labels
    label_x, label_y, label_text = [], [], []
    for item in sorted_pred:
        nid = item['node_id']
        if nid in pos:
            label_x.append(pos[nid][0])
            label_y.append(pos[nid][1])
            label_text.append(str(item['order']))
            
    seq_trace = go.Scatter(
        x=label_x, y=label_y, mode='text', text=label_text,
        textfont=dict(color='white', size=10, family='Arial Black'), hoverinfo='none'
    )

    layout = go.Layout(
        title=dict(text=title, font=dict(color='white', size=16)),
        showlegend=False, hovermode='closest', paper_bgcolor='#111', plot_bgcolor='#111',
        margin=dict(b=0,l=0,r=0,t=40), xaxis=dict(visible=False), yaxis=dict(visible=False), height=550,
        dragmode='pan' # Enable panning by default
    )
    return go.Figure(data=[base_edges, trajectory_trace, node_trace, seq_trace], layout=layout)

def draw_radar(risk_vector):
    categories = ["Threat", "Vulnerability", "Impact", "Cascade", "Response", "Safety", "Urgency"]
    if risk_vector is None: risk_vector = [0]*7
    safe_vec = []
    for v in risk_vector:
        if isinstance(v, (list, np.ndarray)): safe_vec.append(float(np.mean(v)))
        else: safe_vec.append(float(v))
    if len(safe_vec) < 7: safe_vec += [0]*(7-len(safe_vec))
            
    fig = go.Figure(data=go.Scatterpolar(
        r=safe_vec, theta=categories, fill='toself', 
        line=dict(color='#FF4B4B', width=3), fillcolor='rgba(255, 75, 75, 0.2)'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, linecolor='#333'),
            angularaxis=dict(linecolor='#333', tickcolor='#333'), bgcolor='#111'
        ),
        paper_bgcolor='#111', font=dict(color='white'),
        margin=dict(l=40, r=40, t=20, b=20), showlegend=False, height=300
    )
    return fig

# ==========================================
# 4. APP LOGIC
# ==========================================
def main():
    if os.path.exists("eelogo.JPG"): st.sidebar.image("eelogo.JPG", width=150)
    else: st.sidebar.title("KraftgeneAI")
    
    st.sidebar.markdown("### 🎛️ Operator Controls")

    # State
    if 'window_start' not in st.session_state: st.session_state['window_start'] = 0
    if 'window_end' not in st.session_state: st.session_state['window_end'] = 0
    if 'last_result' not in st.session_state: st.session_state['last_result'] = None
    if 'is_processing' not in st.session_state: st.session_state['is_processing'] = False
    
    predictor = get_predictor()
    G, positions = load_topology("grid_topology.pkl")
    if not predictor: st.error("AI Model Offline - Check Server"); return

    # Data Source
    DATA_FOLDER = "data"
    if not os.path.exists(DATA_FOLDER): os.makedirs(DATA_FOLDER)
    data_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.pkl')]
    
    if not data_files:
        st.sidebar.error("No .pkl files found!")
        selected_file = None
    else:
        selected_file = st.sidebar.selectbox("Select Scenario File", data_files)

    st.sidebar.divider()
    debug_mode = st.sidebar.toggle("🛠️ Engineer Mode (Debug)", value=False)
    st.sidebar.divider()

    st.sidebar.markdown("---")
    st.sidebar.info("💻 **Project Source Code**\n\n[View on GitHub](https://github.com/KraftgeneAI)")

    # Simulation Button
    def run_simulation():
        st.session_state['is_processing'] = True
        window_size = 30
        current_start = st.session_state['window_start']
        if current_start >= 120: st.session_state['window_start'] = 0
        else: st.session_state['window_start'] = current_start + 1
        st.session_state['window_end'] = st.session_state['window_start'] + window_size

    st.sidebar.button("▶ NEXT TIME STEP (Slide Window)", disabled=st.session_state['is_processing'] or selected_file is None, on_click=run_simulation)

    # Processing
    if st.session_state['is_processing'] and selected_file:
        s_t, e_t = st.session_state['window_start'], st.session_state['window_end']
        with st.status(f"Processing [t={s_t} → t={e_t}]...", expanded=True) as status:
            res = predictor.predict_scenario("data", selected_file, start_step=s_t, end_step=e_t)
            time.sleep(0.1)
            st.session_state['last_result'] = res
            st.session_state['is_processing'] = False
            status.update(label="✅ Complete", state="complete", expanded=False)
            st.rerun()

    # Render
    res = st.session_state['last_result']
    if res is None:
        st.title("System Standby")
        st.plotly_chart(draw_interactive_grid(G, positions, [], [], None), width="stretch", config={'scrollZoom': True})
        return

    # Data Extraction
    pred_path_data = res.get('cascade_path', [])
    gt_data = res.get('ground_truth', {})
    gt_path_data = gt_data.get('cascade_path', []) if gt_data else None
    pred_failures = set([x['node_id'] for x in pred_path_data])
    actual_failures = set(gt_data.get('failed_nodes', [])) if gt_data else set()
    node_risks = [0.0] * 2000
    for x in pred_path_data: node_risks[x['node_id']] = x['ranking_score']

    # Banner
    num_failures = len(pred_failures)
    if num_failures == 0: detection = {"type": "SYSTEM NOMINAL", "style": "hazard-safe", "desc": "No cascading risks detected."}
    elif num_failures < 3: detection = {"type": "GRID INSTABILITY", "style": "hazard-warning", "desc": "Minor localized failures."}
    else: detection = {"type": "CRITICAL CASCADE", "style": "hazard-critical", "desc": f"Major Event: {num_failures} Nodes Impacted"}

    st.markdown(f"""<div class="hazard-alert {detection['style']}">⚠ {detection['type']}<br><span style='font-size:0.6em; opacity:0.8'>{detection['desc']}</span></div>""", unsafe_allow_html=True)

    # 4. METRICS ROW (Enhanced with Comparisons)
    c1, c2, c3, c4 = st.columns(4)
    
    # Fetch Data
    freq = res.get('system_state', {}).get('frequency', 60.0)
    prob = res.get('cascade_probability', 0.0)
    # Get the model's threshold (default to 0.1 if missing)
    thresh = getattr(predictor, 'cascade_threshold', 0.1)
    
    # Counts for comparisons
    num_pred = len(pred_failures)
    num_actual = len(actual_failures)
    
    # 1. Frequency: Compare Current vs Nominal (60Hz)
    with c1: 
        st.metric(
            label="Grid Freq (Current / Nom)", 
            value=f"{freq:.2f} / 60.0 Hz", 
            delta=f"{freq-60:.2f} Hz"
        )
    
    # 2. Confidence: Compare Prediction vs Threshold
    with c2: 
        st.metric(
            label="Confidence (Pred / Thresh)", 
            value=f"{prob*100:.1f}% / {thresh*100:.0f}%",
            delta=f"{(prob-thresh)*100:.1f}%"
        )
    
    # 3. Nodes Failed: Logic depends on Mode
    with c3:
        if debug_mode:
            # Engineer Mode: Compare Predicted vs Ground Truth
            st.metric(
                label="Nodes Failed (Pred / GT)", 
                value=f"{num_pred} / {num_actual}",
                delta=f"{num_pred - num_actual} Diff",
                delta_color="inverse"
            )
        else:
            # Operator Mode: Show only Predicted (Standard)
            st.metric("Nodes Failed (Predicted)", f"{num_pred}")

    # 4. Window Info
    with c4:
        st.metric("Window Size", f"{st.session_state['window_end'] - st.session_state['window_start']} steps")

    # ==========================================
    # 5. MAIN VISUALIZATION (ENGINEER VS OPERATOR)
    # ==========================================
    if debug_mode:
        st.markdown("### 🛠️ Engineer Diagnostics")
        
        # --- A. SPLIT VIEW VISUALIZATION ---
        col_viz, col_data = st.columns([2, 1])
        
        with col_viz:
            st.subheader("Visual Ground Truth Overlay")
            st.caption("Green Border = Correct | Red Border = Missed | Orange = False Alarm")
            
            # Pass gt_path_data to trigger comparison mode
            fig = draw_interactive_grid(
                G, positions, 
                pred_path_data, 
                node_risks, 
                gt_path_data, 
                title="Model vs. Reality"
            )
            st.plotly_chart(fig, width="stretch", config={'scrollZoom': True, 'displayModeBar': True})

        with col_data:
            st.subheader("Confusion Matrix")
            # Calculate metrics locally for this view
            tp = len(pred_failures.intersection(actual_failures))
            fp = len(pred_failures - actual_failures)
            fn = len(actual_failures - pred_failures)
            
            matrix_data = pd.DataFrame(
                [[tp, fp], [fn, "N/A"]], 
                columns=["Pred Positive", "Pred Negative"],
                index=["Actual Positive", "Actual Negative"]
            )
            # FIX: Convert to string to avoid PyArrow error with mixed int/str types
            st.dataframe(matrix_data.astype(str))
            
            c_prec, c_rec = st.columns(2)
            c_prec.metric("Precision", f"{tp / (tp + fp + 1e-9):.2f}")
            c_rec.metric("Recall", f"{tp / (tp + fn + 1e-9):.2f}")

        # --- B. RAW INSPECTOR & PHYSICS ---
        with st.expander("🔍 Deep Dive: Probabilities & Physics", expanded=False):
            
            # 1. Physics Analysis
            st.markdown("#### Signal Analysis")
            c1, c2 = st.columns(2)
            with c1:
                # Voltage Histogram
                volts = res.get('system_state', {}).get('voltages', [])
                if volts:
                    flat_v = [v for sub in volts for v in (sub if isinstance(sub, list) else [sub])] if isinstance(volts, list) else volts
                    fig_hist = px.histogram(x=flat_v, nbins=20, labels={'x': 'Voltage (p.u.)'}, title="Voltage Distribution", template='plotly_dark')
                    fig_hist.update_layout(bargap=0.1, height=200, margin=dict(t=30,b=0,l=0,r=0))
                    st.plotly_chart(fig_hist, width="stretch")
            
            with c2:
                st.write("**Drift Indicators**")
                st.metric("Freq Deviation", f"{abs(freq - 60.0):.3f} Hz", delta_color="inverse")
            
            # 2. Node Table
            st.markdown("#### Node-Level Probabilities")
            inspection_data = []
            all_relevant_nodes = pred_failures.union(actual_failures)
            
            for nid in all_relevant_nodes:
                score = node_risks[nid] if nid < len(node_risks) else 0.0
                is_fail = nid in actual_failures
                is_pred = nid in pred_failures
                
                status = "✅ Correct" if (is_fail and is_pred) else \
                         "❌ Missed" if (is_fail and not is_pred) else \
                         "⚠️ False Alarm"
                
                inspection_data.append({
                    "Node ID": nid,
                    "Score": f"{score:.4f}",
                    "Status": status,
                    "Ground Truth": "Fail" if is_fail else "Safe"
                })
            
            if inspection_data:
                df_debug = pd.DataFrame(inspection_data).sort_values(by="Score", ascending=False)
                st.dataframe(df_debug, width="stretch", hide_index=True)

    else:
        # --- OPERATOR MODE (STANDARD VIEW) ---
        col_main, col_side = st.columns([3, 1])
        with col_main:
            # Pass None for GT to hide answers
            fig = draw_interactive_grid(G, positions, pred_path_data, node_risks, None, "Predicted Cascade Propagation")
            st.plotly_chart(fig, width="stretch", config={'scrollZoom': True, 'displayModeBar': True})

        with col_side:
            st.subheader("Risk Profile")
            st.plotly_chart(draw_radar(res.get('risk_assessment')), width="stretch")
            st.subheader("Voltage Stability")
            volts = res.get('system_state', {}).get('voltages', [])
            if volts:
                flat_v = [v for sub in volts for v in (sub if isinstance(sub, list) else [sub])] if isinstance(volts, list) else volts
                st.line_chart(pd.DataFrame({"Voltage (p.u.)": flat_v}), height=200)

    # --- DETAILED METRICS SECTION ---
    st.divider()
    col_perf, col_phys = st.columns(2)
    
    with col_perf:
        with st.expander("📋 Detailed Performance Report", expanded=True):
            # Calculate metrics
            tp = len(pred_failures.intersection(actual_failures))
            fp = len(pred_failures - actual_failures)
            fn = len(actual_failures - pred_failures)
            thresh = 0.35 # Fixed threshold from inference logic
            
            st.markdown(f"""
            **1. Overall Verdict**
            {'✅ Correctly detected a cascade.' if num_failures > 0 else '⚠️ No cascade detected.'}
            * Prediction Probability: `{prob:.3f}` (Threshold: 0.100)
            * Ground Truth: `{gt_data.get('is_cascade', 'Unknown')}`
            
            **2. Node-Level Analysis**
            * Predicted Nodes at Risk: `{len(pred_failures)}` (Threshold: {thresh})
            * Actual Failed Nodes: `{len(actual_failures)}`
                * ✅ Correctly Identified (TP): **{tp}**
                * ❌ Missed Nodes (FN): **{fn}**
                * ⚠️ False Alarms (FP): **{fp}**
            """)

    with col_phys:
        with st.expander("⚡ Physics & Risk Analysis", expanded=True):
            # Parse Risk Vector
            r = res.get('risk_assessment', [0]*7)
            def lvl(v): return "Critical" if v>0.8 else "Severe" if v>0.6 else "Medium" if v>0.3 else "Low"
            
            # Parse Physics
            phys_desc = "Nominal operation."
            if freq < 59.8: phys_desc = "⚠ **Under-Frequency:** High load/Low generation imbalance."
            elif freq > 60.2: phys_desc = "⚠ **Over-Frequency:** Low load/High generation imbalance."
            
            v_vals = res.get('system_state', {}).get('voltages', [])
            if v_vals:
                min_v = np.min(v_vals) if len(v_vals)>0 else 0
                if min_v < 0.9: phys_desc += f" ⚠ **Low Voltage ({min_v:.2f} p.u.):** Potential voltage collapse."

            st.markdown(f"""
            **Aggregated Risk Assessment (7-Dimensions):**
            * **Threat:** {r[0]:.3f} ({lvl(r[0])}) | **Vulnerability:** {r[1]:.3f} ({lvl(r[1])})
            * **Impact:** {r[2]:.3f} ({lvl(r[2])}) | **Cascade Prob:** {r[3]:.3f} ({lvl(r[3])})
            * **Response:** {r[4]:.3f} ({lvl(r[4])}) | **Urgency:** {r[6]:.3f} ({lvl(r[6])})
            
            **Physics Condition:**
            {phys_desc}
            
            ---
            **Risk Definitions:**
            * **Critical (0.8+):** Immediate Failure
            * **Severe (0.6+):** High Danger
            * **Dimensions:** Threat (Stress), Vulnerability (Weakness), Impact (Consequence).
            """)

    # ==========================================
    # 6. FOOTER
    # ==========================================
    st.markdown("""
    <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #0e1117;
            color: #888;
            text-align: center;
            font-size: 12px;
            padding: 10px;
            border-top: 1px solid #333;
            z-index: 100;
        }
    </style>
    <div class="footer">
        &copy; 2025 Kraftgene AI Inc. All Rights Reserved. | 
        <a href="https://github.com/KraftgeneAI" target="_blank" style="color: #FF4B4B; text-decoration: none;">GitHub Repository</a>
    </div>
    """, unsafe_allow_html=True)
if __name__ == "__main__":
    main()