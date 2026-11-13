"""Connected Streamlit sections backed exclusively by executable data."""
from pathlib import Path
from dataclasses import asdict
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import streamlit as st
from .environment import (DeliveryMitigationEnv,EnvConfig,Order,generate_orders,SCENARIOS,
    ACTION_NAMES,STAGES,DISTANCES,STATE_DIMS,N_STATES,risk,valid_actions)
from .agents import TrainingConfig,QLearningAgent,SarsaAgent,load_agent
from .training import train
from .evaluation import evaluate,default_policies,robustness,choose
from . import charts

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["Overview","RL environment","Train the agents","Policy comparison","Live order simulator",
         "Q-table & policy","Scenario analysis","Research experiments","Methodology & limitations"]

def figure(fig):
    st.plotly_chart(fig,width="stretch")

def data_table(frame):
    """Accessible HTML keeps tables portable when native Arrow is unavailable.

    Explicit escaping prevents any user-controlled label becoming markup.
    CSV exports remain available for sorting or further analysis.
    """
    st.html("<div class='data-table'>"+frame.to_html(index=False,escape=True,border=0,na_rep="—")+"</div>")

def bootstrap():
    if "agents" not in st.session_state:
        st.session_state.agents = {}; st.session_state.curves = {}
        for name in ("q_learning","sarsa"):
            path = ROOT/"models"/f"{name}.npz"
            if path.exists():
                st.session_state.agents[name] = load_agent(path)
            curve_path = ROOT/"results"/f"{name}_training.csv"
            if curve_path.exists() and name in st.session_state.agents:
                st.session_state.curves[name] = pd.read_csv(curve_path)
        improved_metadata=ROOT/"results"/"improved_training"/"metadata.json"
        if improved_metadata.exists():
            metadata=json.loads(improved_metadata.read_text(encoding="utf-8"))
            if metadata.get("status")=="complete":
                # First predeclared seed, never a best-test-seed selection.
                seed=metadata["confirmation_seeds"][0]
                for algorithm,setting in metadata["selected_settings"].items():
                    model_path=ROOT/"models"/"improved_training"/f"{algorithm}_{setting}_seed{seed}.npz"
                    if model_path.exists():
                        st.session_state.agents[f"{algorithm} · extended training · seed {seed}"]=load_agent(model_path)

def model_selector(key):
    agents = st.session_state.agents
    if not agents:
        st.info("Train an agent in ‘Train the agents’ first, or run scripts/run_experiments.py and restart the app.")
        return None
    names=list(agents)
    preferred=next((i for i,name in enumerate(names) if "extended training" in name),0)
    name = st.selectbox("Learned policy",names,index=preferred,key=key)
    agent = agents[name]
    st.caption(f"Model: {agent.config.episodes:,} training episodes · seed {agent.config.seed} · {agent.env_config.scenario} · γ={agent.config.gamma} · visit-decay exponent {agent.config.visit_decay}")
    if agent.ensemble_members:
        st.caption(f"Explicit mean-Q ensemble of {len(agent.ensemble_members)} independently trained models. Episode count above is per member.")
    return agent

def overview():
    st.markdown('<div class="eyebrow">OPERATIONS RESEARCH / SYNTHETIC LOGISTICS LAB</div>',unsafe_allow_html=True)
    st.title("A better delivery decision.\nAt every stage.")
    st.markdown("### Cost-Aware Delivery Delay Mitigation Using Reinforcement Learning")
    st.write("Which sequence of interventions offers the best balance between fewer late deliveries and the cost of helping them arrive sooner?")
    columns = st.columns(3)
    for col,value,label in zip(columns,["3","2","1"],["Sequential decisions per order","Learning algorithms, implemented from scratch","Economic objective: minimise total cost"]):
        col.metric(label,value)
    st.markdown("<div class='flow'>SYNTHETIC ORDER <span>→</span> OBSERVE STATE <span>→</span> CHOOSE ACTION <span>→</span> SIMULATE & LEARN</div>",unsafe_allow_html=True)
    left,right = st.columns([1.25,1])
    with left:
        st.subheader("Spend where it makes a difference")
        st.write("Priority processing can help in the warehouse. Expedited transportation can help on the road. A limited budget makes timing matter: an early purchase changes both the delivery outlook and the options available later.")
        st.markdown("**Approved → Preparing → Dispatched → Delivered**")
        st.write("Explore the environment, train a policy, compare held-out outcomes, then walk an individual order through its decisions.")
    with right:
        st.subheader("The economic objective")
        st.latex(r"\min_\pi\;\mathbb{E}_\pi\left[\sum_{t=0}^{2}c(a_t)+15\max(0,L_3)+20\mathbf{1}_{L_3>0}\right]")
        st.caption("Illustrative Brazilian reais (R$). All costs and intervention effects are configurable simulation assumptions.")
    st.info("Synthetic simulation only. These policies learn the simulator’s cost trade-offs; their effectiveness has not been established in real logistics operations.")
    improvement_file=ROOT/"results"/"improved_training"/"final_summary.csv"
    if improvement_file.exists():
        st.subheader("Latest extended training · fresh 5,000-order test")
        extended=pd.read_csv(improvement_file)
        figure(charts.style(px.bar(extended,x="policy",y="average_cost",color="policy",error_y="cost_seed_ci95",
            title="Validation-selected 200,000-episode policies",
            labels={"average_cost":"Mean economic cost (R$/order)","policy":"Policy"})))
        st.caption("Five independent training seeds. Individual learned policies, unchanged simulator and baselines. Small mean differences need the paired uncertainty analysis in Research experiments; the ranking is not a universal performance guarantee.")
    research_folder=ROOT/"results"/"studies"
    if (research_folder/"metadata.json").exists():
        study_metadata=json.loads((research_folder/"metadata.json").read_text(encoding="utf-8"))
        if study_metadata.get("status")=="complete" and (research_folder/"final_summary.csv").exists():
            st.subheader("Repeated-seed research · fresh final test")
            final=pd.read_csv(research_folder/"final_summary.csv")
            figure(charts.style(px.bar(final,x="policy",y="average_cost",color="policy",error_y="cost_seed_ci95",
                title="Fresh held-out cost across independent training seeds",
                labels={"average_cost":"Mean economic cost (R$/order)","policy":"Policy"})))
            st.caption(f"{len(study_metadata['training_seeds'])} training seeds · {study_metadata['evaluation_orders']} final-test orders · settings selected on a separate validation set. Error bars show training-seed variability conditional on the shared test orders. Open Research experiments for coverage, full protocol and economic sensitivity.")
    if (ROOT/"results"/"evaluation_summary.csv").exists():
        st.subheader("Original single-seed experiment")
        summary = pd.read_csv(ROOT/"results"/"evaluation_summary.csv")
        figure(charts.comparison(summary))
        st.caption("Saved sample experiment. See results/experiment_metadata.json for its protocol. Current session models are evaluated separately on Policy comparison.")

def order_inputs(prefix):
    c1,c2,c3 = st.columns(3)
    outlook = c1.slider("Signed delivery outlook (days)",-3,8,4,key=prefix+"outlook")
    distance = c2.selectbox("Distance",DISTANCES,index=2,key=prefix+"distance")
    load = c3.selectbox("Operational load",["Normal","High"],index=1,key=prefix+"load")
    budget = c1.slider("Intervention budget (R$)",0,50,40,key=prefix+"budget")
    scenario = c2.selectbox("Effectiveness scenario",list(SCENARIOS),index=1,key=prefix+"scenario")
    seed = c3.number_input("Simulation seed",0,2**31-1,7,key=prefix+"seed")
    if st.checkbox("Configure initial risk instead of delivery outlook",key=prefix+"risk_mode"):
        requested_risk = st.slider("Initial delay risk / pressure",.01,.99,.72,.01,key=prefix+"risk")
        implied = 1.7*np.log(requested_risk/(1-requested_risk))-.25*DISTANCES.index(distance)-.35*int(load=="High")+.5
        outlook = int(np.clip(round(implied),-3,8))
        st.caption("Risk input is mapped to the nearest supported integer-day outlook. The actual coupled risk and expected lateness are shown below; independent inconsistent combinations are excluded.")
    order = Order("MANUAL",outlook,DISTANCES.index(distance),int(load=="High"),budget)
    st.caption(f"Derived initial delay pressure: {risk(outlook,order.distance,order.load):.1%} · projected late days: {max(0,outlook)}. Risk is coupled to outlook, distance and load; negative outlook represents early-delivery slack.")
    return order,EnvConfig(scenario),int(seed)

def environment_page():
    st.title("A small, explainable delivery world")
    st.write("One order. Three decisions. No repeated purchases. Every transition changes a single signed delivery trajectory.")
    st.subheader("Inspect the actions available to an order")
    order,config,seed = order_inputs("env")
    stage = st.select_slider("Stage",options=[0,1,2],format_func=lambda s:STAGES[s])
    services = st.selectbox("Purchased services",[0,1,2,3],format_func=lambda n:["None","Priority","Transport","Both"][n])
    env = DeliveryMitigationEnv(config); env.reset(seed,order); env.stage=stage; env.services=services
    data_table(pd.DataFrame([{"Action":name,"Cost (R$)":config.costs[a],"Available":a in env.get_valid_actions()} for a,name in enumerate(ACTION_NAMES)]))
    st.caption("Diagnostic states may be unreachable. Dispatch applies a previously purchased transport service once, without charging again.")
    st.subheader("State design without hidden budget or service status")
    st.code("(stage, signed_outlook + 3, operational_load, distance, exact_remaining_budget, service_bits)")
    st.write(f"72 states is the conceptual reference (3 × 3 × 4 × 2). The implementation uses {N_STATES:,} allocated states: 3 × 12 × 2 × 3 × 51 × 4, with {4*N_STATES:,} Q entries. Many combinations are unreachable. Risk and lateness bands are derived from these finer observations.")
    st.subheader("Moderate assumptions")
    st.write("Priority succeeds with probability 65%; transport 80%. Both services together use 90% per component. Priority saves approximately one day when successful; transport saves 1–3 days at dispatch. Scenario multipliers change integer-day savings through stochastic rounding.")
    st.write("Disruption probability = scenario base + 0.10 × high load + 0.07 × distance category at dispatch (capped at 0.85). A disruption adds 1–2 days; a draw above 0.92 gives a one-day recovery. Otherwise outlook is unchanged. Outlook is clipped to [−3, 8] days.")
    st.latex(r"\rho(L,d,h)=\operatorname{sigmoid}((L+0.25d+0.35h-0.5)/1.7)")
    st.caption("ρ is an illustrative risk/pressure proxy, not an independently sampled terminal outcome or calibrated lateness probability.")

def training_page():
    st.title("Watch a policy learn")
    st.write("Each episode generates a fresh synthetic order. The agent explores valid actions, receives real simulated costs, and updates its Q-table.")
    with st.form("training"):
        a,b,c = st.columns(3)
        algorithm = a.selectbox("Algorithm",["q_learning","sarsa"])
        episodes = b.number_input("Training episodes",100,500000,15000,100)
        scenario = c.selectbox("Training scenario",list(SCENARIOS),index=1)
        alpha = a.slider("Learning rate α",.01,1.,.1,.01)
        gamma = b.slider("Discount factor γ",0.,1.,1.,.05)
        seed = c.number_input("Training seed",0,2**31-1,42)
        start = a.slider("Starting epsilon",0.,1.,1.,.05)
        minimum = b.slider("Minimum epsilon",0.,1.,.05,.01)
        decay = c.slider("Fraction of episodes used for decay",.1,1.,.85,.05)
        visit_decay = a.slider("Learning-rate visit-decay exponent",0.,1.,0.,.05,
            help="0 keeps a constant rate. Positive values use α / visit_count^exponent separately for each state-action pair.")
        submitted = st.form_submit_button("Train policy",type="primary")
    if submitted:
        if minimum > start:
            st.error("Minimum epsilon must not exceed starting epsilon."); return
        cfg = TrainingConfig(int(episodes),alpha,gamma,start,minimum,decay,int(seed),visit_decay)
        cls = QLearningAgent if algorithm=="q_learning" else SarsaAgent
        agent = cls(cfg,EnvConfig(scenario)); progress = st.progress(0.)
        curve = train(agent,progress.progress)
        st.session_state.agents[algorithm] = agent; st.session_state.curves[algorithm] = curve
        for key in ("evaluation","robustness","simulation"):
            st.session_state.pop(key,None)
        st.success(f"Trained {episodes:,} episodes. Evaluation and live simulation caches have been cleared.")
    curves = st.session_state.curves
    if curves:
        figure(charts.learning(curves)); figure(charts.learning(curves,"return")); figure(charts.learning(curves,"epsilon"))
        for name,frame in curves.items():
            with st.expander(f"{name}: training diagnostics & export"):
                agent = st.session_state.agents[name]
                st.json({"last_250_mean_cost":float(frame.total_cost.tail(250).mean()),
                          "visited_states":int(np.count_nonzero(agent.visits.sum(axis=1))),
                          "local_policy_changes":int(frame.policy_changes.sum()),
                          "mean_interventions":float(frame.interventions.mean())})
                st.caption("Training observations include exploration. Compare policies using held-out evaluation.")
                st.download_button("Download model + metadata",agent.to_bytes(),file_name=f"{name}.npz",key=f"export_{name}")
                st.download_button("Download learning history",frame.to_csv(index=False),file_name=f"{name}_training.csv",key=f"curve_{name}")

def comparison_page():
    st.title("Does the policy earn its intervention cost?")
    st.write("Frozen policies face the same held-out orders and stage-indexed random disturbances. Lower total economic cost is the primary objective.")
    a,b,c = st.columns(3)
    n = a.number_input("Held-out orders",100,20000,1000,100)
    seed = b.number_input("Held-out seed",0,2**31-20000,2026)
    scenario = c.selectbox("Evaluation scenario",list(SCENARIOS),index=1)
    diagnostic = st.checkbox("Balanced diagnostic outlook distribution")
    if st.button("Evaluate frozen policies",type="primary"):
        with st.spinner("Simulating held-out delivery decisions…"):
            rows,summary = evaluate(default_policies(st.session_state.agents),int(n),int(seed),EnvConfig(scenario),diagnostic)
        st.session_state.evaluation = (rows,summary,{"orders":n,"seed":seed,"scenario":scenario,"balanced":diagnostic})
    if "evaluation" not in st.session_state:
        st.info("Run an evaluation to compare the current session policies. Baselines work even before training."); return
    rows,summary,metadata = st.session_state.evaluation
    st.caption(f"Displayed run: {metadata}. Controls take effect when evaluation is run.")
    data_table(summary.round(3))
    for metric in ("average_cost","on_time_rate","average_late_days","intervention_cost","intervention_frequency","savings"):
        figure(charts.comparison(summary,metric))
    figure(charts.action_distribution(rows))
    st.caption("Cost bars show approximate 95% Monte Carlo confidence intervals. savings_ci95 uses paired per-order differences. These quantify simulation sampling error, not real-world uncertainty or variation across training seeds.")
    st.download_button("Export episode results",rows.to_csv(index=False),"evaluation.csv")

def simulator_page():
    st.title("Follow one order, decision by decision")
    st.write("The policy selects an action. The environment samples what happens. A single episode illustrates a decision sequence; average effectiveness belongs in Policy comparison.")
    agent = model_selector("sim_agent")
    order,config,seed = order_inputs("sim")
    mode = st.radio("Decision mode",["Automatic learned policy","Manual interventions"],horizontal=True)
    a,b,c = st.columns(3)
    if a.button("Generate order"):
        generated = generate_orders(1,seed)[0]
        st.session_state.generated_order = generated
    if "generated_order" in st.session_state:
        generated = st.session_state.generated_order
        st.caption(f"Generated candidate: {generated.record()}")
    use_generated = st.checkbox("Use generated candidate",value=False,disabled="generated_order" not in st.session_state)
    if b.button("Start simulation",type="primary"):
        selected = st.session_state.generated_order if use_generated and "generated_order" in st.session_state else order
        env = DeliveryMitigationEnv(config); env.reset(seed,selected)
        baseline = DeliveryMitigationEnv(config); baseline.reset(seed,selected)
        for _ in range(3): baseline.step(0)
        # Snapshot the selected agent, mode and configuration at start.
        st.session_state.simulation = {"env":env,"baseline":baseline,"agent":agent,"mode":mode,"seed":seed}
    if c.button("Reset simulation"):
        st.session_state.pop("simulation",None)
    if "simulation" not in st.session_state:
        return
    sim = st.session_state.simulation; env = sim["env"]
    st.caption(f"Active run: {env.config.scenario}, seed {sim['seed']}, {sim['mode']}. Restart to apply input changes.")
    metrics = st.columns(4)
    metrics[0].metric("Stage",STAGES[env.stage]); metrics[1].metric("Delivery outlook",f"{env.outlook:+d} days")
    metrics[2].metric("Budget remaining",f"R$ {env.budget}"); metrics[3].metric("Cost so far",f"R$ {env.spent+env.delay_cost:.0f}")
    if env.stage < 3:
        if sim["mode"] == "Manual interventions":
            action = st.selectbox("Valid intervention",env.get_valid_actions(),format_func=lambda a:f"{ACTION_NAMES[a]} · R$ {env.config.costs[a]}")
        elif sim["agent"] is not None:
            action = choose(sim["agent"],env)
            state = env.get_state(); visits = sim["agent"].visits[state].sum()
            st.write(f"**Learned choice: {ACTION_NAMES[action]}** · {int(visits)} training visits to this state")
            if visits == 0:
                st.warning("This state was unseen during training. Zero-value ties use the first valid action; this is an unlearned fallback.")
        else:
            st.warning("Train an agent or restart in manual mode."); return
        left,right = st.columns(2)
        advance = left.button("Next stage",type="primary")
        full = right.button("Run full episode")
        if advance or full:
            env.step(action)
            if full:
                while env.stage < 3:
                    # Manual full-run applies the chosen action if still valid,
                    # otherwise no intervention; no impossible repeat purchase.
                    next_action = (action if action in env.get_valid_actions() else 0) if sim["mode"]=="Manual interventions" else choose(sim["agent"],env)
                    env.step(next_action)
            st.rerun()
        if sim["mode"]=="Manual interventions":
            st.caption("Manual full run reuses your selected action only while valid; subsequent unavailable actions become no intervention.")
    figure(charts.trajectory(env))
    if env.history:
        data_table(pd.DataFrame(env.history).drop(columns=["valid_actions"]))
    if env.stage == 3:
        baseline = sim["baseline"]
        a,b = st.columns(2)
        a.metric("Chosen policy · final cost",f"R$ {env.spent+env.delay_cost:.0f}")
        b.metric("No intervention · same order & disturbances",f"R$ {baseline.spent+baseline.delay_cost:.0f}")
        st.write(f"Final late days: **{max(0,env.outlook)}** for chosen policy; **{max(0,baseline.outlook)}** for no intervention.")
        figure(charts.cost_breakdown({"Chosen policy":env,"No intervention":baseline}))

def policy_page():
    st.title("Look inside the learned Q-table")
    agent = model_selector("policy_agent")
    if agent is None: return
    a,b,c = st.columns(3)
    stage = a.selectbox("Stage",[0,1,2],format_func=lambda n:STAGES[n],key="policy_stage")
    distance = b.selectbox("Distance",[0,1,2],format_func=lambda n:DISTANCES[n],key="policy_distance")
    load = c.selectbox("Load",[0,1],format_func=lambda n:["Normal","High"][n])
    budget = a.slider("Exact remaining budget (R$)",0,50,40)
    services = b.selectbox("Active service bits",[0,1,2,3])
    risk_filter = c.multiselect("Risk bands",["Low","Medium","High"],default=["Low","Medium","High"])
    outlooks = st.slider("Signed outlook range (days)",-3,8,(-3,8))
    lateness_filter = st.multiselect("Lateness bands",["On time","1–2 days","3–5 days",">5 days"],default=["On time","1–2 days","3–5 days",">5 days"])
    valid = valid_actions(stage,budget,services,agent.env_config.costs)
    rows=[]; z=[]; labels=[]
    for outlook in range(outlooks[0],outlooks[1]+1):
        r = risk(outlook,distance,load); band = "Low" if r < .3 else ("Medium" if r < .6 else "High")
        lateness = "On time" if outlook <= 0 else ("1–2 days" if outlook<=2 else ("3–5 days" if outlook<=5 else ">5 days"))
        if band not in risk_filter or lateness not in lateness_filter: continue
        state = int(np.ravel_multi_index((stage,outlook+3,load,distance,budget,services),STATE_DIMS))
        values=[float(agent.q[state,i]) if i in valid else np.nan for i in range(4)]
        z.append(values); labels.append(f"{outlook:+d} days · {band}")
        rows.append({"state":state,"outlook":outlook,"risk_band":band,"visits":int(agent.visits[state].sum()),
                     "best_action":ACTION_NAMES[agent.select(state,valid)],**dict(zip(ACTION_NAMES,values))})
    if not rows:
        st.info("No states match these filters."); return
    fig=go.Figure(go.Heatmap(z=z,x=ACTION_NAMES,y=labels,colorscale="Teal",colorbar=dict(title="Return (R$)"),hoverongaps=False))
    fig.update_layout(title="Learned expected future returns · invalid actions are blank",xaxis_title="Intervention",yaxis_title="Delivery outlook / risk")
    figure(charts.style(fig)); data_table(pd.DataFrame(rows))
    st.caption("The policy maximises Q over valid actions. Q-values estimate expected future return, not probabilities. Zero may mean an unvisited action rather than a free outcome. Some filtered combinations are unreachable.")

def scenario_page():
    st.title("How dependent are decisions on assumptions?")
    st.write("Robustness protocol: hold the learned policy fixed and change the environment. This is different from retraining a separate policy in each scenario.")
    data_table(pd.DataFrame([asdict(s) for s in SCENARIOS.values()]))
    n = st.number_input("Orders per scenario",100,10000,1000,100)
    seed = st.number_input("Robustness seed",0,2**31-20000,2026)
    if st.button("Run fixed-policy robustness",type="primary"):
        with st.spinner("Evaluating all three scenarios…"):
            st.session_state.robustness = (robustness(st.session_state.agents,int(n),int(seed)),int(n),int(seed))
    if "robustness" in st.session_state:
        frame,run_n,run_seed = st.session_state.robustness
        st.caption(f"Displayed run: {run_n} matched orders per scenario; seed {run_seed}.")
        figure(charts.scenario_chart(frame)); data_table(frame.round(3))
        metric = st.selectbox("Additional sensitivity metric",["on_time_rate","average_late_days","intervention_cost","intervention_frequency"])
        figure(charts.scenario_chart(frame,metric))
        st.download_button("Export sensitivity results",frame.to_csv(index=False),"robustness.csv")
    st.caption("A policy can become less cost-effective when success rates fall. These experiments expose sensitivity to assumed effects; they do not validate those effects. Research experiments measures training-seed variability separately.")

def research_page():
    st.title("Stronger evidence, beyond a single training run")
    st.write("Predefined training configurations are compared on validation orders. The selected setting is then evaluated across independent training seeds on a separate, fresh final test set.")
    improved_folder=ROOT/"results"/"improved_training"
    if (improved_folder/"metadata.json").exists():
        show_improved=st.toggle("Show extended RL training study",value=True)
        if show_improved:
            improvement_page(improved_folder)
            return
    folder=ROOT/"results"/"studies"
    if not (folder/"metadata.json").exists():
        st.info("Run python scripts/run_studies.py to generate the repeated-seed research study.")
        return
    metadata=json.loads((folder/"metadata.json").read_text(encoding="utf-8"))
    st.caption(f"Training seeds: {metadata['training_seeds']} · {metadata['evaluation_orders']} orders per evaluation · validation {metadata['validation_seed']} · final test {metadata['final_test_seed']}")
    if metadata["status"]!="complete":
        st.info("The study is running. Completed validation runs are saved incrementally; refresh after completion to see final results.")
        if (folder/"validation_runs.csv").exists(): data_table(pd.read_csv(folder/"validation_runs.csv").round(3))
        return
    section=st.radio("Research view",["Training & coverage","Fresh final test","Economic sensitivity"],horizontal=True)
    if section=="Training & coverage":
        validation=pd.read_csv(folder/"validation_summary.csv")
        runs=pd.read_csv(folder/"validation_runs.csv")
        data_table(pd.DataFrame(metadata["settings"]))
        figure(charts.style(px.bar(validation,x="setting",y="average_cost",color="policy",barmode="group",
            error_y="cost_seed_ci95",title="Validation cost across independently trained policies",
            labels={"average_cost":"Mean economic cost (R$/order)","setting":"Training configuration","policy":"Algorithm"})))
        figure(charts.style(px.bar(validation,x="setting",y="support_action_coverage",color="policy",barmode="group",
            title="Coverage of feasible state-action pairs in the training distribution",
            labels={"support_action_coverage":"Fraction visited at least once","setting":"Training configuration","policy":"Algorithm"})))
        columns=["policy","setting","average_cost","cost_seed_ci95","support_state_coverage",
                 "support_action_coverage","support_pairs_under_10_visits","unseen_state_fraction","unseen_action_fraction","incomplete_state_fraction"]
        data_table(validation[columns].round(4))
        st.write("Coverage uses the support of the default generator, not every allocated table row. A visited state can still have poorly sampled alternatives. ‘Under 10 visits’ describes feasible state-action pairs, including zero-visit pairs; it is a diagnostic threshold, not a convergence guarantee.")
        st.caption("The retained exact state avoids hiding budget or purchased services. A smaller array with the same lossless mapping would save memory but would not provide additional learning samples.")
        curve_algorithm=st.selectbox("Learning-curve algorithm",list(metadata["selected_settings"]))
        curve_setting=st.selectbox("Learning-curve configuration",[setting["label"] for setting in metadata["settings"]])
        curves=[]
        for seed in metadata["training_seeds"]:
            curve_path=folder/f"curve_{curve_algorithm}_{curve_setting}_seed{seed}.csv"
            if curve_path.exists(): curves.append(pd.read_csv(curve_path).assign(training_seed=str(seed)))
        if curves:
            figure(charts.style(px.line(pd.concat(curves,ignore_index=True),x="episode",y="block_mean_return",color="training_seed",
                title="Actual training experience: nonoverlapping 250-episode means",
                labels={"episode":"Training episode","block_mean_return":"Block mean return (R$)","training_seed":"Training seed"})))
            st.caption("These are recorded training returns with exploration, shown separately for each seed. Final policy quality is measured on held-out orders.")
        st.download_button("Download per-seed validation and coverage",runs.to_csv(index=False),"validation_runs.csv")
    elif section=="Fresh final test":
        final=pd.read_csv(folder/"final_summary.csv")
        figure(charts.style(px.bar(final,x="policy",y="average_cost",color="policy",error_y="cost_seed_ci95",
            title="Fresh held-out test: training-seed variability",
            labels={"average_cost":"Mean economic cost (R$/order)","policy":"Policy"})))
        data_table(final.round(4))
        st.write(f"Selected settings, using validation means only: {metadata['selected_settings']}.")
        st.caption("Error bars are approximate Student-t 95% intervals over independent training-seed means, conditional on the shared test orders. Baselines do not train and are identical across these replicates: zero seed variability does not mean zero order-sampling uncertainty. Per-run Monte Carlo intervals are exported separately. The final set is never used for setting selection.")
        algorithm=st.selectbox("Research algorithm",list(metadata["selected_settings"]))
        seed=st.selectbox("Research training seed",metadata["training_seeds"])
        setting=metadata["selected_settings"][algorithm]
        if st.button("Load research model for live demonstration",type="primary"):
            name=f"{algorithm} · {setting} · seed {seed}"
            st.session_state.agents[name]=load_agent(ROOT/"models"/"studies"/f"{algorithm}_{setting}_seed{seed}.npz")
            for key in ("evaluation","robustness","simulation"): st.session_state.pop(key,None)
            st.success("Research model loaded. Select it in Live order simulator or Q-table & policy.")
        st.caption("Choose a seed for demonstration, not a winner selected from final test outcomes. All seeds contribute equally to the reported results.")
        st.download_button("Download final per-seed metrics",(folder/"final_runs.csv").read_bytes(),"final_runs.csv")
    else:
        frame=pd.read_csv(folder/"economics_summary.csv")
        st.latex(r"\text{Saving}=p_{day}(\mathbb{E}[D_0]-\mathbb{E}[D_\pi])+F(\Pr[D_0>0]-\Pr[D_\pi>0])-\mathbb{E}[C_{intervention}]")
        st.caption("Intervention pays economically when expected avoided late-day and fixed penalties exceed intervention expense. Positive savings favour the policy over no intervention.")
        factor=st.selectbox("Economic factor",list(frame.factor.unique()))
        metric=st.selectbox("Economic metric",["savings","average_cost","on_time_rate","intervention_frequency","unseen_action_fraction"])
        labels={"Budget":"Starting budget (R$)","Intervention price multiplier":"Price multiplier (×)","Late-day penalty":"Penalty per late day (R$)"}
        units={"savings":"Saving versus no intervention (R$/order)","average_cost":"Total cost (R$/order)",
               "on_time_rate":"On-time fraction","intervention_frequency":"Fraction receiving intervention","unseen_action_fraction":"Fraction of decisions using unvisited actions"}
        part=frame[frame.factor==factor]
        fig=px.line(part,x="value",y=metric,color="policy",markers=True,
            title=f"Fixed-policy stress test: {factor.lower()}",labels={"value":labels[factor],metric:units[metric]})
        if metric=="savings": fig.add_hline(y=0,line_dash="dot",line_color="#97a6ba")
        figure(charts.style(fig)); data_table(part.round(4))
        st.warning("These are frozen-policy stress tests, not policies retrained for new prices or penalties. Price changes can introduce unvisited exact-budget states; inspect coverage before interpreting losses as economic effects alone.")
        st.caption("Budget values are represented in the original mixed-budget training distribution. For reward changes, the frozen policy keeps its actions while the accounting objective changes. New optimal decisions require retraining.")
        st.download_button("Download economic sensitivity",frame.to_csv(index=False),"economics_summary.csv")

def improvement_page(folder):
    meta=json.loads((folder/"metadata.json").read_text(encoding="utf-8"))
    st.subheader("Longer training and more stable action-value estimates")
    st.write("The simulator, intervention costs, reward, order generator and baseline rules remain unchanged. Settings are selected on validation orders. A separate final set tests the selected policies.")
    st.caption(f"Run status: {meta['status']} · validation {meta['validation_seed']} · fresh final test {meta['final_seed']}")
    data_table(pd.DataFrame([dict(setting=name,**values) for name,values in meta["settings"].items()]))
    st.latex(r"\alpha_n(s,a)=\alpha_0 / n(s,a)^p")
    st.caption("The visit schedule changes only the step size of the usual Q-learning or SARSA update. p=0 gives a constant learning rate. Original models remain compatible.")
    if (folder/"validation_runs.csv").exists():
        with st.expander("All pilot validation results"):
            validation=pd.read_csv(folder/"validation_runs.csv")
            data_table(validation.round(4))
    if meta["status"]!="complete":
        st.info("Training and evaluation are in progress. Refresh after completion to inspect the recorded final results.")
        return
    final=pd.read_csv(folder/"final_summary.csv")
    figure(charts.style(px.bar(final,x="policy",y="average_cost",color="policy",error_y="cost_seed_ci95",
        title="Fresh 5,000-order test of validation-selected policies",
        labels={"average_cost":"Mean economic cost (R$/order)","policy":"Policy"})))
    data_table(final.round(4))
    st.write(f"Selected configurations: {meta['selected_settings']}.")
    st.caption("Five independent training seeds share the same 5,000 held-out orders. Error bars quantify training-seed variability conditional on those orders. Per-run paired order intervals are exported separately. Baseline training variability is zero, but order-sampling uncertainty remains.")
    algorithm=st.selectbox("Extended-study algorithm",list(meta["selected_settings"]))
    seed=st.selectbox("Extended-study training seed",meta["confirmation_seeds"])
    setting=meta["selected_settings"][algorithm]
    if st.button("Load extended-study model",type="primary"):
        name=f"{algorithm} · extended training · seed {seed}"
        st.session_state.agents[name]=load_agent(ROOT/"models"/"improved_training"/f"{algorithm}_{setting}_seed{seed}.npz")
        for key in ("evaluation","robustness","simulation"): st.session_state.pop(key,None)
        st.success("Model loaded for the live simulator and Q-table explorer.")
    st.download_button("Export extended-study metrics",(folder/"final_runs.csv").read_bytes(),"improved_training_final_runs.csv")
    if (folder/"ensemble_final_summary.csv").exists():
        with st.expander("Additional experiment: five-run mean-Q ensembles"):
            data_table(pd.read_csv(folder/"ensemble_final_summary.csv").round(4))
            st.caption("These average the learned Q-values of all five seeds. They are additional policy variants, not individual Q-learning or SARSA runs. All individual-agent results remain above. No member was selected using final test performance.")

def methodology_page():
    st.title("The method, and its limits")
    st.markdown("""**MDP = (S, A, P, R, γ).** The state describes stage, signed outlook, load, distance, exact budget and purchased services. The action is a feasible intervention. The transition combines service effects with operational disturbances. The reward is negative economic cost. A policy maps states to decisions; an episode is one order’s three decisions.

**Why sequential reinforcement learning?** Buying priority or transport changes the next delivery outlook, remaining budget and available actions. A contextual bandit would omit these later decision dependencies.

**Q-learning is off-policy:** it bootstraps from the best valid next action. **SARSA is on-policy:** it uses the next action actually selected by the current epsilon-greedy behaviour policy. Exploration tries valid alternatives; exploitation selects the highest learned return.
""")
    st.latex(r"Q(s,a)\leftarrow Q(s,a)+\alpha[r+\gamma\max_{a'\in A(s')}Q(s',a')-Q(s,a)]")
    st.latex(r"Q(s,a)\leftarrow Q(s,a)+\alpha[r+\gamma Q(s',a')-Q(s,a)]\quad\text{(SARSA)}")
    st.write("Terminal updates use reward alone. Immediate reward is −action cost; the final transition also charges 15 × late days + 20 if late. With γ=1, episode return is exactly negative total economic cost. Smaller γ changes the optimisation objective.")
    st.markdown("""**Hard budgets:** unaffordable, stage-inappropriate and duplicate services are excluded from exploration, exploitation and Bellman targets.

**Reproducibility:** orders, exploration and stage-indexed outcomes use explicit independent random generators. Held-out evaluation uses matched initial orders and disturbances for every policy. Intervention success draws are indexed by stage/component, so an extra purchase cannot shift later disturbances.

**Limits:** synthetic, uncalibrated effects; integer-day trajectory clipped to −3…8; fixed load and distance; no shared fleet or warehouse capacity; a large fraction of allocated states is unreachable or unvisited. The risk display is a logistic pressure proxy rather than a calibrated probability. Monte Carlo intervals exclude simulator misspecification and training-seed uncertainty. Finite training offers no guarantee of optimality or superiority over baselines.

**Future work:** calibrate from real logistics data, include carrier pricing and shared warehouse capacity, model delivery networks, and use contextual bandits for genuinely single-stage decisions. Larger RL models become relevant only if the decision space requires them.
""")

def main():
    st.set_page_config(page_title="ARLO · Delivery Decision Lab",page_icon="↗",layout="wide")
    st.markdown("""<style>
    .stApp {background:#f5f8fc;} h1,h2,h3{color:#142d4e;letter-spacing:-.025em;}
    h1{font-size:2.65rem!important;max-width:1000px;} .block-container{padding-top:2.8rem;max-width:1400px;}
    [data-testid="stMetric"]{background:white;border:1px solid #e1e8f0;border-radius:14px;padding:20px;}
    [data-testid="stSidebar"]{background:#fff;border-right:1px solid #e1e8f0;}
    .eyebrow{font-size:.72rem;letter-spacing:.17em;font-weight:700;color:#087f8c;margin-bottom:15px;}
    .flow{background:#e8f3f5;border:1px solid #cfe6e8;border-radius:14px;padding:24px;margin:28px 0;font-size:.8rem;font-weight:700;letter-spacing:.05em;}
    .flow span{color:#087f8c;padding:0 15px;}
    .data-table{overflow-x:auto;background:white;border:1px solid #e1e8f0;border-radius:12px;margin:15px 0;}
    .data-table table{border-collapse:collapse;width:100%;font-size:.85rem;white-space:nowrap;}
    .data-table th{background:#e8f3f5;text-align:left;padding:12px;color:#142d4e;}
    .data-table td{padding:10px 12px;border-top:1px solid #e1e8f0;}
    </style>""",unsafe_allow_html=True)
    bootstrap()
    with st.sidebar:
        st.markdown("## ARLO ↗")
        st.caption("DELIVERY DECISION LAB")
        page = st.radio("Explore the project",PAGES,label_visibility="collapsed")
        st.divider(); st.caption("ACADEMIC SIMULATION")
        st.write("Learn the trade-off.\nInspect the decision.")
        st.caption(f"{len(st.session_state.agents)} trained policies available · R$ illustrative costs")
    handlers = [overview,environment_page,training_page,comparison_page,simulator_page,policy_page,scenario_page,research_page,methodology_page]
    handlers[PAGES.index(page)]()
