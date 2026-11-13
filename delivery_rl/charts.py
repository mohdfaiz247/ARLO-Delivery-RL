"""Plotly figures shared by the dashboard and reproducible HTML exports."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from .environment import ACTION_NAMES

COLORS = ["#087f8c","#3765ad","#97a6ba","#bd8433"]

def style(fig):
    fig.update_layout(template="plotly_white",colorway=COLORS,font=dict(family="Arial",color="#142d4e"),
        margin=dict(l=20,r=20,t=55,b=30),paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h",y=-.22),hovermode="closest")
    return fig

def learning(curves, column="moving_return"):
    data = pd.concat([frame.assign(algorithm=name) for name,frame in curves.items()],ignore_index=True)
    labels = {"moving_return":"250-episode mean return (R$)","return":"Episode return (R$)","epsilon":"Exploration probability"}
    return style(px.line(data,x="episode",y=column,color="algorithm",title="Exploration schedule" if column=="epsilon" else "Learning from simulated experience",
                         labels={"episode":"Training episode",column:labels[column]}))

def comparison(summary,metric="average_cost"):
    labels = {"average_cost":"Mean economic cost (R$)","on_time_rate":"On-time delivery fraction",
              "average_late_days":"Mean late days","intervention_cost":"Mean intervention expense (R$)",
              "intervention_frequency":"Fraction of orders receiving intervention","savings":"Cost saved per order (R$)"}
    return style(px.bar(summary,x="policy",y=metric,color="policy",color_discrete_sequence=COLORS,
        error_y="cost_ci95" if metric=="average_cost" else None,title=labels[metric],labels={metric:labels[metric],"policy":"Policy"}))

def action_distribution(episodes):
    counts = episodes.groupby("policy")[[f"action_{i}" for i in range(4)]].sum()
    counts = counts.div(counts.sum(axis=1),axis=0).reset_index().melt(id_vars="policy",var_name="action",value_name="fraction")
    counts["action"] = counts.action.map({f"action_{i}":name for i,name in enumerate(ACTION_NAMES)})
    return style(px.bar(counts,x="policy",y="fraction",color="action",title="Decisions by intervention",barmode="stack",
                        labels={"fraction":"Fraction of decisions","policy":"Policy","action":"Intervention"}))

def scenario_chart(frame,metric="average_cost"):
    labels={"average_cost":"Mean economic cost (R$)","on_time_rate":"On-time fraction",
            "average_late_days":"Mean late days","intervention_cost":"Intervention expense (R$)",
            "intervention_frequency":"Fraction of orders receiving intervention"}
    return style(px.line(frame,x="scenario",y=metric,color="policy",markers=True,
        error_y="cost_ci95" if metric=="average_cost" else None,title="Fixed-policy robustness across assumed effectiveness",
        labels={metric:labels[metric],"scenario":"Evaluation environment","policy":"Policy"}))

def trajectory(env):
    initial = {"stage":0,"outlook":env.order.outlook,"risk":None}
    from .environment import risk
    initial["risk"] = risk(env.order.outlook,env.order.distance,env.order.load)
    rows = [initial]+env.history
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[r["stage"] for r in rows],y=[r["outlook"] for r in rows],name="Signed outlook (days)",mode="lines+markers"))
    fig.add_trace(go.Scatter(x=[r["stage"] for r in rows],y=[r["risk"] for r in rows],name="Delay pressure (0–1)",yaxis="y2",mode="lines+markers"))
    fig.update_layout(title="Order trajectory after each simulated decision",xaxis=dict(title="Operational stage",tickvals=[0,1,2,3],ticktext=["Approved","Preparing","Dispatched","Delivered"]),
        yaxis=dict(title="Signed delivery outlook (days)"),yaxis2=dict(title="Delay pressure",overlaying="y",side="right",range=[0,1]))
    return style(fig)

def cost_breakdown(envs):
    rows = [{"policy":name,"component":component,"cost":value} for name,env in envs.items()
            for component,value in (("Interventions",env.spent),("Delay penalty",env.delay_cost))]
    return style(px.bar(pd.DataFrame(rows),x="policy",y="cost",color="component",barmode="stack",
                       title="Actual cost of this simulated order",labels={"cost":"Economic cost (R$)","policy":"Policy"}))
