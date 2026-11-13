from pathlib import Path
from streamlit.testing.v1 import AppTest
from delivery_rl.dashboard import PAGES

ROOT=Path(__file__).resolve().parents[1]

def app():
    return AppTest.from_file(str(ROOT/"app.py"),default_timeout=120).run()

def test_all_pages_load():
    at=app()
    for page in PAGES:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, f"{page}: {at.exception}"

def test_manual_simulation_next_and_full_episode():
    at=app(); at.sidebar.radio[0].set_value("Live order simulator").run()
    next(r for r in at.radio if r.label=="Decision mode").set_value("Manual interventions").run()
    next(b for b in at.button if b.label=="Start simulation").click().run()
    assert at.session_state["simulation"]["env"].stage==0
    next(b for b in at.button if b.label=="Next stage").click().run()
    assert at.session_state["simulation"]["env"].stage==1
    next(b for b in at.button if b.label=="Run full episode").click().run()
    assert at.session_state["simulation"]["env"].stage==3
    assert not at.exception

def test_training_evaluation_and_export():
    at=app(); at.sidebar.radio[0].set_value("Train the agents").run()
    next(n for n in at.number_input if n.label=="Training episodes").set_value(100)
    next(b for b in at.button if b.label=="Train policy").click().run()
    assert not at.exception
    assert at.session_state["agents"]["q_learning"].visits.sum()==300
    at.sidebar.radio[0].set_value("Policy comparison").run()
    next(n for n in at.number_input if n.label=="Held-out orders").set_value(100)
    next(b for b in at.button if b.label=="Evaluate frozen policies").click().run()
    assert not at.exception and len(at.session_state["evaluation"][0])>=300

def test_automatic_episode_and_risk_input():
    at=app(); at.sidebar.radio[0].set_value("Train the agents").run()
    next(n for n in at.number_input if n.label=="Training episodes").set_value(100)
    next(b for b in at.button if b.label=="Train policy").click().run()
    at.sidebar.radio[0].set_value("Live order simulator").run()
    next(c for c in at.checkbox if c.label=="Configure initial risk instead of delivery outlook").check().run()
    next(b for b in at.button if b.label=="Start simulation").click().run()
    next(b for b in at.button if b.label=="Run full episode").click().run()
    env=at.session_state["simulation"]["env"]
    assert env.stage==3 and env.budget>=0 and len(env.history)==3
    assert not at.exception
    next(b for b in at.button if b.label=="Reset simulation").click().run()
    assert "simulation" not in at.session_state

def test_research_views_and_load_model(tmp_path,monkeypatch):
    import delivery_rl.dashboard as dashboard
    from delivery_rl.studies import run_studies,StudySetting
    run_studies(tmp_path,seeds=(1,2),settings=(StudySetting("tiny",10,.1),),evaluation_orders=4,progress=lambda message:None)
    monkeypatch.setattr(dashboard,"ROOT",tmp_path)
    at=app(); at.sidebar.radio[0].set_value("Research experiments").run()
    assert not at.exception
    for view in ["Training & coverage","Economic sensitivity","Fresh final test"]:
        next(r for r in at.radio if r.label=="Research view").set_value(view).run()
        assert not at.exception
    next(b for b in at.button if b.label=="Load research model for live demonstration").click().run()
    assert len(at.session_state["agents"])==1 and not at.exception

def test_extended_study_page_and_load(tmp_path,monkeypatch):
    import json
    import delivery_rl.dashboard as dashboard
    from delivery_rl.agents import QLearningAgent,TrainingConfig
    from delivery_rl.training import train
    from delivery_rl.evaluation import evaluate,default_policies
    folder=tmp_path/"results"/"improved_training"; folder.mkdir(parents=True)
    model_folder=tmp_path/"models"/"improved_training"; model_folder.mkdir(parents=True)
    agent=QLearningAgent(TrainingConfig(episodes=50,seed=1,learning_rate=1.,visit_decay=.6)); train(agent)
    agent.save(model_folder/"q_learning_tiny_seed1.npz")
    _,summary=evaluate(default_policies({"q_learning":agent}),10,99)
    summary["cost_seed_ci95"]=0.
    summary.to_csv(folder/"final_summary.csv",index=False)
    summary.to_csv(folder/"final_runs.csv",index=False)
    (folder/"metadata.json").write_text(json.dumps({"status":"complete","validation_seed":98,
        "final_seed":99,"settings":{"tiny":{"episodes":50,"learning_rate":1.,"visit_decay":.6}},
        "selected_settings":{"q_learning":"tiny"},"confirmation_seeds":[1]}))
    monkeypatch.setattr(dashboard,"ROOT",tmp_path)
    at=app(); at.sidebar.radio[0].set_value("Research experiments").run()
    assert not at.exception
    next(b for b in at.button if b.label=="Load extended-study model").click().run()
    assert len(at.session_state["agents"])==1 and not at.exception
