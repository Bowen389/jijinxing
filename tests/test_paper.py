import json
import numpy as np
import pandas as pd
from paper import sim
from engine.em_features import holders_table

def test_save_restart_deduplicates(tmp_path, monkeypatch):
    monkeypatch.setattr(sim, 'ACC_DIR', str(tmp_path/'accounts'))
    monkeypatch.setattr(sim, 'SIG_DIR', str(tmp_path/'signals'))
    a = sim.Account('S_top3')
    d = pd.Timestamp('2026-10-12')
    a.init(d)
    a.s['last_date'] = str(d.date())
    a.trade(d, '买入', 'SH600001', 100, 10, 5)
    a.navs = [dict(日期=str(d.date()), 总资产=a.total())]
    a.save({})
    b = sim.Account('S_top3')
    b.trades = [dict(a=0, 日期=str(d.date()), 操作='买入', 代码='600001.SH')]
    b.navs = [dict(日期=str(d.date()), 总资产=b.total())]
    b.save({})
    assert len(pd.read_csv(tmp_path/'accounts/S_top3/trades.csv')) == 1
    assert len(pd.read_csv(tmp_path/'accounts/S_top3/nav.csv')) == 1
    assert b.s['positions'] == a.s['positions']
    assert not list(tmp_path.rglob('*.tmp'))

def test_holders_does_not_use_late_disclosed_previous_period(tmp_path, monkeypatch):
    import engine.em_features as ef
    monkeypatch.setattr(ef, 'EXTRA', str(tmp_path))
    reports = pd.DataFrame([
        dict(code='600001', end='2026-03-31', notice='2026-08-01', holders=100, avg_hold=1000, shares=100000),
        dict(code='600001', end='2026-06-30', notice='2026-07-01', holders=50, avg_hold=2000, shares=100000),
    ])
    reports.to_parquet(tmp_path/'em_holders.parquet')
    h = holders_table()
    first, last = h.iloc[0], h.iloc[1]
    assert pd.isna(first.HN_CHG1)
    assert first.avail == pd.Timestamp('2026-07-02')
    assert last.end == pd.Timestamp('2026-06-30')
    assert np.isclose(last.HN_CHG1, np.log(.5))

def test_continuous_run_restart_and_migration(tmp_path, monkeypatch):
    d0,d1,d2 = pd.to_datetime(['2026-10-09','2026-10-12','2026-10-13'])
    monkeypatch.setattr(sim, 'ACC_DIR', str(tmp_path/'accounts'))
    monkeypatch.setattr(sim, 'SIG_DIR', str(tmp_path/'signals'))
    monkeypatch.setattr(sim, 'names_table', lambda: {})
    monkeypatch.setattr(sim, 'trading_days', lambda: [d0,d1,d2])
    class MK:
        bench = pd.Series([0.,.01,0.], index=[d0,d1,d2])
        def day(self,d):
            return pd.DataFrame([dict(in_pool=True, status_known=True, buy_ok=True)])
        def quote(self,d,j):
            return dict(raw_open=10., raw_close=11., factor=1., susp=False, dn_open=False,up_open=False,buy_ok=True)
    monkeypatch.setattr(sim, 'Market', MK)
    from senti.live import CFG
    monkeypatch.setattr(sim, 'strat_cfg', lambda n: CFG)
    def signal(a,d):
        return dict(signal_date=str(d.date()), sell=[], buy=[dict(inst='SH600001')] if not a.s['positions'] else [], backup=[]), pd.DataFrame(), 'ok'
    monkeypatch.setattr(sim, 'gen_signal', signal)
    a=sim.Account('S_top3');a.init(d0);a.s.pop('execution_version');a.s.pop('execution_start')
    a.s['last_date']=str(d0.date());a.s['pending']=signal(a,d0)[0];a.save({})
    sim.run_one('S_top3', until=str(d1.date()))
    b=sim.Account('S_top3')
    assert b.s['execution_start']==str(d1.date())
    assert b.s['positions']['SH600001']['shares']==3300
    assert b.s['positions']['SH600001']['last_px']==11
    sim.run_one('S_top3')
    state=sim.Account('S_top3').s
    sim.run_one('S_top3')
    assert sim.Account('S_top3').s==state
    assert len(pd.read_csv(tmp_path/'accounts/S_top3/trades.csv'))==1
    assert len(pd.read_csv(tmp_path/'accounts/S_top3/nav.csv'))==2

def test_signal_filters_and_actual_pending(tmp_path, monkeypatch):
    from senti import live
    import strategies.live  # load legacy module side effects before sandboxing cwd
    import argparse
    from engine.common import KEY
    monkeypatch.setattr(live, 'EXTRA', str(tmp_path))
    monkeypatch.chdir(tmp_path)
    pd.DataFrame([dict(instrument=j, code_name='测试', industry='行业') for j in ('SH600001','SH600002','SH600003','SH600004','SH688001')]).to_parquet(tmp_path/'industry.parquet')
    d=pd.Timestamp('2026-10-12')
    inst=['SH600001','SH600002','SH600003','SH600004','SH688001']
    scores=pd.DataFrame(dict(datetime=[d]*5,instrument=inst,score=[5.,4.,3.,2.,1.]))
    detail=scores.assign(up=.5,dn=.1)
    monkeypatch.setattr(live,'scores',lambda a,b:(scores,detail,scores))
    px=pd.DataFrame(dict(datetime=[d]*5,instrument=inst,raw_close=[10.]*5,up_lim=[False]*5,dn_lim=[False]*5,
                         susp=[False]*5,in_pool=[True]*5,buy_ok=[False,False,True,True,True],
                         status_known=[True,False,True,True,True],is_st=[True,False,False,False,False],is_delisted=[False]*5))
    monkeypatch.setattr(live,'read_parts',lambda *a,**kw:px.copy())
    h=tmp_path/'hold.csv';pd.DataFrame(dict(code=['600001.SH'],shares=[100])).to_csv(h,index=False)
    fn=live.cmd_signal(argparse.Namespace(date=str(d.date()),cash=5000.,holdings=str(h)))
    orders=pd.read_csv(fn)
    assert orders.loc[orders['操作']=='卖出','代码'].tolist()==['600001.SH']
    assert set(orders.loc[orders['操作']=='买入','代码'])=={'600003.SH','600004.SH','688001.SH'}
    assert '600002.SH' not in set(orders['代码'])
    # Read the exact CSV through the paper adapter, preserving all planned candidates.
    monkeypatch.setattr(sim,'PAPER',str(tmp_path/'paper'))
    monkeypatch.setattr(sim,'SIG_DIR',str(tmp_path/'paper/signals'))
    monkeypatch.setattr(sim,'ACC_DIR',str(tmp_path/'paper/accounts'))
    acc=sim.Account('S_top3');acc.init(d);acc.s['cash']=5000.
    acc.s['positions']={'SH600001':dict(shares=100.,cost=1005.,last_px=10.,buy_date='2026-10-09',factor=1.)}
    p,out,text=sim.gen_signal(acc,d)
    assert p['sell']==['SH600001']
    assert {x['inst'] for x in p['buy']}=={'SH600003','SH600004','SH688001'}
    assert '未知' in text
