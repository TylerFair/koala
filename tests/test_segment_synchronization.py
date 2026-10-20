"""Host synchronization policy must not alter segment order or outputs."""
import numpy as np
import pytest
import models.independent_nuts as engine

@pytest.mark.parametrize('collect', [True, False])
@pytest.mark.parametrize('show,synchronize,expected_waits', [(False,False,0),(False,True,3),(True,False,1),(True,True,3)])
def test_silent_segments_can_dispatch_without_host_waits(monkeypatch,collect,show,synchronize,expected_waits):
    waits=[];progress=[]
    monkeypatch.setattr(engine.jax,'block_until_ready',lambda value:waits.append(value))
    class Bar:
        def __init__(self,*args):pass
        def update(self,n):progress.append(n)
        def close(self):progress.append('closed')
    monkeypatch.setattr(engine,'_SegmentProgress',Bar)
    def program(state,increment):
        new=state+increment
        return (new,np.array([new])) if collect else new
    result=engine.IndependentNUTSRunner._run_segments(program,0,2,6,'test',show,collect,(2,),synchronize=synchronize)
    if collect:
        state,draws=result
        np.testing.assert_array_equal(draws,[2,4,6])
    else:state=result
    assert state==6 and len(waits)==expected_waits
    assert progress==([6,'closed'] if show and not synchronize else [2,2,2,'closed'])
