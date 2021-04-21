package mff.agent.helper;

import mff.forwardmodel.slim.core.MarioForwardModelSlim;

public interface IMarioAgentSlim {

    void initialize(MarioForwardModelSlim model);

    boolean[] getActions(MarioForwardModelSlim model, MarioTimerSlim timer);

    String getAgentName();
    
}
