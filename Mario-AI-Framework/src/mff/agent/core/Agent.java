package mff.agent.core;

import engine.helper.MarioActions;
import mff.agent.helper.IMarioAgentSlim;
import mff.agent.helper.MarioTimerSlim;
import mff.forwardmodel.slim.core.MarioForwardModelSlim;

public class Agent implements IMarioAgentSlim {
    private boolean[] action;
    private AStarTree tree;

    @Override
    public void initialize(MarioForwardModelSlim model) {
        this.action = new boolean[MarioActions.numberOfActions()];
        this.tree = new AStarTree();
    }

    @Override
    public boolean[] getActions(MarioForwardModelSlim model, MarioTimerSlim timer) {
        action = this.tree.optimise(model, timer);
        return action;
    }

    @Override
    public String getAgentName() {
        return "Robin Baumgarten agent with slim forward model and performance improvements";
    }
}
