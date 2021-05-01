package mff.agent.astar;

import engine.helper.MarioActions;
import mff.agent.helper.IMarioAgentSlim;
import mff.agent.helper.MarioTimerSlim;
import mff.forwardmodel.slim.core.MarioForwardModelSlim;

import java.util.ArrayList;

public class Agent implements IMarioAgentSlim {

    private boolean[] action;
    private ArrayList<boolean[]> actionsList = new ArrayList<>();
    //private boolean first = true;
    private boolean finished = false;

    @Override
    public void initialize(MarioForwardModelSlim model) {
        this.action = new boolean[MarioActions.numberOfActions()];
        AStarTree.winFound = false;
        AStarTree.exitTileX = model.getWorld().level.exitTileX * 16;
    }

    @Override
    public boolean[] getActions(MarioForwardModelSlim model, MarioTimerSlim timer) {
    	/*AStarTree tree = new AStarTree(model);
        action = tree.search(timer);
        return action;*/

        if (finished)
            return MarioAction.NO_ACTION.value;

//        if (finished) { // TODO: finished state
//            if (actionsList.size() > 0) {
//                System.out.println(actionsList.size());
//                return actionsList.remove((actionsList.size() - 1));
//            }
//            else {
//                return MarioAction.NO_ACTION.value;
//            }
//        }

        AStarTree tree = new AStarTree(model, 1);
        ArrayList<boolean[]> newActionsList = null;
        if (!AStarTree.winFound)
             newActionsList = tree.search(timer);

//        if (AStarTree.winFound) { // TODO: set win path and dont change it
//            actionsList = newActionsList;
//            finished = true;
//            return actionsList.remove((actionsList.size() - 1));
//        }

        if (newActionsList != null && newActionsList.size() > actionsList.size()) {
            if (!AStarTree.winFound)
                actionsList = newActionsList;
        }

        if (actionsList.size() == 0) { //TODO means finished?
            System.out.println("FINISHED");
            finished = true;
            return MarioAction.NO_ACTION.value;
        }

        //System.out.println(actionsList.size());
        return actionsList.remove(actionsList.size() - 1);

        /*if (first) {
            AStarTree tree = new AStarTree(model);
            actionsList = tree.search(new MarioTimerSlim(10000));
            first = false;
        }

        return actionsList.remove(0);*/
    }

    @Override
    public String getAgentName() {
        return "MFF Pure AStar Agent";
    }
}
