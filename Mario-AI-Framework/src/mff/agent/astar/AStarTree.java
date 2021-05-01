package mff.agent.astar;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.PriorityQueue;

import mff.agent.helper.MarioTimerSlim;
import mff.forwardmodel.slim.core.MarioForwardModelSlim;

public class AStarTree {
    public SearchNode bestNode;
    public float bestNodeCost;

    float marioXStart;
    float marioYStart;
    float levelCurrentTime;
    int searchSteps;

    static boolean winFound = false;
    static final float maxMarioSpeedX = 10.9f;
    static float exitTileX;

    PriorityQueue<SearchNode> opened = new PriorityQueue<>(new CompareByCost());
    /**
     * INT STATE -> STATE COST
     */
    HashMap<Integer, Float> visitedStates = new HashMap<>();
    
    public AStarTree(MarioForwardModelSlim startState, int searchSteps) {
    	levelCurrentTime = startState.getWorld().currentTimer;
    	this.searchSteps = searchSteps;

    	marioXStart = startState.getMarioX();
    	marioYStart = startState.getMarioY();
    	
    	bestNode = getStartNode(startState);
    	bestNodeCost = calculateCost(startState, bestNode.nodeDepth);
    	
    	opened.add(bestNode);    		
    }
    
    private int getIntState(MarioForwardModelSlim model) {
    	return getIntState((int) model.getMarioX(), (int) model.getMarioY(), model.getWorld().currentTick);
    }
    
    private int getIntState(int x, int y, int t) {
        //return (x << 18) | (y << 8) | t;
    	return (x << 16) | y;
    }
    
    private SearchNode getStartNode(MarioForwardModelSlim state) {
    	// TODO: pooling
    	return new SearchNode(state);
    }
    
    private SearchNode getNewNode(MarioForwardModelSlim state, SearchNode parent, float cost, MarioAction action) {
    	// TODO: pooling
    	return new SearchNode(state, parent, cost, action);
    }
    
    private float calculateCost(MarioForwardModelSlim nextState, int nodeDepth) {
        // current.nodeDepth + remaining_distance / max_speed_dx_per_frame
        // check for Mario alive after advance

//    	int marioState = nextState.getMarioMode() * 100 + (nextState.getWorld().mario.alive ? 0 : Integer.MIN_VALUE);
//    	int winBonus = nextState.getGameStatusCode() == 1 ? 1000 : 0;
//		return (nextState.getMarioX() - marioXStart) * 1.5f + marioState /*+ nextState.getWorld().currentTimer / 1000.0f*/
//                + (marioYStart - nextState.getMarioY()) + winBonus;

        float timeToFinish = (exitTileX - nextState.getMarioX()) / maxMarioSpeedX;
        //timeToFinish *= 10;
        //System.out.println("DEPTH: " + nodeDepth + " | timeToFinish: " + timeToFinish);
        return nodeDepth + timeToFinish;
	}
    
    public ArrayList<boolean[]> search(MarioTimerSlim timer) {
    	int iterations = 0;

    	if (winFound)
    	    return null; // TODO better solution

        while (opened.size() > 0 && timer.getRemainingTime() > 0) {
        	iterations++;
            SearchNode current = opened.remove();

            MarioForwardModelSlim nextState = current.state.clone();

            for (int i = 0; i < searchSteps; i++) {
                nextState.advance(current.marioAction.value);
            }

            if (!nextState.getWorld().mario.alive) {
                continue;
            }

            float nextCost = calculateCost(nextState, current.nodeDepth);
            int nextStateInt = getIntState(nextState);

            //System.out.println("BEST: " + bestNodeCost + " NEXT: " + nextCost);

            float nextStateIntOldScore = visitedStates.getOrDefault(nextStateInt, -1.0f);            
            if (nextStateIntOldScore >= 0) {
            	// WE HAVE ALREADY REACHED THIS STATE
            	if (nextCost >= nextStateIntOldScore) {
                    // AND WE DO NOT HAVE BETTER SCORE
                    continue;
                }
            }
            
            if (bestNodeCost > nextCost) {
            	bestNode = current;
            	bestNodeCost = nextCost;
            }
            
            // NEW STATE or BETTER STATE
            visitedStates.put(nextStateInt, nextCost);
            
            List<MarioAction> actions = Helper.getPossibleActions(nextState);
            for (MarioAction action : actions) {
                //if (action == MarioAction.JUMP_RIGHT_SPEED)
                //    opened.add(getNewNode(nextState, current, nextCost + 1, action));
                //else
                    opened.add(getNewNode(nextState, current, nextCost, action));
            }

            if (nextState.getGameStatusCode() == 1) {
                bestNode = current;
                System.out.print("WIN FOUND ");
                winFound = true;
                break;
            }
        }

        ArrayList<boolean[]> actionsList = new ArrayList<>();

        SearchNode curr = bestNode;

        //actionsList.add(curr.marioAction.value);

        while (curr.parent != null) {
            for (int i = 0; i < searchSteps; i++) {
                actionsList.add(curr.marioAction.value);
            }
            curr = curr.parent;
        }

        //if (winFound)
        //    actionsList.add(0, )

//        System.out.println("ITERATIONS: " + iterations + " | Best X: " + bestNode.state.getMarioX()
//            + " | Number of actions: " + actionsList.size());

        return actionsList;

        //System.out.println("ITERATIONS: " + iterations + " | Best X: " + bestNode.state.getMarioX());

        /*SearchNode curr = bestNode;
        SearchNode prev = curr.parent;

        if (prev == null) {
        	return MarioAction.NO_ACTION.value;
        }
        
        while (prev.parent != null) {
        	curr = prev;
        	prev = prev.parent;        	
        }

        return curr.marioAction.value;*/
    }
}
