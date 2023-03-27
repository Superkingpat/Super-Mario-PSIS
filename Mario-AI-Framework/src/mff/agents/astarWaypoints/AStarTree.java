package mff.agents.astarWaypoints;

import mff.agents.astarHelper.CompareByCost;
import mff.agents.astarHelper.Helper;
import mff.agents.astarHelper.MarioAction;
import mff.agents.astarHelper.SearchNode;
import mff.agents.common.MarioTimerSlim;
import mff.forwardmodel.slim.core.MarioForwardModelSlim;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.PriorityQueue;

public class AStarTree {
    private final int[][] levelTilesWithPath;

    public static final int WAYPOINT_DENSITY = 16;
    public static final int WAYPOINT_DISTANCE_TOLERANCE = 16;
    public static ArrayList<Waypoint> waypoints = new ArrayList<>();

    public Waypoint currentGoalWaypoint = waypoints.get(0);
    public int currentGoalWaypointIndex = 0;

    public SearchNode furthestNodeTowardsWaypoint;
    public float furthestNodeTowardsWaypointDistanceFromWaypoint;

    float marioXStart;
    int searchSteps;

    static boolean winFound = false;
    static final float maxMarioSpeedX = 10.91f;
    static float exitTileX;

    public int nodesEvaluated = 0;
    public int mostBacktrackedNodes = 0;
    private int farthestReachedX;
    private int nodesBeforeNewFarthestX = 0;

    public static float NODE_DEPTH_WEIGHT = 1f;
    public static float TIME_TO_FINISH_WEIGHT = 1.5f;
    public static float DISTANCE_FROM_PATH_TOLERANCE = 1;
    public static float DISTANCE_FROM_PATH_ADDITIVE_PENALTY = 0;
    public static float DISTANCE_FROM_PATH_MULTIPLICATIVE_PENALTY = 7;

    PriorityQueue<SearchNode> opened = new PriorityQueue<>(new CompareByCost());
    /**
     * INT STATE -> STATE COST
     */
    HashMap<Integer, Float> visitedStates = new HashMap<>();

    public AStarTree(MarioForwardModelSlim startState, int searchSteps, int[][] levelTilesWithPath) {
    	this.searchSteps = searchSteps;
        this.levelTilesWithPath = levelTilesWithPath;

    	marioXStart = startState.getMarioX();

    	furthestNodeTowardsWaypoint = getStartNode(startState);
    	furthestNodeTowardsWaypoint.cost = calculateCost(startState, furthestNodeTowardsWaypoint.nodeDepth);
        furthestNodeTowardsWaypointDistanceFromWaypoint = Math.abs(currentGoalWaypoint.x - furthestNodeTowardsWaypoint.state.getMarioX())
            + Math.abs(currentGoalWaypoint.y - furthestNodeTowardsWaypoint.state.getMarioY());

        farthestReachedX = (int) furthestNodeTowardsWaypoint.state.getMarioX();
    	
    	opened.add(furthestNodeTowardsWaypoint);
    }
    
    private int getIntState(MarioForwardModelSlim model) {
    	return getIntState((int) model.getMarioX(), (int) model.getMarioY());
    }
    
    private int getIntState(int x, int y) {
    	return (x << 16) | y;
    }
    
    private SearchNode getStartNode(MarioForwardModelSlim state) {
    	return new SearchNode(state);
    }
    
    private SearchNode getNewNode(MarioForwardModelSlim state, SearchNode parent, float cost, MarioAction action) {
    	return new SearchNode(state, parent, cost, action);
    }
    
    private float calculateCost(MarioForwardModelSlim nextState, int nodeDepth) {
//        float timeToFinish = (exitTileX - nextState.getMarioX()) / maxMarioSpeedX;
        float timeToFinish = (Math.abs(currentGoalWaypoint.x - nextState.getMarioX())
                + Math.abs(currentGoalWaypoint.y - nextState.getMarioY()))
                / maxMarioSpeedX;
        float distanceFromGridPathCost = calculateDistanceFromGridPathCost(nextState);
        return NODE_DEPTH_WEIGHT * nodeDepth
                + TIME_TO_FINISH_WEIGHT * timeToFinish
                + distanceFromGridPathCost;
	}

    private float calculateDistanceFromGridPathCost(MarioForwardModelSlim nextState) {
        int distanceFromGridPath = calculateDistanceFromGridPath(nextState);
        if (distanceFromGridPath <= DISTANCE_FROM_PATH_TOLERANCE)
            return 0;
        else
            return (distanceFromGridPath - DISTANCE_FROM_PATH_TOLERANCE)
                    * DISTANCE_FROM_PATH_MULTIPLICATIVE_PENALTY
                    + DISTANCE_FROM_PATH_ADDITIVE_PENALTY;
    }

    private int calculateDistanceFromGridPath(MarioForwardModelSlim nextState) {
        int marioTileX = (int) (nextState.getMarioX() / 16);
        int marioTileY = (int) (nextState.getMarioY() / 16);
        if (marioTileX >= 0 && marioTileX < levelTilesWithPath.length &&
            marioTileY >= 0 && marioTileY < levelTilesWithPath[0].length)
                if (levelTilesWithPath[marioTileX][marioTileY] == 1)
                    return 0;

        int distance = 1;
        do {
            // upper and lower side
            for (int x = marioTileX - distance; x <= marioTileX + distance; x++) {
                if (x < 0 || x >= levelTilesWithPath.length)
                    continue;

                int y = marioTileY - distance;
                if (y >= 0 && y < levelTilesWithPath[0].length)
                    if (levelTilesWithPath[x][y] == 1)
                        return distance;

                y = marioTileY + distance;
                if (y >= 0 && y < levelTilesWithPath[0].length)
                    if (levelTilesWithPath[x][y] == 1)
                        return distance;
            }
            // left and right side
            for (int y = marioTileY - distance + 1; y <= marioTileY + distance - 1; y++) {
                if (y < 0 || y >= levelTilesWithPath[0].length)
                    continue;

                int x = marioTileX - distance;
                if (x >= 0 && x < levelTilesWithPath.length)
                    if (levelTilesWithPath[x][y] == 1)
                        return distance;

                x = marioTileX + distance;
                if (x >= 0 && x < levelTilesWithPath.length)
                    if (levelTilesWithPath[x][y] == 1)
                        return distance;
            }
            distance++;
        } while (distance < 64);
        throw new IllegalStateException("Something seems wrong, distance to grid path shouldn't be this big.");
    }

    public ArrayList<boolean[]> search(MarioTimerSlim timer) {
        while (opened.size() > 0 && timer.getRemainingTime() > 0) {
            SearchNode current = opened.remove();
            nodesEvaluated++;

            if ((int) current.state.getMarioX() > farthestReachedX) {
                mostBacktrackedNodes = Math.max(nodesBeforeNewFarthestX, mostBacktrackedNodes);
                farthestReachedX = (int) current.state.getMarioX();
                nodesBeforeNewFarthestX = 0;
            } else {
                nodesBeforeNewFarthestX++;
            }

            if (Math.abs(currentGoalWaypoint.x - current.state.getMarioX()) + Math.abs(currentGoalWaypoint.y - current.state.getMarioY())
                < furthestNodeTowardsWaypointDistanceFromWaypoint) {
                    furthestNodeTowardsWaypoint = current;
                    furthestNodeTowardsWaypointDistanceFromWaypoint = Math.abs(currentGoalWaypoint.x - current.state.getMarioX());
            }

            if (current.state.getGameStatusCode() == 1) {
                furthestNodeTowardsWaypoint = current;
                winFound = true;
                break;
            }

            if ((Math.abs(currentGoalWaypoint.x - current.state.getMarioX())
                + (Math.abs(currentGoalWaypoint.y - current.state.getMarioY())))
                <= WAYPOINT_DISTANCE_TOLERANCE) {
                    if (currentGoalWaypointIndex != waypoints.size() - 1) {
                        deleteGridPathToCurrentWaypoint();
                        currentGoalWaypointIndex++;
                        currentGoalWaypoint = waypoints.get(currentGoalWaypointIndex);
                        furthestNodeTowardsWaypoint = current;
                        furthestNodeTowardsWaypointDistanceFromWaypoint = Math.abs(currentGoalWaypoint.x - current.state.getMarioX())
                                                                        + Math.abs(currentGoalWaypoint.y - current.state.getMarioY());
                }
            }

            ArrayList<MarioAction> actions = Helper.getPossibleActions(current.state);
            for (MarioAction action : actions) {
                MarioForwardModelSlim newState = current.state.clone();

                for (int i = 0; i < searchSteps; i++) {
                    newState.advance(action.value);
                }

                if (!newState.getWorld().mario.alive)
                    continue;

                float newStateCost = calculateCost(newState, current.nodeDepth + 1);

                int newStateCode = getIntState(newState);
                float newStateOldScore = visitedStates.getOrDefault(newStateCode, -1.0f);
                if (newStateOldScore >= 0 && newStateCost >= newStateOldScore)
                    continue;

                visitedStates.put(newStateCode, newStateCost);
                opened.add(getNewNode(newState, current, newStateCost, action));
            }
        }

        ArrayList<boolean[]> actionsList = new ArrayList<>();

        SearchNode curr = furthestNodeTowardsWaypoint;

        while (curr.parent != null) {
            for (int i = 0; i < searchSteps; i++) {
                actionsList.add(curr.marioAction.value);
            }
            curr = curr.parent;
        }

        return actionsList;
    }

    private void deleteGridPathToCurrentWaypoint() {

    }

    static class Waypoint {
        public int x;
        public int y;

        public Waypoint(int x, int y) {
            this.x = x;
            this.y = y;
        }
    }
}
