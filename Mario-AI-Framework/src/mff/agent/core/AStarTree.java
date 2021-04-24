package mff.agent.core;

import java.util.*;

import mff.agent.helper.MarioTimerSlim;
import mff.forwardmodel.slim.core.MarioForwardModelSlim;
import mff.forwardmodel.slim.core.MarioWorldSlim;

public class AStarTree {
    public SearchNode bestPosition;
    public SearchNode furthestPosition;
    float currentSearchStartingMarioXPos;
    //ArrayList<SearchNode> posPool;
    PriorityQueue<SearchNode> posPool;
    LinkedHashSet<Integer> visitedStates = new LinkedHashSet<>();
    //private byte[][][] visitedStates;
    //private byte searchNumber;
    //ArrayList<int[]> visitedStates = new ArrayList<>();

    private ArrayList<boolean[]> currentActionPlan;
    int ticksBeforeReplanning = 0;
    static boolean first = true;

    private static final boolean[] fastRightMovement = new boolean[] { false, true, false, false, true };

    private static SearchNode previousPos;

    public AStarTree(MarioForwardModelSlim model) {
        /*searchNumber = 1;
        this.visitedStates = new byte[(int) ((model.getLevelFloatDimensions()[0]) + 64) / 3][(int) ((model.getLevelFloatDimensions()[1] + 64) / 3)][100 / 5];*/
    }

    private void search(MarioTimerSlim timer) {
        SearchNode current = bestPosition;
        boolean currentGood = false;
        int maxRight = 176;
        while (posPool.size() != 0
                && ((bestPosition.sceneSnapshot.getMarioX() - currentSearchStartingMarioXPos < maxRight) || !currentGood)
                && timer.getRemainingTime() > 0) {
            System.out.println(posPool.size());
            current = pickBestPos(posPool);
            if (current == null) {
                return;
            }
            currentGood = false;
            float realRemainingTime = current.simulatePos();

            if (realRemainingTime < 0) {
                continue;
            } else if (!current.isInVisitedList && isInVisited((int) current.sceneSnapshot.getMarioX(),
                    (int) current.sceneSnapshot.getMarioY(), current.timeElapsed)) {
                realRemainingTime += Helper.visitedListPenalty;
                current.isInVisitedList = true;
                current.remainingTime = realRemainingTime;
                current.remainingTimeEstimated = realRemainingTime;
                current.calculateCost();
                posPool.add(current);
            } else if (realRemainingTime - current.remainingTimeEstimated > 0.1) {
                // current item is not as good as anticipated. put it back in pool and look for best again
                current.remainingTimeEstimated = realRemainingTime;
                current.calculateCost();
                posPool.add(current);
            } else {
                currentGood = true;
                visited((int) current.sceneSnapshot.getMarioX(), (int) current.sceneSnapshot.getMarioY(), current.timeElapsed);
                posPool.addAll(current.generateChildren());
            }
            if (currentGood) {
                if (bestPosition.getRemainingTime() > current.getRemainingTime())
                    bestPosition = current;
                if (current.sceneSnapshot.getMarioX() > furthestPosition.sceneSnapshot.getMarioX())
                    furthestPosition = current;
            }
            currentGood = false;
            if (bestPosition.sceneSnapshot.getGameStatusCode() == MarioWorldSlim.WIN ||
            furthestPosition.sceneSnapshot.getGameStatusCode() == MarioWorldSlim.WIN) {
                System.out.println("WIN FOUND");
                return;
            }
        }
        if (current.sceneSnapshot.getMarioX() - currentSearchStartingMarioXPos < maxRight
                && furthestPosition.sceneSnapshot.getMarioX() > bestPosition.sceneSnapshot.getMarioX() + 20)
            // Couldnt plan till end of screen, take furthest
            bestPosition = furthestPosition;
    }

    public void startSearch(MarioForwardModelSlim model, int repetitions) {
        SearchNode startPos = new SearchNode(null, repetitions, null);
        startPos.initializeRoot(model);

        posPool = new PriorityQueue<>(new CompareByCost());
        visitedStates.clear();
        /*searchNumber++;
        if (searchNumber == 0) {
            System.out.println("resetting"); //TODO
            for (byte[][] row : visitedStates) {
                for (byte[] depth : row) {
                    Arrays.fill(depth, (byte) 0);
                }
            }
        }*/

        posPool.addAll(startPos.generateChildren());
        currentSearchStartingMarioXPos = model.getMarioX();

        bestPosition = startPos;
        furthestPosition = startPos;
    }

    private ArrayList<boolean[]> extractPlan() {
        ArrayList<boolean[]> actions = new ArrayList<>();

        // just move forward if no best position exists
        if (bestPosition == null) {
            for (int i = 0; i < 10; i++) {
                actions.add(new boolean[] {false, false, false, false, false});
            }
            return actions;
        }

        SearchNode current = bestPosition;
        while (current.parentPos != null) {
            if (current.parentPos == previousPos)
                break;
            for (int i = 0; i < current.repetitions; i++)
                actions.add(0, current.action);
            current = current.parentPos;
        }
        previousPos = bestPosition;
        return actions;
    }

    private SearchNode pickBestPos(PriorityQueue<SearchNode> posPool) {
        SearchNode bestPos = posPool.peek();
        if (bestPos != null)
            posPool.remove(bestPos);
        return bestPos;

        /*
        SearchNode bestPos = null;
        float bestPosCost = 10000000;
        for (SearchNode current : posPool) {
            float currentCost = current.getRemainingTime() + current.timeElapsed * 0.9f; // slightly bias towards furthest positions
            if (currentCost < bestPosCost) {
                bestPos = current;
                bestPosCost = currentCost;
            }
        }
        posPool.remove(bestPos);
        return bestPos;*/
    }

    public boolean[] optimise(MarioForwardModelSlim model, MarioTimerSlim timer) {
        if (first) {
            search(new MarioTimerSlim(1000000));
            currentActionPlan = extractPlan();
            first = false;
        }

        return currentActionPlan.remove(0);

        /*
        ticksBeforeReplanning--;
        if (ticksBeforeReplanning <= 0 || currentActionPlan.size() == 0) {
            currentActionPlan = extractPlan();
            ticksBeforeReplanning = 3; //TODO
        }

        MarioForwardModelSlim originalModel = model.clone();
        for (int i = 0; i < currentActionPlan.size(); i++) {
            model.advance(currentActionPlan.get(i));
        }
        if (model.getGameStatusCode() == MarioWorldSlim.LOSE) {
            startSearch(originalModel, 2);
        }
        search(timer);

        boolean[] action = new boolean[5];
        if (currentActionPlan.size() > 0)
            action = currentActionPlan.remove(0);
        return action;
*/
        /*
        int planAhead = 3;
        int stepsPerSearch = 2;
        MarioForwardModelSlim originalModel = model.clone();
        ticksBeforeReplanning--;
        if (ticksBeforeReplanning <= 0 || currentActionPlan.size() == 0) {
            currentActionPlan = extractPlan();
            if (currentActionPlan.size() < planAhead) {
                planAhead = currentActionPlan.size();
            }

            // simulate ahead to predicted future state, and then plan for this future state
            for (int i = 0; i < planAhead; i++) {
                model.advance(currentActionPlan.get(i));
            }
            startSearch(model, stepsPerSearch);
            ticksBeforeReplanning = 3;
        }
        if (model.getGameStatusCode() == MarioWorldSlim.LOSE) {
            startSearch(originalModel, stepsPerSearch);
        }
        search(timer);

        boolean[] action = new boolean[5];
        if (currentActionPlan.size() > 0)
            action = currentActionPlan.remove(0);
        return action;*/
    }

    private void visited(int x, int y, int t) {
       visitedStates.add((x << 18) | (y << 8));
    }

    private boolean isInVisited(int x, int y, int t) {
        return visitedStates.contains((x << 18) | (y << 8));
    }

    /*private void visited(int x, int y, int t) {
        visitedStates.add(new int[]{x, y, t});
    }

    private boolean isInVisited(int x, int y, int t) {
        int timeDiff = 5;
        int xDiff = 2;
        int yDiff = 2;
        for (int[] v : visitedStates) {
            if (Math.abs(v[0] - x) < xDiff && Math.abs(v[1] - y) < yDiff && Math.abs(v[2] - t) < timeDiff
                    && t >= v[2]) {
                return true;
            }
        }
        return false;
    }*/

/*
    private void visited(int x, int y, int t) {
        try {
            visitedStates[x / 3][y / 3][t / 5] = searchNumber;
        }
        catch (IndexOutOfBoundsException ignored) {
        }
        //visitedStates.add((x << 18) | (y << 8) | t);
    }

    // TODO: one int, x = 14 bits, y = 10 bits, t = 8 bits
    // TODO: can we restore considering similar states equal?
    // TODO: index out of bounds exception for index -3 for length 320, probably "y"
    private boolean isInVisited(int x, int y, int t) {
        try {
            return visitedStates[x / 3][y / 3][t / 5] == searchNumber;
        }
        catch (IndexOutOfBoundsException e) {
            return false; // TODO: better solution
        }

        /*return visitedStates.contains((x << 18) | (y << 8) | t - 4) ||
        visitedStates.contains((x << 18) | (y << 8) | t - 3) ||
        visitedStates.contains((x << 18) | (y << 8) | t - 2) ||
        visitedStates.contains((x << 18) | (y << 8) | t - 1) ||
        visitedStates.contains((x << 18) | (y << 8) | t) ||
        visitedStates.contains((x << 18) | (y << 8) | t + 1) ||
        visitedStates.contains((x << 18) | (y << 8) | t + 2) ||
        visitedStates.contains((x << 18) | (y << 8) | t + 3) ||
        visitedStates.contains((x << 18) | (y << 8) | t + 4)
        ||
         visitedStates.contains((x - 1 << 18) | (y - 1 << 8) | t) ||
         visitedStates.contains((x - 1 << 18) | (y << 8) | t) ||
         visitedStates.contains((x - 1 << 18) | (y + 1 << 8) | t) ||
         visitedStates.contains((x << 18) | (y - 1 << 8) | t) ||
         //visitedStates.contains((x << 18) | (y << 8) | t) ||
         visitedStates.contains((x << 18) | (y + 1 << 8) | t) ||
         visitedStates.contains((x + 1 << 18) | (y - 1 << 8) | t) ||
         visitedStates.contains((x + 1 << 18) | (y << 8) | t) ||
         visitedStates.contains((x + 1 << 18) | (y + 1 << 8) | t);
    }*/
}
