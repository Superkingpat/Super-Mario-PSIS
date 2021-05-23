package mff.agents.robinBaumgartenSlimImproved;

import java.util.*;

import mff.agents.common.MarioTimerSlim;
import mff.forwardmodel.slim.core.MarioForwardModelSlim;
import mff.forwardmodel.slim.core.MarioWorldSlim;

public class AStarTree {
    public SearchNode bestPosition;
    public SearchNode furthestPosition;
    float currentSearchStartingMarioXPos;
    PriorityQueue<SearchNode> posPool;
    LinkedHashSet<Integer> visitedStates = new LinkedHashSet<>();

    private boolean[] currentAction;
    int ticksBeforeReplanning = 0;

    public static float exitTileX;
    public static final float maxMarioSpeedX = 10.91f;

    public int nodesEvaluated = 0;

    private static final boolean[] fastRightMovement = new boolean[] { false, true, false, false, true };

    private void search(MarioTimerSlim timer) {
        SearchNode current = bestPosition;
        boolean currentGood = false;
        int maxRight = 176;
        while (posPool.size() != 0
                && ((bestPosition.sceneSnapshot.getMarioX() - currentSearchStartingMarioXPos < maxRight) || !currentGood)
                && timer.getRemainingTime() > 0) {
            current = pickBestPos(posPool);
            nodesEvaluated++;
            if (current == null) {
                return;
            }
            currentGood = false;
            float realRemainingTime = current.remainingTime;

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
        }
        if (current.sceneSnapshot.getMarioX() - currentSearchStartingMarioXPos < maxRight
                && furthestPosition.sceneSnapshot.getMarioX() > bestPosition.sceneSnapshot.getMarioX() + 20)
            // Couldnt plan till end of screen, take furthest
            bestPosition = furthestPosition;
    }

    private void startSearch(MarioForwardModelSlim model, int repetitions) {
        SearchNode startPos = new SearchNode(null, repetitions, null);
        startPos.initializeRoot(model);

        posPool = new PriorityQueue<>((o1, o2) -> Float.compare(o1.cost, o2.cost));
        visitedStates.clear();

        posPool.addAll(startPos.generateChildren());
        currentSearchStartingMarioXPos = model.getMarioX();

        bestPosition = startPos;
        furthestPosition = startPos;
    }

    private boolean[] extractPlan() {
        // just move forward if no best position exists
        if (bestPosition == null)
            return fastRightMovement;

        boolean[] result = null;
        SearchNode current = bestPosition;
        while (current.parentPos != null) {
            result = current.action;
            current = current.parentPos;
        }
        return (result != null) ? result : fastRightMovement;
    }

    private SearchNode pickBestPos(PriorityQueue<SearchNode> posPool) {
        SearchNode bestPos = posPool.peek();
        if (bestPos != null)
            posPool.remove(bestPos);
        return bestPos;
    }

    public boolean[] optimise(MarioForwardModelSlim model, MarioTimerSlim timer) {
        int stepsPerSearch = 3;

        startSearch(model, stepsPerSearch);

        search(timer);

        currentAction = extractPlan();

        return currentAction;
    }

    private void visited(int x, int y, int t) {
        visitedStates.add((x << 18) | (y << 8) | t);
    }

    private boolean isInVisited(int x, int y, int t) {
        return visitedStates.contains((x << 18) | (y << 8) | t);
    }
}
